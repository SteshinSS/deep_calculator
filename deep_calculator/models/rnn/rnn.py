from typing import Tuple

import pytorch_lightning as pl
import torch
from torch import nn
from torch.functional import Tensor
from torchtyping import TensorType
from typeguard import typechecked
import einops

batch = None
left_time = None
feature = None
hidden = None
output_time = None
shifted_time = None


@typechecked
class Encoder(pl.LightningModule):

    def __init__(
        self, vocab_size: int, hidden_size: int, recurrent_dropout: float
    ):
        super(Encoder, self).__init__()
        self.rnn_first = nn.GRUCell(vocab_size, hidden_size)
        self.dropout = nn.Dropout(recurrent_dropout)
        self.hidden_size = hidden_size

    def forward(
        self, inputs: TensorType["batch", "left_time", "feature"]
    ) -> TensorType["left_time", "batch", "hidden"]:
        batch_size, time_size, feature_size = inputs.shape
        inputs = einops.rearrange(inputs, 'b t f -> t b f')
        hidden_states = []
        first_states = None
        for time in range(time_size):
            first_states = self.rnn_first(inputs[time], first_states)
            hidden_states.append(first_states)
            first_states = self.dropout(first_states)
        return torch.stack(hidden_states)


@typechecked
class Attention(pl.LightningModule):

    def __init__(self):
        super(Attention, self).__init__()

    def forward(
        self,
        query: TensorType["batch", "hidden"],
        hidden_states: TensorType["left_time", "batch", "hidden"],
    ) -> TensorType["batch", "hidden"]:
        score = torch.tensordot(query, hidden_states, dims=([0, 1], [1, 2]))
        weight = torch.softmax(score, dim=0)
        context = torch.einsum('i,ijk->jk', weight, hidden_states)
        return context


@typechecked
class Decoder(pl.LightningModule):

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        maximal_length: int,
        start_symbol: TensorType[1, "feature"],
        dropout: float,
        recurrent_dropout: float
    ):
        super(Decoder, self).__init__()
        self.rnn_first = nn.GRUCell(vocab_size, hidden_size)
        self.to_vocab = nn.Linear(2 * hidden_size, vocab_size)
        self.attention = Attention()

        self.dropout = nn.Dropout(dropout)
        self.recurrent_dropout = nn.Dropout(recurrent_dropout)
        self.maximal_length = maximal_length
        self.register_buffer("start_symbol", start_symbol)

    def forward(
        self,
        hidden_states: TensorType["left_time", "batch", "hidden"],
        shifted: TensorType["batch", "shifted_time", "feature"]
    ) -> TensorType["batch", "feature", "output_time"]:
        batch_size, time_size, feature_size = shifted.shape
        shifted = einops.rearrange(shifted, 'b t f -> t b f')
        decoder_state = None
        result = []
        for time in range(time_size):
            decoder_state = self.rnn_first(shifted[time], decoder_state)
            context = self.attention(decoder_state, hidden_states)
            context_and_state = torch.cat((context, decoder_state), dim=1)
            output = self.to_vocab(context_and_state)
            result.append(output)
            decoder_state = self.dropout(decoder_state)
        result = einops.rearrange(result, 't b f -> b f t')
        return result

    def predict(
        self, hidden_states: TensorType["left_time", "batch", "hidden"]
    ) -> TensorType["batch", "feature", "output_time"]:
        time_size, batch_size, hidden_size = hidden_states.shape
        last_output = einops.repeat(
            self.start_symbol, 'b f -> (repeat b) f', repeat=batch_size
        )
        decoder_state = None
        result = []
        for time in range(self.maximal_length):
            decoder_state = self.rnn_first(last_output, decoder_state)
            context = self.attention(decoder_state, hidden_states)
            context_and_state = torch.cat((context, decoder_state), dim=1)
            last_output = self.to_vocab(context_and_state)
            result.append(last_output)
            last_output = nn.functional.softmax(last_output, dim=1)
        result = einops.rearrange(result, 't b f -> b f t')
        return result


@typechecked
class Seq2Seq(pl.LightningModule):

    def __init__(
        self,
        vocab_size: int,
        maximal_length: int,
        start_symbol: TensorType[1, "feature"],
        hidden_size: int,
        recurrent_dropout: float,
        dropout: float,
        learning_rate: float
    ):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(vocab_size, hidden_size, recurrent_dropout)
        self.decoder = Decoder(
            vocab_size,
            hidden_size,
            maximal_length,
            start_symbol,
            dropout,
            recurrent_dropout
        )
        self.criterion = nn.functional.cross_entropy
        self.save_hyperparameters(
            ignore=['start_symbol', 'maximal_length', 'vocab_size']
        )

    def forward(
        self,
        left: TensorType["batch", "left_time", "feature"],
        shifted: TensorType["batch", "shifted_time", "feature"]
    ) -> TensorType["batch", "feature", "output_time"]:
        hidden_states = self.encoder(left)
        result = self.decoder(hidden_states, shifted)
        return result

    def predict(
        self, left: TensorType["batch", "left_time", "feature"]
    ) -> TensorType["batch", "feature", "output_time"]:
        hidden_states = self.encoder(left)
        result = self.decoder.predict(hidden_states)
        return result

    def training_step(self, batch, batch_idx):
        left, shifted, right = batch
        y_pred = self.forward(left, shifted)
        loss = self.criterion(y_pred, right)
        self.log(
            'train_loss', loss, prog_bar=True, on_epoch=True, on_step=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        left, right = batch
        y_pred = self.predict(left)
        loss = self.criterion(y_pred, right)
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, verbose=True, cooldown=50, factor=0.5, patience=40
        )
        lr_dict = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'monitor': 'train_loss',
        }
        return [optimizer], [lr_dict]

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
