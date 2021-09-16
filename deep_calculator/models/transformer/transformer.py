from typing import Optional

import einops
import pytorch_lightning as pl
import torch
from torch import nn
from torchtyping import TensorType
from typeguard import typechecked

# This line does nothing. I use it for shutting down flake8
batch = None
kv_time = None
feature = None
q_time = None
left_time = None
dv = None
shifted_time = None
vocab = None
output_time = None


class SelfAttention(pl.LightningModule):

    @typechecked
    def __init__(self, feature_size: int, dk: int, dv: int):
        super(SelfAttention, self).__init__()
        self.keys_projection = nn.Linear(feature_size, dk)
        self.values_projection = nn.Linear(feature_size, dv)
        self.query_projection = nn.Linear(feature_size, dk)
        self.register_buffer("sqrt_dk",
                             torch.sqrt(torch.tensor(dk, device=self.device)))

    @typechecked
    def forward(
        self,
        keys: TensorType["batch", "kv_time", "feature"],
        values: TensorType["batch", "kv_time", "feature"],
        queries: TensorType["batch", "q_time", "feature"],
        mask: Optional[TensorType[-1]] = None,
    ) -> TensorType["batch", "q_time", "dv"]:
        keys = self.keys_projection(keys)
        values = self.values_projection(values)
        queries = self.query_projection(queries)

        scores = torch.einsum('bqf,bkf->bqk', queries, keys)
        scores = scores / self.sqrt_dk

        key_batch, key_time, key_feature = keys.shape
        q_batch, q_time, q_feature = queries.shape
        assert key_batch == q_batch
        assert key_feature == q_feature
        res_batch, res_time, res_feature = scores.shape
        assert res_batch == key_batch
        assert res_time == q_time
        assert res_feature == key_time

        if mask is not None:
            scores[:, mask, :] = -1e9
        weights = torch.softmax(scores, dim=2)
        result = torch.einsum('bqk,bkv->bqv', weights, values)
        return result


class MultiheadAttention(pl.LightningModule):

    @typechecked
    def __init__(self, feature_size: int, dk: int, dv: int, n_heads: int = 2):
        super(MultiheadAttention, self).__init__()
        self.heads = nn.ModuleList(
            [SelfAttention(feature_size, dk, dv) for _ in range(n_heads)])
        self.linear = nn.Linear(dk * n_heads, feature_size)

    @typechecked
    def forward(
        self,
        keys: TensorType["batch", "kv_time", "feature"],
        values: TensorType["batch", "kv_time", "feature"],
        queries: TensorType["batch", "q_time", "feature"],
        mask: Optional[TensorType[-1]] = None,
    ) -> TensorType["batch", "q_time", "feature"]:
        results = [head(keys, values, queries, mask) for head in self.heads]
        results = torch.cat(results, dim=2)
        return self.linear(results)


class Embedder(pl.LightningModule):

    def __init__(self, vocab_size, feature_size, total_time, dropout=0.1):
        super(Embedder, self).__init__()
        self.linear = nn.Linear(vocab_size, feature_size)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "positions",
            torch.arange(total_time, dtype=torch.float).reshape(
                (1, total_time)))
        self.linear_to_positions = nn.Linear(total_time, feature_size)
        self.feature_size = feature_size

    def forward(self, inputs):
        embeddings = self.linear(inputs)
        embeddings = nn.functional.relu(embeddings)
        positions = self.linear_to_positions(self.positions).reshape(
            (1, 1, self.feature_size))
        return self.dropout(embeddings + positions)


class EncoderLayer(pl.LightningModule):

    @typechecked
    def __init__(self,
                 feature_size: int,
                 total_time: int,
                 inner_size: int = 1024,
                 dropout: float = 0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiheadAttention(feature_size, feature_size // 2,
                                            feature_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.layernorm_first = nn.LayerNorm([total_time, feature_size])
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_size, inner_size),
            nn.ReLU(),
            nn.Linear(inner_size, feature_size),
        )
        self.layernorm_second = nn.LayerNorm([total_time, feature_size])

    @typechecked
    def forward(
        self, left: TensorType["batch", "left_time", "feature"]
    ) -> TensorType["batch", "left_time", "feature"]:
        x = self.attention(left, left, left)
        x = self.dropout(x)
        x = self.layernorm_first(x + left)
        y = self.feed_forward(x)
        y = self.dropout(y)
        y = self.layernorm_second(y + x)
        return y


class Encoder(pl.LightningModule):

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        feature_size: int,
        total_time: int,
        n_layers: int,
        inner_size: int = 1024,
        dropout: float = 0.1,
    ):
        super(Encoder, self).__init__()
        self.encoders = nn.ModuleList([
            EncoderLayer(feature_size, total_time, inner_size, dropout)
            for _ in range(n_layers)
        ])

    @typechecked
    def forward(
        self, left: TensorType["batch", "left_time", "feature"]
    ) -> TensorType["batch", "left_time", "feature"]:
        for encoder in self.encoders:
            left = encoder(left)
        return left


class DecoderLayer(pl.LightningModule):

    @typechecked
    def __init__(
        self,
        total_time: int,
        feature_size: int,
        inner_size: int = 1024,
        dropout: float = 0.1,
    ):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiheadAttention(feature_size,
                                                 feature_size // 2,
                                                 feature_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.layernorm_1 = nn.LayerNorm([total_time, feature_size])
        self.context_attention = MultiheadAttention(feature_size,
                                                    feature_size // 2,
                                                    feature_size // 2)
        self.layernorm_2 = nn.LayerNorm([total_time, feature_size])
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_size, inner_size),
            nn.ReLU(),
            nn.Linear(inner_size, feature_size),
            nn.Dropout(dropout),
        )
        self.layernorm_3 = nn.LayerNorm([total_time, feature_size])

    @typechecked
    def forward(
        self,
        x: TensorType["batch", "shifted_time", "feature"],
        encodings: TensorType["batch", "left_time", "feature"],
        mask: Optional[TensorType[-1]] = None,
    ) -> TensorType["batch", "shifted_time", "feature"]:
        y = self.self_attention(x, x, x, mask=mask)
        y = self.dropout(y)
        y = self.layernorm_1(y + x)
        z = self.context_attention(encodings, encodings, y)
        z = self.dropout(z)
        z = self.layernorm_2(z + y)
        y = self.feed_forward(z)
        y = self.layernorm_3(y + z)
        return y


class Decoder(pl.LightningModule):

    @typechecked
    def __init__(self,
                 total_time: int,
                 vocab_size: int,
                 feature_size: int,
                 n_layers: int,
                 start_symbol: TensorType[1, "vocab"],
                 inner_size=1024,
                 dropout=0.1):
        super(Decoder, self).__init__()
        self.decoders = nn.ModuleList([
            DecoderLayer(total_time, feature_size, inner_size, dropout)
            for _ in range(n_layers)
        ])
        self.to_vocab = nn.Linear(total_time * feature_size, vocab_size)  # ?
        self.total_time = total_time

    @typechecked
    def forward(
        self,
        encodings: TensorType["batch", "left_time", "feature"],
        shifted: TensorType["batch", "shifted_time", "feature"],
    ) -> TensorType["batch", "vocab", "output_time"]:
        batch_size, time_size, vocab_size = shifted.shape
        results = []
        for time_step in range(time_size):
            mask = self.generate_mask(time_step)
            x = shifted
            for decoder in self.decoders:
                x = decoder(x, encodings, mask)
            x = einops.rearrange(x, 'b t f -> b (t f)')
            results.append(self.to_vocab(x))
        return einops.rearrange(results, 't b f -> b f t')

    @typechecked
    def predict(
        self, encodings: TensorType["batch", "left_time", "feature"]
    ) -> TensorType["batch", "vocab", "output_time"]:
        batch_size = encodings.shape[0]
        last_output = einops.repeat(self.start_symbol,
                                    'b f -> (repeat b) f',
                                    repeat=batch_size)
        last_output = einops.repeat(last_output,
                                    'b f -> b t f',
                                    t=self.total_time)
        results = []
        for time in range(self.total_time - 1):
            mask = self.generate_mask(time)
            x = self.embedder(last_output)
            for decoder in self.decoders:
                x = decoder(x, encodings, mask)
            x = einops.rearrange(x, 'b t f -> b (t f)')
            x = self.to_vocab(x)
            results.append(x)
            x = torch.softmax(x, dim=1)
            last_output[:, time + 1, :] = x

        return einops.rearrange(results, 't b f -> b f t')

    @typechecked
    def generate_mask(
        self,
        time_step: int,
    ) -> TensorType["shifted_time"]:
        return torch.arange(0, 1 + time_step, device=self.device)


class Transformer(pl.LightningModule):

    @typechecked
    def __init__(
        self,
        total_time: int,
        vocab_size: int,
        feature_size: int,
        n_encoders: int,
        n_decoders: int,
        learning_rate: float,
        start_symbol: TensorType[1, "vocab"],
        inner_size=1024,
        dropout=0.1,
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(vocab_size, feature_size, total_time, n_encoders,
                               inner_size, dropout)
        self.decoder = Decoder(total_time, vocab_size, feature_size, n_decoders,
                               start_symbol, inner_size, dropout)
        self.embedder = Embedder(vocab_size, feature_size, total_time)
        self.register_buffer('start_symbol', start_symbol)

        self.criterion = nn.functional.cross_entropy
        self.save_hyperparameters()

    @typechecked
    def forward(
        self,
        left: TensorType["batch", "left_time", "vocab"],
        shifted: TensorType["batch", "shifted_time", "vocab"],
    ) -> TensorType["batch", "vocab", "output_time"]:
        left = self.embedder(left)
        encodings = self.encoder(left)
        shifted = self.embedder(shifted)
        predictions = self.decoder(encodings, shifted)
        return predictions

    @typechecked
    def predict(
        self, left: TensorType["batch", "left_time", "vocab"]
    ) -> TensorType["batch", "vocab", "output_time"]:
        left = self.embedder(left)
        encodings = self.encoder(left)

        batch_size = encodings.shape[0]
        last_output = einops.repeat(self.start_symbol,
                                    'b f -> (repeat b) f',
                                    repeat=batch_size)
        last_output = einops.repeat(last_output,
                                    'b f -> b t f',
                                    t=self.total_time)
        result = self.decoder.predict(encodings)
        return result

    def validation_step(self, batch, batch_idx):
        left, right = batch
        y_pred = self.predict(left)
        loss = self.criterion(y_pred, right)
        self.log('val_loss', loss, prog_bar=True)

    def training_step(self, batch, batch_idx):
        left, shifted, right = batch
        y_pred = self.forward(left, shifted)
        loss = self.criterion(y_pred, right)
        self.log('train_loss',
                 loss,
                 prog_bar=True,
                 on_epoch=True,
                 on_step=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               verbose=True,
                                                               cooldown=50,
                                                               factor=0.5,
                                                               patience=40)
        lr_dict = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'monitor': 'train_loss',
        }
        return [optimizer], [lr_dict]
