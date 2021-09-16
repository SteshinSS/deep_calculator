import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
import yaml
from torchtyping import patch_typeguard

patch_typeguard()

import models
import data
import cli


def load_model(config, dataset_generator, args):
    start_symbol = dataset_generator.get_start_symbol()
    vocab_size = len(dataset_generator.alphabet)

    if config['name'] == 'rnn':
        model = models.rnn.Seq2Seq(
            vocab_size=vocab_size,
            maximal_length=11,
            start_symbol=start_symbol,
            hidden_size=config['hidden_size'],
            dropout=config['dropout'],
            recurrent_dropout=config['recurrent_dropout'],
            learning_rate=config['learning_rate']
        )
        model.init_weights()

    if not args.train:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
    return model


def get_prediction(model, prompt, dataset_generator):
    vectorized_prompt = dataset_generator.vectorize_dataset([prompt])
    result = model.predict(vectorized_prompt)
    result = result[0]  # first sample in batch
    result = torch.argmax(result, dim=0).tolist()
    return dataset_generator.idx2str(result)


def open_config(config_path):
    with config_path.open() as config_file:
        return yaml.safe_load(config_file)


def train_model(model, dataloaders, epochs, checkpoint_folder):
    if checkpoint_folder == cli.DEFAULT_CHECKPOINT_PATH:
        resume_from_checkpoint = None
        default_root_dir = checkpoint_folder
    else:
        resume_from_checkpoint = checkpoint_folder
        default_root_dir = None

    wandb_logger = WandbLogger(
        log_model=True,
        project='deep_calculator',
        save_code=True,
    )
    wandb_logger.watch(model)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=epochs,
        logger=wandb_logger,
        deterministic=True,
        default_root_dir=default_root_dir,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    trainer.fit(model, dataloaders['train'], dataloaders['val'])


if __name__ == '__main__':
    args = cli.parse_args()
    config = open_config(args.config)

    dataset_generator = data.DatasetGenerator(config['data']['max_digits'])
    dataloaders = data.get_dataloaders(config['data'], dataset_generator)

    model = load_model(config['model'], dataset_generator, args)

    if args.train:
        train_model(model, dataloaders, args.epochs, args.checkpoint)
    else:
        while True:
            print("Enter prompt or leave it empty to exit")
            prompt = input()
            if not prompt:
                exit(0)
            else:
                prediction = get_prediction(model, prompt, dataset_generator)
                print(prediction)
