from data import data
from model import transformer as model
import torch
import pytorch_lightning as pl
from torchtyping import patch_typeguard


def get_dataloaders(dataset_generator,
                    train_size: int = 20000,
                    val_size: int = 512,
                    test_size: int = 1024,
                    max_digits=4):
    train_ds = dataset_generator.generate_full_dataset(train_size)
    val_ds = dataset_generator.generate_simple_dataset(val_size)
    test_ds = dataset_generator.generate_simple_dataset(test_size)
    start_symbol = dataset_generator.get_start_symbol()

    train_dataloader = torch.utils.data.DataLoader(train_ds,
                                                   batch_size=128,
                                                   shuffle=True,
                                                   num_workers=0,
                                                   pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_ds,
                                                 batch_size=128,
                                                 shuffle=True,
                                                 num_workers=0,
                                                 pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_ds,
                                                  batch_size=128,
                                                  shuffle=True,
                                                  num_workers=0,
                                                  pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader, start_symbol


def get_prediction(model, prompt, dataset_generator):
    model.eval()
    vectorized_prompt = dataset_generator.vectorize_dataset([prompt])
    result = model.predict(vectorized_prompt)
    result = result[0]  # first sample in batch
    result = torch.argmax(result, dim=0).tolist()
    return dataset_generator.idx2str(result)


def main():
    patch_typeguard()
    dataset_generator = data.DatasetGenerator(max_digits=4)
    train_dataloader, val_dataloader, test_dataloader, start_symbol = get_dataloaders(
        dataset_generator)

    transformer = model.Transformer(
        total_time=11,
        vocab_size=14,
        feature_size=512,
        n_encoders=4,
        n_decoders=4,
        learning_rate=0.005,
        start_symbol=start_symbol,
        dropout=0.0,
    )

    trainer = pl.Trainer(gpus=1,
                         max_epochs=1000,
                         logger=None,
                         overfit_batches=1,
                         deterministic=True)
    trainer.fit(transformer, train_dataloader)


if __name__ == '__main__':
    main()
