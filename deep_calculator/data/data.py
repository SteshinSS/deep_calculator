import numpy as np
import torch
import torchtext


class FullDataset(torch.utils.data.Dataset):
    """Dataloader for training.

    Returns problems in vectorized batched left-padded form:
        (left, shifted), right
    Where
        left.shape == (batch, enc_time, vocab) -- input
        right.shape == (batch, dec_time, vocab) -- output
        shifted.shape == (batch, dec_time, vocab) -- output shifted by sos token
    """

    def __init__(self, left, right, shifted):
        self.left = left
        self.right = right
        self.shifted = shifted

    def __len__(self):
        return self.left.shape[0]

    def __getitem__(self, idx):
        return self.left[idx], self.shifted[idx], self.right[idx]


class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __len__(self):
        return self.left.shape[0]

    def __getitem__(self, idx):
        return self.left[idx], self.right[idx]


class DatasetGenerator():

    def __init__(self, max_digits=4, max_padded_length=None):
        if max_padded_length is None:
            max_padded_length = max_digits + 1 + max_digits + 2  # 'nnn+nnn= '

        self.max_padded_length = max_padded_length
        self.alphabet = '1234567890+=! '
        self.vocab = torchtext.vocab.build_vocab_from_iterator(self.alphabet)
        self.max_digits = max_digits

    def generate_full_dataset(self, total_problems):
        left, right, shifted = self._generate_dataset(total_problems)
        left = self.vectorize_dataset(left)
        right = self.vectorize_dataset(right, onehot=False)
        shifted = self.vectorize_dataset(shifted)
        return FullDataset(left, right, shifted)

    def generate_simple_dataset(self, total_problems):
        left, right, _ = self._generate_dataset(total_problems)
        left = self.vectorize_dataset(left)
        right = self.vectorize_dataset(right, onehot=False)
        return SimpleDataset(left, right)

    def idx2str(self, indices):
        return ''.join(self.vocab.lookup_tokens(indices))

    def get_start_symbol(self):
        start_symbol = self.vocab.lookup_indices(['!'])
        start_symbol = torch.Tensor(start_symbol).to(torch.int64)
        start_symbol = torch.nn.functional.one_hot(start_symbol,
                                                   len(self.alphabet)).to(
                                                       torch.float32)
        return start_symbol

    def vectorize_dataset(self, dataset, onehot=True):
        result = torch.zeros((len(dataset), self.max_padded_length),
                             dtype=torch.int64)
        for i, string in enumerate(dataset):
            indices = self.vocab.lookup_indices(list(string))
            for j, idx in enumerate(indices):
                result[i, j] = idx
        if onehot:
            result = torch.nn.functional.one_hot(result, len(self.alphabet)).to(
                torch.float32)
        return result

    def _generate_dataset(self, total_problems):
        left_parts = []
        right_parts = []
        right_parts_shifted = []
        for _ in range(total_problems):
            left_part, right_part, right_part_shifted = self._generate_problem()
            left_parts.append(left_part)
            right_parts.append(right_part)
            right_parts_shifted.append(right_part_shifted)
        return left_parts, right_parts, right_parts_shifted

    def _generate_problem(self):
        a = np.random.randint(10**(self.max_digits - 1))
        b = np.random.randint(10**(self.max_digits - 1))
        c = a + b
        left_part = str(a) + '+' + str(b) + '='
        right_part = str(c)
        right_part_shifted = '!' + right_part
        return left_part, right_part, right_part_shifted


def get_dataloaders(config, dataset_generator):
    
    train_ds = dataset_generator.generate_full_dataset(config['train_size'])
    val_ds = dataset_generator.generate_simple_dataset(config['val_size'])
    test_ds = dataset_generator.generate_simple_dataset(config['test_size'])

    batch_size = config['batch_size']

    train_dataloader = torch.utils.data.DataLoader(train_ds,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=1)
    val_dataloader = torch.utils.data.DataLoader(val_ds,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=1)
    test_dataloader = torch.utils.data.DataLoader(test_ds,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=1)

    dataloaders = {
        'train': train_dataloader,
        'val': val_dataloader,
        'test': test_dataloader,
    }
    return dataloaders