from . import data
import torch
import pytest

import re


class TestDataloaders:
    @pytest.fixture(scope='class')
    def dataset_generator(self):
        return data.DatasetGenerator(max_digits=4)

    @pytest.fixture(scope='class')
    def dataloaders(self, dataset_generator):
        train_ds = dataset_generator.generate_full_dataset(10000)
        val_ds = dataset_generator.generate_simple_dataset(128)
        test_ds = dataset_generator.generate_simple_dataset(1000)
        start_symbol = dataset_generator.get_start_symbol()
        return train_ds, val_ds, test_ds, start_symbol

    def test_full_dataset(self, dataloaders, dataset_generator):
        dataset, val_dataloader, test_dataloader, start_symbol = dataloaders

        for _ in range(10):
            problem_id = torch.randint(len(dataset), size=(1,)).item()
            left, shifted, right = dataset[problem_id]
            left = torch.argmax(left, dim=1)
            left = dataset_generator.idx2str(left.tolist())

            shifted = torch.argmax(shifted, dim=1)
            shifted = dataset_generator.idx2str(shifted.tolist())

            right = dataset_generator.idx2str(right.tolist())

            m = re.match(r'(\d+)\+(\d+)=', left)
            a, b = m.group(1), m.group(2)
            a = int(a)
            b = int(b)
            c = int(right)
            assert a + b == c

            assert c == int(shifted[1:])

    def test_simple_dataset(self, dataloaders, dataset_generator):
        _, dataset, test_dataloader, start_symbol = dataloaders
        for _ in range(10):
            problem_id = torch.randint(len(dataset), size=(1,)).item()
            left, right = dataset[problem_id]
            left = torch.argmax(left, dim=1)
            left = dataset_generator.idx2str(left.tolist())

            right = dataset_generator.idx2str(right.tolist())

            m = re.match(r'(\d+)\+(\d+)=', left)
            a, b = m.group(1), m.group(2)
            a = int(a)
            b = int(b)
            c = int(right)

            assert a + b == c
