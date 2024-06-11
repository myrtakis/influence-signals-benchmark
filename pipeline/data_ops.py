import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, TensorDataset
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_pil_image



def subset_selection(dataset, labels, ratio, random_seed):
    ratio = int(len(labels) * ratio)
    (
        sel_ids,
        _,
    ) = train_test_split(
        np.arange(len(labels)),
        train_size=ratio,
        random_state=random_seed,
        stratify=labels,
    )
    return Subset(dataset, sel_ids), labels[sel_ids], sel_ids


def convert_labels(labels):
    if type(labels) == torch.Tensor:
        all_train_labels = labels.numpy()
    else:
        all_train_labels = np.array(labels)
    return all_train_labels


def gen_save_subset(dataset, labels, ratio, savedir, fname, seed=42):
    labels = convert_labels(labels)
    subset, labels, _ = subset_selection(dataset, labels, ratio, seed)
    Path(savedir).mkdir(parents=True, exist_ok=True)
    x_tensor = torch.stack([x for x, _ in subset])
    y_tensor = torch.tensor(labels)
    fpath = os.path.join(savedir, fname)
    subset_dataset = TensorDataset(x_tensor, y_tensor)
    torch.save(subset_dataset, fpath)
    with open(os.path.join(savedir, "subset_info.json"), "w") as f:
        json.dump({"subset_ratio": ratio, "seed": seed}, f)
    return subset_dataset

class MnistLoader:
    def load_data(self, data_folder_path):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        trainset = MNIST(
            root=data_folder_path, download=True, train=True, transform=transform
        )
        testset = MNIST(
            root=data_folder_path, download=True, train=False, transform=transform
        )
        return trainset, testset, trainset.targets, testset.targets


class FmnistLoader:
    def load_data(self, data_folder_path):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        trainset = FashionMNIST(
            root=data_folder_path, download=True, train=True, transform=transform
        )
        testset = FashionMNIST(
            root=data_folder_path, download=True, train=False, transform=transform
        )
        return trainset, testset, trainset.targets, testset.targets


class Cifar10Loader:
    def load_data(self, data_folder_path):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        trainset = CIFAR10(
            root=data_folder_path, download=True, train=True, transform=transform
        )
        trainset.targets = torch.tensor(trainset.targets)
        testset = CIFAR10(
            root=data_folder_path, download=True, train=False, transform=transform
        )
        testset.targets = torch.tensor(testset.targets)
        return trainset, testset, trainset.targets, testset.targets


class CustomLoader:
    def load_data(self, data_folder_path):
        train_data_fp = os.path.join(data_folder_path, "train.pt")
        test_data_fp = os.path.join(data_folder_path, "test.pt")
        train_data = torch.load(train_data_fp)
        test_data = torch.load(test_data_fp)
        train_labels = train_data.tensors[1]
        test_labels = test_data.tensors[1]
        return train_data, test_data, train_labels, test_labels


class MnistCorruptedLoader:

   def load_data(self, data_folder_path, corruption=None):
       assert corruption is not None
       transform = transforms.Compose(
           [
               transforms.ToTensor(),
               transforms.Normalize((0.5,), (0.5,)),
           ]
       )
       train_data_fp = Path(data_folder_path, 'mnist_corrupted', corruption, 'train.pt')
       train_data = torch.load(train_data_fp)
       train_labels = train_data.tensors[1]
       normalised_train_data = [transform(to_pil_image(train_data.tensors[0][i])) for i in range(len(train_labels))]
       train_data = TensorDataset(torch.stack(normalised_train_data, dim=0), train_labels)
       return train_data, None, train_labels, None


class Cifar10CorruptedLoader:

   def load_data(self, data_folder_path, corruption=None):
       assert corruption is not None
       transform = transforms.Compose(
           [
               transforms.ToTensor(),
               transforms.Normalize((0.5,), (0.5,)),
           ]
       )
       train_data_fp = Path(data_folder_path, 'cifar10_corrupted', corruption, 'train.pt')
       train_data = torch.load(train_data_fp)
       train_labels = train_data.tensors[1]
       normalised_train_data = [transform(to_pil_image(train_data.tensors[0][i])) for i in range(len(train_labels))]
       train_data = TensorDataset(torch.stack(normalised_train_data, dim=0), train_labels)
       return train_data, None, train_labels, None

class TabularLoader:
    def load_tabular_data(self, data_folder_path):
        pass


dispatcher = {
    "mnist": MnistLoader,
    "fmnist": FmnistLoader,
    "cifar10": Cifar10Loader,
    "mnist_corrupted": MnistCorruptedLoader,
    "cifar10_corrupted": Cifar10CorruptedLoader,
    "custom_loader": CustomLoader,
    "tabular_loader": TabularLoader
}
