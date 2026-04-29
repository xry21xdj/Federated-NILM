import os
import pickle
import string

import torch
from torchvision.datasets import EMNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
import numpy as np
from PIL import Image











class TabularDataset(Dataset):
    """
    Constructs a torch.utils.Dataset object from a pickle file;
    expects pickle file stores tuples of the form (x, y) where x is vector and y is a scalar

    Attributes
    ----------
    data: iterable of tuples (x, y)

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, path):
        """
        :param path: path to .pkl file
        """
        with open(path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.int64), idx


class SubFEMNIST(Dataset):
    """
    Constructs a subset of FEMNIST dataset corresponding to one client;
    Initialized with the path to a `.pt` file;
    `.pt` file is expected to hold a tuple of tensors (data, targets) storing the images and there corresponding labels.

    Attributes
    ----------
    transform
    data: iterable of integers
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, path):
        self.transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])

        self.data, self.targets = torch.load(path)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = np.uint8(img.numpy() * 255)
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index


class SubEMNIST(Dataset):
    """
    Constructs a subset of EMNIST dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, path, emnist_data=None, emnist_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param emnist_data: EMNIST dataset inputs
        :param emnist_targets: EMNIST dataset labels
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if transform is None:
            self.transform =\
                Compose([
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,))
                ])

        if emnist_data is None or emnist_targets is None:
            self.data, self.targets = get_emnist()
        else:
            self.data, self.targets = emnist_data, emnist_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index







def get_loader(type_, path, batch_size, train, inputs=None, targets=None,num_workers=2):
    """
    constructs a torch.utils.DataLoader object from the given path

    :param type_: type of the dataset; possible are `tabular`, `images` and `text`
    :param path: path to the data file
    :param batch_size:
    :param train: flag indicating if train loader or test loader
    :param inputs: tensor storing the input data; only used with `cifar10`, `cifar100` and `emnist`; default is None
    :param targets: tensor storing the labels; only used with `cifar10`, `cifar100` and `emnist`; default is None
    :return: torch.utils.DataLoader

    """
    if type_ == "tabular":
        dataset = TabularDataset(path)
    elif type_ == "cifar10":
        dataset = SubCIFAR10(path, cifar10_data=inputs, cifar10_targets=targets)
    elif type_ == "cifar100":
        dataset = SubCIFAR100(path, cifar100_data=inputs, cifar100_targets=targets)
    elif type_ == "emnist":
        dataset = SubEMNIST(path, emnist_data=inputs, emnist_targets=targets)
    elif type_ == "imagenet":
        dataset = SubImageNet(path, imagenet_data=inputs, imagenet_targets=targets)
    elif type_ == "femnist":
        dataset = SubFEMNIST(path)
    elif type_ == "shakespeare":
        dataset = CharacterDataset(path, chunk_len=SHAKESPEARE_CONFIG["chunk_len"])
    else:
        raise NotImplementedError(f"{type_} not recognized type; possible are {list(LOADER_TYPE.keys())}")

    if len(dataset) == 0:
        return

    # drop last batch, because of BatchNorm layer used in mobilenet_v2
    drop_last = ((type_ == "cifar100") or (type_ == "cifar10")) and (len(dataset) > batch_size) and train

    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=drop_last,num_workers=num_workers)