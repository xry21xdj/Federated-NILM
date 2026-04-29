import os
import pickle
import string

import torch
from torchvision.datasets import EMNIST, CIFAR10 ,CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_loader(type_, path, batch_size, train, inputs=None, targets=None,num_workers=1):
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
    if type_ == "cifar10":
        dataset = SubCIFAR10(path, cifar10_data=inputs, cifar10_targets=targets)
    elif type_ == "cifar100":
        dataset = SubCIFAR100(path, cifar100_data=inputs, cifar100_targets=targets)
    elif type_ == "emnist":
        dataset = SubEMNIST(path, emnist_data=inputs, emnist_targets=targets)
    else:
        raise NotImplementedError(f"{type_} not recognized type; possible are ")

    if len(dataset) == 0:
        return

    # drop last batch, because of BatchNorm layer used in mobilenet_v2
    drop_last = ((type_ == "cifar100") or (type_ == "cifar10")) and (len(dataset) > batch_size) and train

    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=drop_last)#,num_workers=num_workers)



def get_loaders(type_, root_path, batch_size, is_validation,num_workers):
    """
    constructs lists of `torch.utils.DataLoader` object from the given files in `root_path`;
     corresponding to `train_iterator`, `val_iterator` and `test_iterator`;
     `val_iterator` iterates on the same dataset as `train_iterator`, the difference is only in drop_last

    :param type_: type of the dataset;
    :param root_path: path to the data folder
    :param batch_size:
    :param is_validation: (bool) if `True` validation part is used as test
    :return:
        train_iterator, val_iterator, test_iterator
        (List[torch.utils.DataLoader], List[torch.utils.DataLoader], List[torch.utils.DataLoader])

    """
    if type_ == "cifar10":
        inputs, targets = get_cifar10()
    elif type_ == "cifar100":
        inputs, targets = get_cifar100()
    elif type_ == "emnist":
        inputs, targets = get_emnist()
    else:
        inputs, targets = None, None

    train_iterators, val_iterators, test_iterators = [], [], []

    for task_id, task_dir in enumerate(tqdm(os.listdir(root_path))):
        task_data_path = os.path.join(root_path, task_dir)

        train_iterator = \
            get_loader(
                type_=type_,
                path=os.path.join(task_data_path, "train.pkl"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=True,
                num_workers=num_workers
            )

        val_iterator = \
            get_loader(
                type_=type_,
                path=os.path.join(task_data_path, "train.pkl"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=False,
                num_workers=num_workers
            )

        if is_validation:
            test_set = "val"
        else:
            test_set = "test"

        test_iterator = \
            get_loader(
                type_=type_,
                path=os.path.join(task_data_path, f"{test_set}.pkl"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=False,
                num_workers=num_workers
            )

        train_iterators.append(train_iterator)
        val_iterators.append(val_iterator)
        test_iterators.append(test_iterator)

    return train_iterators, val_iterators, test_iterators





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


class SubCIFAR10(Dataset):
    """
    Constructs a subset of CIFAR10 dataset from a pickle file;
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
    def __init__(self, path, cifar10_data=None, cifar10_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param cifar10_data: Cifar-10 dataset inputs stored as torch.tensor
        :param cifar10_targets: Cifar-10 dataset labels stored as torch.tensor
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)
                    )
                ])

        if cifar10_data is None or cifar10_targets is None:
            self.data, self.targets = get_cifar10()
        else:
            self.data, self.targets = cifar10_data, cifar10_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        target = target

        return img, target, index


class SubCIFAR100(Dataset):
    """
    Constructs a subset of CIFAR100 dataset from a pickle file;
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
    def __init__(self, path, cifar100_data=None, cifar100_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices:
        :param cifar100_data: CIFAR-100 dataset inputs
        :param cifar100_targets: CIFAR-100 dataset labels
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)
                    )
                ])

        if cifar100_data is None or cifar100_targets is None:
            self.data, self.targets = get_cifar100()

        else:
            self.data, self.targets = cifar100_data, cifar100_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        target = target

        return img, target, index

def get_emnist():
    """
    gets full (both train and test) EMNIST dataset inputs and labels;
    the dataset should be first downloaded (see data/emnist/README.md)
    :return:
        emnist_data, emnist_targets
    """
    emnist_path = os.path.join("data", "emnist", "raw_data")
    assert os.path.isdir(emnist_path), "Download EMNIST dataset!!"

    emnist_train =\
        EMNIST(
            root=emnist_path,
            split="byclass",
            download=True,
            train=True
        )

    emnist_test =\
        EMNIST(
            root=emnist_path,
            split="byclass",
            download=True,
            train=True
        )

    emnist_data =\
        torch.cat([
            emnist_train.data,
            emnist_test.data
        ])

    emnist_targets =\
        torch.cat([
            emnist_train.targets,
            emnist_test.targets
        ])

    return emnist_data, emnist_targets


def get_cifar10():
    """
    gets full (both train and test) CIFAR10 dataset inputs and labels;
    the dataset should be first downloaded (see data/emnist/README.md)
    :return:
        cifar10_data, cifar10_targets
    """
    cifar10_path = os.path.join("data", "cifar10", "raw_data")
    assert os.path.isdir(cifar10_path), "Download cifar10 dataset!!"

    cifar10_train =\
        CIFAR10(
            root=cifar10_path,
            train=True, download=False
        )

    cifar10_test =\
        CIFAR10(
            root=cifar10_path,
            train=False,
            download=False)

    cifar10_data = \
        torch.cat([
            torch.tensor(cifar10_train.data),
            torch.tensor(cifar10_test.data)
        ])

    cifar10_targets = \
        torch.cat([
            torch.tensor(cifar10_train.targets),
            torch.tensor(cifar10_test.targets)
        ])

    return cifar10_data, cifar10_targets


def get_cifar100():
    """
    gets full (both train and test) CIFAR100 dataset inputs and labels;
    the dataset should be first downloaded (see data/cifar100/README.md)
    :return:
        cifar100_data, cifar100_targets
    """
    cifar100_path = os.path.join("data", "cifar100", "raw_data")
    assert os.path.isdir(cifar100_path), "Download cifar10 dataset!!"

    cifar100_train =\
        CIFAR100(
            root=cifar100_path,
            train=True, download=False
        )

    cifar100_test =\
        CIFAR100(
            root=cifar100_path,
            train=False,
            download=False)

    cifar100_data = \
        torch.cat([
            torch.tensor(cifar100_train.data),
            torch.tensor(cifar100_test.data)
        ])

    cifar100_targets = \
        torch.cat([
            torch.tensor(cifar100_train.targets),
            torch.tensor(cifar100_test.targets)
        ])

    return cifar100_data, cifar100_targets