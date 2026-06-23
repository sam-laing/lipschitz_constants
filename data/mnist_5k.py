import os
from typing import Optional, Tuple
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader


class MNIST_5k(Dataset):
    """5-class (0-4), 1k-per-class subset of MNIST. Images: (28,28) uint8 -> flattened to 784."""

    def __init__(self, train: bool = True, transform=None, data_root: Optional[str] = None):
        split = "train" if train else "test"
        root = data_root if data_root else os.path.join(os.path.dirname(__file__), "mnist_5k")
        data = np.load(os.path.join(root, split, "data.npz"))
        self.images = data["images"]   # (N, 28, 28) uint8
        self.labels = data["labels"]   # (N,) int64
        assert self.images.shape[0] == self.labels.shape[0]
        self.num_samples = self.images.shape[0]
        self.transform = transform

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image = np.expand_dims(self.images[index], axis=-1)  # (28, 28, 1) for ToTensor
        label = int(self.labels[index])
        if self.transform:
            image = self.transform(image)
        return image.reshape(-1), label


class FashionMNIST_5k(Dataset):
    """5-class (0-4), 1k-per-class subset of FashionMNIST. Images: (28,28) uint8 -> flattened to 784."""

    def __init__(self, train: bool = True, transform=None, data_root: Optional[str] = None):
        split = "train" if train else "test"
        root = data_root if data_root else os.path.join(os.path.dirname(__file__), "fmnist_5k")
        data = np.load(os.path.join(root, split, "data.npz"))
        self.images = data["images"]
        self.labels = data["labels"]
        assert self.images.shape[0] == self.labels.shape[0]
        self.num_samples = self.images.shape[0]
        self.transform = transform

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image = np.expand_dims(self.images[index], axis=-1)
        label = int(self.labels[index])
        if self.transform:
            image = self.transform(image)
        return image.reshape(-1), label


def make_loaders(cfg):
    """Train/val/test loaders. Test split is halved per-class into val and test (controlled by cfg.seed)."""
    if cfg.dataset == 'mnist_5k':
        # mean/std over full MNIST; close enough for the 5-class subset
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081]),
        ])
        data_root = getattr(cfg, 'data_root', None)
        train_dataset = MNIST_5k(train=True, transform=transform, data_root=data_root)
        test_dataset = MNIST_5k(train=False, transform=transform, data_root=data_root)

    elif cfg.dataset == 'fmnist_5k':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.2860], std=[0.3530]),
        ])
        data_root = getattr(cfg, 'data_root', None)
        train_dataset = FashionMNIST_5k(train=True, transform=transform, data_root=data_root)
        test_dataset = FashionMNIST_5k(train=False, transform=transform, data_root=data_root)

    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    class_indices = {i: [] for i in range(5)}
    for idx, (_, label) in enumerate(test_dataset):
        class_indices[label].append(idx)

    val_indices, test_indices = [], []
    for c in range(5):
        indices = class_indices[c]
        np.random.shuffle(indices)
        split = len(indices) // 2
        val_indices.extend(indices[:split])
        test_indices.extend(indices[split:])

    val_dataset = torch.utils.data.Subset(test_dataset, val_indices)
    test_dataset_final = torch.utils.data.Subset(test_dataset, test_indices)

    pin = torch.cuda.is_available()
    batch_size = cfg.batch_size
    if batch_size == "full" or batch_size is None:
        batch_size = len(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_dataset_final, batch_size=len(test_dataset_final), shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader
