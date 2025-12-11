import os
from typing import Dict, Any, Tuple, Optional
import torchvision
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10


class CIFAR10_5k(Dataset):
    """   
    just a very simple dataset loader for the cifar10 5k dataset from HF
    Expects data in npz format with 'images' and 'labels' arrays.
    """

    def __init__(self, train: bool = True, transform: Optional[torchvision.transforms.Compose] = None) -> None:
        if train:
            data_dir = "/fast/slaing/data/vision/cifar10_5class_5k/train"
        else:
            data_dir = "/fast/slaing/data/vision/cifar10_5class_5k/test"
        npz_path = os.path.join(data_dir, "data.npz")
        data = np.load(npz_path)
        self.images = data["images"]  # shape (5000, 32, 32, 3), uint8
        self.labels = data["labels"]  # shape (5000,), int64
        assert self.images.shape[0] == self.labels.shape[0], "Mismatched images and labels"
        self.num_samples = self.images.shape[0]
        self.transform = transform

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        image = self.images[index]
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        image = image.reshape(-1)
        return image, label
 
    
def make_loaders(cfg):
    """  
    Return train, val and test loaders. Split the test set into val/test with equal samples per class.
    Split is controlled by cfg.seed.
    """
    if cfg.dataset == 'cifar10_5k':
        """  
        Mean: [0.4914423  0.48771504 0.45364332]
        Std: [0.24486475 0.2414065  0.26222563]
        """
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.4914423, 0.48771504, 0.45364332],
                                             std=[0.24486475, 0.2414065, 0.26222563]),
            torchvision.transforms.Lambda(lambda x: x.view(x.size(0), -1))  
            
            ])



        train_dataset = CIFAR10_5k(train=True, transform=transform)
        test_dataset = CIFAR10_5k(train=False, transform=transform)

        if cfg.seed is not None:
            np.random.seed(cfg.seed)
        
        class_indices = {i: [] for i in range(5)}  # CIFAR10_5k has 5 classes
        for idx, (_, label) in enumerate(test_dataset):
            class_indices[label].append(idx)

        val_indices, test_indices = [], []
        for class_idx in range(5):
            indices = class_indices[class_idx]
            np.random.shuffle(indices)
            split = len(indices) // 2
            val_indices.extend(indices[:split])
            test_indices.extend(indices[split:])

        val_dataset = torch.utils.data.Subset(test_dataset, val_indices)
        test_dataset_final = torch.utils.data.Subset(test_dataset, test_indices)


        if cfg.batch_size == "full" or cfg.batch_size is None:
            cfg.batch_size = len(train_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=len(val_dataset),  # full batch for val
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset_final,
            batch_size=len(test_dataset_final),  # full batch for test
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader

    elif cfg.dataset == 'cifar10':
        data_path = "/fast/slaing/data/vision/cifar10/"
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                             std=[0.2023, 0.1994, 0.2010]), 
            
            ])  
        train_dataset = CIFAR10(root=data_path, train=True, download=False, transform=transform)
        test_dataset = CIFAR10(root=data_path, train=False, download=False, transform=transform)

        if cfg.seed is not None:
            np.random.seed(cfg.seed)

        class_indices = {i: [] for i in range(10)}  # CIFAR10 has 10 classes
        for idx, (_, label) in enumerate(test_dataset):
            class_indices[label].append(idx)
        val_indices, test_indices = [], []
        for class_idx in range(10): 
            indices = class_indices[class_idx]
            np.random.shuffle(indices)
            split = len(indices) // 2
            val_indices.extend(indices[:split])
            test_indices.extend(indices[split:])
        
        val_dataset = torch.utils.data.Subset(test_dataset, val_indices)
        test_dataset_final = torch.utils.data.Subset(test_dataset, test_indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=len(val_dataset),  # full batch for val
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset_final,
            batch_size=len(test_dataset_final),  # full batch for test
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader


if __name__ == "__main__":

    cfg = {
        "dataset": "cifar10",
        "batch_size": 64,
        "num_workers": 4,
        "seed": 42
    }
    from dataclasses import dataclass
    @dataclass
    class Config:
        dataset: str
        batch_size: int
        num_workers: int
        seed: Optional[int] = None

    cfg = Config(**cfg)


    train_loader, val_loader, test_loader = make_loaders(cfg)
    #count each class in train loader
    class_counts = [0] * 10
    for x, labels in train_loader:
        for label in labels:
            print(x.shape)

            class_counts[label] += 1
        break
    print("Class distribution in train loader:", class_counts)

    #count each class in val loader
    class_counts = [0] * 10
    for _, labels in val_loader:
        for label in labels:
            class_counts[label] += 1
    print("Class distribution in val loader:", class_counts)
    #count each class in test loader
    class_counts = [0] * 10
    for _, labels in test_loader:
        for label in labels:
            class_counts[label] += 1
    print("Class distribution in test loader:", class_counts)


    print(f"Train loader size: {len(train_loader.dataset)}")
    print(f"Val loader size: {len(val_loader.dataset)}")
    print(f"Test loader size: {len(test_loader.dataset)}")