import os
from typing import Dict, Any, Tuple, Optional
import torchvision
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10


def make_loaders(cfg):
    """  
    Return train, val and test loaders for full CIFAR-10 dataset.
    Split the test set into val/test with equal samples per class.
    Split is controlled by cfg.seed.
    
    Note: Full CIFAR-10 has 50k training samples, so batch_size cannot be "full".
    Uses sensible default of 256 if "full" or None is specified.
    """
    data_path = "/fast/slaing/data/vision/cifar10/"
    
    # Correct CIFAR-10 normalization stats (normalize BEFORE flattening)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  # Shape: (3, 32, 32)
        torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010]),  # Still (3, 32, 32)
        torchvision.transforms.Lambda(lambda x: x.reshape(-1))  # Flatten to 3072-dim vector
    ])
    
    train_dataset = CIFAR10(root=data_path, train=True, download=False, transform=transform)
    test_dataset = CIFAR10(root=data_path, train=False, download=False, transform=transform)

    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    # Split test set into val/test with stratification by class
    class_indices = {i: [] for i in range(10)}  # CIFAR-10 has 10 classes
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

    # Cannot do full batch for full CIFAR-10 (50k samples)
    # Use reasonable batch size if cfg.batch_size is 'full' or None
    if cfg.batch_size == "full" or cfg.batch_size is None:
        batch_size_train = 256  # sensible default for full CIFAR-10
    else:
        batch_size_train = cfg.batch_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
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
    from dataclasses import dataclass
    
    @dataclass
    class Config:
        batch_size: int = 256
        num_workers: int = 4
        seed: Optional[int] = 42

    cfg = Config()
    train_loader, val_loader, test_loader = make_loaders(cfg)
    
    # Count each class in train loader
    class_counts = [0] * 10
    for x, labels in train_loader:
        for label in labels:
            class_counts[label] += 1
        break
    print("Class distribution in train loader (first batch):", class_counts)

    # Count each class in val loader
    class_counts = [0] * 10
    for _, labels in val_loader:
        for label in labels:
            class_counts[label] += 1
    print("Class distribution in val loader:", class_counts)
    
    # Count each class in test loader
    class_counts = [0] * 10
    for _, labels in test_loader:
        for label in labels:
            class_counts[label] += 1
    print("Class distribution in test loader:", class_counts)

    print(f"\nTrain loader size: {len(train_loader.dataset)}")
    print(f"Val loader size: {len(val_loader.dataset)}")
    print(f"Test loader size: {len(test_loader.dataset)}")
