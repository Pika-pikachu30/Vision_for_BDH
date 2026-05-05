"""
STL-10 dataset loading utilities for Vision-BDH experiments.

STL-10 facts:
- 96×96 RGB images (3× higher resolution than CIFAR)
- 10 classes (airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck)
- Train: 5,000 samples (500/class) — 10× less than CIFAR-10
- Test: 8,000 samples
- Specifically designed to test data-scarce regime learning

"""

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import STL10
from torchvision import transforms
import numpy as np


# ─────────────────────────────────────────────
# Normalization stats for STL-10
# (computed from training set)
# ─────────────────────────────────────────────
STL10_MEAN = (0.4409, 0.4279, 0.3868)
STL10_STD  = (0.2683, 0.2610, 0.2687)

STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck"
]


# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────

def get_train_transform(img_size: int = 96) -> transforms.Compose:
    """
    Augmentation matching -- 2025's CIFAR philosophy, scaled to 96×96.
    Deliberately conservative to not over-regularize in data-scarce regime.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(STL10_MEAN, STL10_STD),
    ])


def get_val_transform(img_size: int = 96) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(STL10_MEAN, STL10_STD),
    ])


def get_strong_train_transform(img_size: int = 96) -> transforms.Compose:
    """
    Stronger augmentation for data-efficiency experiments.
    Used as a sensitivity check.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(STL10_MEAN, STL10_STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


# ─────────────────────────────────────────────
# Dataset loaders
# ─────────────────────────────────────────────

def get_stl10_loaders(
    data_root: str = "./data_stl10",
    batch_size: int = 32,
    val_split: float = 0.1,
    img_size: int = 96,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple:
    """
    Returns (train_loader, val_loader, test_loader).

    Train: 4500 samples (90% of 5000), Val: 500 (10%), Test: 8000.
    This matches standard practice for STL-10 supervised experiments.
    """
    train_transform = get_train_transform(img_size)
    val_transform = get_val_transform(img_size)

    full_train = STL10(root=data_root, split="train", download=True, transform=train_transform)
    full_train_for_val = STL10(root=data_root, split="train", download=False, transform=val_transform)
    test_dataset = STL10(root=data_root, split="test", download=True, transform=val_transform)

    n = len(full_train)  # 5000
    val_size = int(val_split * n)
    train_size = n - val_size

    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(range(n), [train_size, val_size], generator=generator)

    train_dataset = Subset(full_train, list(train_idx))
    val_dataset = Subset(full_train_for_val, list(val_idx))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"STL-10 loaded | Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader


def get_stl10_fraction_loader(
    fraction: float,
    data_root: str = "./data_stl10",
    batch_size: int = 32,
    img_size: int = 96,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple:
    """
    Returns (train_loader, val_loader, test_loader) with only `fraction` of train data.
    Used for the label efficiency experiment (10%, 25%, 50%, 100%).

    Stratified sampling: preserves class balance.
    """
    assert 0.0 < fraction <= 1.0
    val_transform = get_val_transform(img_size)
    train_transform = get_train_transform(img_size)

    full_train = STL10(root=data_root, split="train", download=True, transform=train_transform)
    full_train_val = STL10(root=data_root, split="train", download=False, transform=val_transform)
    test_dataset = STL10(root=data_root, split="test", download=True, transform=val_transform)

    labels = np.array(full_train.labels)
    rng = np.random.default_rng(seed)

    train_idx = []
    val_idx = []

    for cls in range(10):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(5, int(0.1 * len(cls_idx)))  # 10% val
        n_train = int(fraction * (len(cls_idx) - n_val))
        n_train = max(1, n_train)  # at least 1 sample per class

        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:n_val + n_train].tolist())

    train_dataset = Subset(full_train, train_idx)
    val_dataset = Subset(full_train_val, val_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=min(batch_size, len(train_dataset)),
        shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(
        f"STL-10 [{fraction*100:.0f}%] | "
        f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}"
    )
    return train_loader, val_loader, test_loader