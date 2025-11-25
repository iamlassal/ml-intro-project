import torch
import numpy as np

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_cifar100(batch_size=64, augment=False, val_ratio=0.1, seed=None):
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)

    transform_train_aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_plain = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_dataset = datasets.CIFAR100(
        root="data",
        train=True,
        download=True
    )

    total = len(full_dataset)
    val_size = int(total * val_ratio)
    train_size = total - val_size

    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )

    if augment:
        train_dataset.dataset.transform = transform_train_aug # pyright: ignore
    else:
        train_dataset.dataset.transform = transform_plain # pyright: ignore

    val_dataset.dataset.transform = transform_plain # pyright: ignore

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    if seed is not None:
        worker_init = seed_worker
    else:
        worker_init = None

    test_dataset = datasets.CIFAR100(
        root="data",
        train=False,
        transform=transform_plain,
        download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init,
    )

    val_loader   = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init,
    )

    test_loader  = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init,
    )

    return train_loader, val_loader, test_loader

def get_cifar100_transfer(batch_size=64, augment=True, val_ratio=0.1, seed=None):
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    if augment:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Load CIFAR-100 (no transform yet)
    full_dataset = datasets.CIFAR100(
        root="data",
        train=True,
        download=True
    )

    total = len(full_dataset)
    val_size = int(total * val_ratio)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(seed) if seed else None

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )

    train_dataset.dataset.transform = transform_train # pyright: ignore
    val_dataset.dataset.transform = transform_test # pyright: ignore

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    worker_init = seed_worker if seed else None

    test_dataset = datasets.CIFAR100(
        root="data",
        train=False,
        transform=transform_test,
        download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init,
    )

    return train_loader, val_loader, test_loader
