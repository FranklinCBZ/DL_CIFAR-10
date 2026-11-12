
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def get_dataloaders(batch_size=128, num_workers=4):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    val_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=val_transform
    )

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(
        len(train_dataset), generator=generator
    ).tolist()

    train_indices = indices[:45000]
    val_indices = indices[45000:]

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders()

    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))

    images, labels = next(iter(train_loader))

    print("Image batch:", images.shape)
    print("Label batch:", labels.shape)
