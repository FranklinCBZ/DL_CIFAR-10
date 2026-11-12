from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGE_DIR = Path("data/train_images")
LABELS_CSV = Path("train_labels.csv")

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class KaggleCIFAR10(Dataset):
    def __init__(self, dataframe, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = IMAGE_DIR / str(row["filename"])

        with Image.open(image_path) as img:
            image = img.convert("RGB")
            image = self.transform(image)

        label = int(row["label"])
        return image, label


def get_kaggle_loaders(batch_size=128, num_workers=4):
    df = pd.read_csv(LABELS_CSV)

    required = {"id", "filename", "label"}
    assert required.issubset(df.columns)
    assert df["label"].between(0, 9).all()
    assert df["filename"].is_unique

    missing = [
        name for name in df["filename"]
        if not (IMAGE_DIR / str(name)).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} training images missing. "
            f"Example: {missing[0]}"
        )

    train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        stratify=df["label"]
    )

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),

        transforms.RandomErasing(
            p=0.5,
            scale=(0.08, 0.15),
            ratio=(1.0, 1.0),
            value=0
        )
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    train_dataset = KaggleCIFAR10(
        train_df, train_transform
    )
    val_dataset = KaggleCIFAR10(
        val_df, val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader


def get_full_train_loader(batch_size=128, num_workers=4):
    df = pd.read_csv(LABELS_CSV)

    required = {"id", "filename", "label"}
    assert required.issubset(df.columns)
    assert df["label"].between(0, 9).all()
    assert df["filename"].is_unique

    missing = [
        name for name in df["filename"]
        if not (IMAGE_DIR / str(name)).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} training images missing. "
            f"Example: {missing[0]}"
        )

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),

        transforms.RandomErasing(
            p=0.5,
            scale=(0.08, 0.15),
            ratio=(1.0, 1.0),
            value=0
        )
    ])

    train_dataset = KaggleCIFAR10(
        df,
        train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0)
    )

    return train_loader



if __name__ == "__main__":
    train_loader, val_loader = get_kaggle_loaders()
    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))
    images, labels = next(iter(train_loader))
    print("Image batch:", images.shape)
    print("Label batch:", labels.shape)
