
import multiprocessing as mp

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

from kaggle_dataset import IMAGE_DIR, LABELS_CSV, MEAN, STD


class AlternatingFlipDataset(Dataset):
    def __init__(self, dataframe, epoch_value):
        self.df = dataframe.reset_index(drop=True)
        self.epoch_value = epoch_value

        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.RandomErasing(
                p=0.5,
                scale=(0.08, 0.15),
                ratio=(1.0, 1.0),
                value=0
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        with Image.open(
            IMAGE_DIR / str(row["filename"])
        ) as img:
            image = img.convert("RGB")

            # Alternate orientation every epoch
            if self.epoch_value.value % 2 == 1:
                image = TF.hflip(image)

            image = self.transform(image)

        return image, int(row["label"])


def get_alternating_loader(batch_size=128, num_workers=4):
    df = pd.read_csv(LABELS_CSV)

    epoch_value = mp.Value("i", 0)

    dataset = AlternatingFlipDataset(
        dataframe=df,
        epoch_value=epoch_value
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0)
    )

    return loader, epoch_value
