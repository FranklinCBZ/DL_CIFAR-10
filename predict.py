
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import CIFARResNet
from model_se import SEResNet
from dataset import MEAN, STD


TEST_CSV = Path("test.csv")
SAMPLE_CSV = Path("sample_submission.csv")
IMAGE_DIR = Path("data/test_images")
CHECKPOINT = Path("experiments/kaggle_v3_se/best_model.pth")
OUTPUT_CSV = Path("submission_kaggle_v3_tta.csv")

BATCH_SIZE = 128


class KaggleTestDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df.iloc[index]["filename"]
        image_path = IMAGE_DIR / filename

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    test_df = pd.read_csv(TEST_CSV)
    sample_df = pd.read_csv(SAMPLE_CSV)

    assert {"id", "filename"}.issubset(test_df.columns)
    assert list(sample_df.columns) == ["id", "label"]
    assert test_df["id"].is_unique
    assert sample_df["id"].is_unique
    assert set(test_df["id"]) == set(sample_df["id"])

    missing = [
        name for name in test_df["filename"]
        if not (IMAGE_DIR / name).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} test images missing. "
            f"First missing file: {missing[0]}"
        )

    #model = CIFARResNet(base_width=40).to(device)
    model = SEResNet(base_width=40).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Device:", device)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Test images:", len(test_df))

    dataset = KaggleTestDataset(test_df)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    predictions = []

    with torch.inference_mode():
        for images in loader:
            images = images.to(device)

            # Original image
            outputs_original = model(images)

            # Horizontally flipped image
            flipped_images = torch.flip(images, dims=[3])
            outputs_flipped = model(flipped_images)

            # Average probabilities
            probs_original = torch.softmax(
                outputs_original, dim=1
            )
            probs_flipped = torch.softmax(
                outputs_flipped, dim=1
            )

            avg_probs = (
                probs_original + probs_flipped
            ) / 2

            labels = avg_probs.argmax(dim=1)
            predictions.extend(labels.cpu().tolist())

    prediction_df = pd.DataFrame({
        "id": test_df["id"],
        "label": predictions
    })

    submission = sample_df[["id"]].merge(
        prediction_df,
        on="id",
        how="left",
        validate="one_to_one"
    )

    assert len(submission) == len(sample_df)
    assert submission["label"].notna().all()
    assert submission["label"].between(0, 9).all()

    submission["label"] = submission["label"].astype(int)
    submission.to_csv(OUTPUT_CSV, index=False)

    print("Saved:", OUTPUT_CSV)
    print(submission.head())


if __name__ == "__main__":
    main()
