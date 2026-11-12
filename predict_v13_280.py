
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from model_v13_preact import PreActDeepSEResNetDropout

# ==========================================
# Configuration
# ==========================================

CHECKPOINT_PATH = Path(
    "experiments/kaggle_v13_preact_finetuned_280/final_model.pth"
)

OUTPUT_CSV = Path(
    "submission_kaggle_v13_epoch280_tta.csv"
)

TEST_CSV = Path("test.csv")
SAMPLE_SUBMISSION = Path("sample_submission.csv")
TEST_IMAGE_DIR = Path("data/test_images")

BATCH_SIZE = 128

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


# ==========================================
# Dataset
# ==========================================

class KaggleTestDataset(Dataset):
    def __init__(self, dataframe, image_dir):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(
            self.image_dir / str(row["filename"])
        ).convert("RGB")

        return self.transform(image), row["id"]


# ==========================================
# Prediction with horizontal-flip TTA
# ==========================================

@torch.inference_mode()
def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    model = PreActDeepSEResNetDropout(
        num_classes=10,
        base_width=39,
        blocks=(3, 3, 3, 2),
        dropout=0.15
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Disables dropout during inference
    model.eval()

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Parameters:", checkpoint["num_parameters"])

    test_df = pd.read_csv(TEST_CSV)
    sample_df = pd.read_csv(SAMPLE_SUBMISSION)

    dataset = KaggleTestDataset(
        test_df,
        TEST_IMAGE_DIR
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    all_ids = []
    all_predictions = []

    for images, ids in loader:
        images = images.to(
            device, non_blocking=True
        )

        # Original prediction
        logits_original = model(images)
        probs_original = torch.softmax(
            logits_original, dim=1
        )

        # Horizontal-flip prediction
        flipped_images = torch.flip(
            images, dims=[3]
        )

        logits_flipped = model(flipped_images)
        probs_flipped = torch.softmax(
            logits_flipped, dim=1
        )

        # Equal-weight TTA
        avg_probs = (
            probs_original + probs_flipped
        ) / 2.0

        predictions = avg_probs.argmax(dim=1)

        if isinstance(ids, torch.Tensor):
            all_ids.extend(ids.cpu().tolist())
        else:
            all_ids.extend(list(ids))

        all_predictions.extend(
            predictions.cpu().tolist()
        )

    predictions_df = pd.DataFrame({
        "id": all_ids,
        "label": all_predictions
    })

    # Preserve Kaggle sample submission order
    submission = sample_df[["id"]].merge(
        predictions_df,
        on="id",
        how="left",
        validate="one_to_one"
    )

    assert submission["label"].notna().all(), (
        "Missing predictions for some test IDs"
    )

    submission["label"] = submission["label"].astype(int)

    assert len(submission) == len(sample_df)
    assert submission["label"].between(0, 9).all()

    submission.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\nSubmission generated successfully!")
    print("File:", OUTPUT_CSV)
    print("Rows:", len(submission))
    print(submission.head())


if __name__ == "__main__":
    main()
