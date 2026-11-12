
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score
)

from model_se import DeepSEResNetDropout
from model_v13_preact import PreActDeepSEResNetDropout
from kaggle_dataset import (
    KaggleCIFAR10,
    get_kaggle_loaders,
    MEAN,
    STD
)


# Use numerical IDs unless you have verified
# the dataset's exact class-name mapping.
CLASS_NAMES = [str(i) for i in range(10)]


@torch.inference_mode()
def evaluate(model, loader, device):
    y_true = []
    y_pred = []

    model.eval()

    for images, labels in loader:
        images = images.to(device)

        # Original orientation
        original = torch.softmax(
            model(images),
            dim=1
        )

        # Horizontal-flip TTA
        flipped = torch.softmax(
            model(torch.flip(images, dims=[3])),
            dim=1
        )

        probabilities = (original + flipped) / 2

        predictions = probabilities.argmax(dim=1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())

    return np.array(y_true), np.array(y_pred)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["v11", "v12", "v13"],
        default="v13"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # ====================================
    # Choose checkpoint and dataset
    # ====================================

    if args.model == "v13":
        checkpoint_path = Path(
            "experiments/kaggle_v13_preact_finetuned_280/final_model.pth"
        )

        # V11 trained on ALL 50,000 images.
        # This is a training-set diagnostic,
        # NOT an independent validation result.
        df = pd.read_csv("train_labels.csv")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

        dataset = KaggleCIFAR10(df, transform)

        loader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=0
        )

        dataset_name = "Training Set"

    else:
        checkpoint_path = Path(
            "experiments/kaggle_v12_finetuned/best_model.pth"
        )

        # V12 has a genuine held-out validation set.
        _, loader = get_kaggle_loaders(
            batch_size=128,
            num_workers=0
        )

        dataset_name = "Validation Set"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # ====================================
    # Load model
    # ====================================

    model = PreActDeepSEResNetDropout(
        num_classes=10,
        base_width=39,
        blocks=(3, 3, 3, 2),
        dropout=0.15
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    print("Device:", device)
    print("Checkpoint:", checkpoint_path)
    print("Epoch:", checkpoint.get("epoch"))
    print("Dataset:", dataset_name)
    print("Number of images:", len(loader.dataset))

    # ====================================
    # Evaluate
    # ====================================

    y_true, y_pred = evaluate(
        model,
        loader,
        device
    )

    acc = accuracy_score(y_true, y_pred) * 100

    print(f"\n{dataset_name} Accuracy: {acc:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(10)),
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(10))
    )

    cm_normalized = (
        cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    )

    # ====================================
    # Generate two matrices
    # ====================================

    for matrix, kind, fmt in [
        (cm, "counts", "d"),
        (cm_normalized, "normalized", ".2f")
    ]:
        fig, ax = plt.subplots(figsize=(10, 8))

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=CLASS_NAMES
        )

        display.plot(
            ax=ax,
            cmap="Blues",
            values_format=fmt,
            colorbar=True
        )

        ax.set_title(
            f"{args.model.upper()} - {dataset_name}\n"
            f"Confusion Matrix ({kind})"
        )

        plt.tight_layout()

        output_path = (
            output_dir /
            f"{args.model}_confusion_{kind}.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        print("Saved:", output_path)


if __name__ == "__main__":
    main()
