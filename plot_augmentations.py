
from pathlib import Path
import random

import pandas as pd
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from kaggle_dataset import MEAN, STD


# ==========================================
# Configuration
# ==========================================

SEED = 42
IMAGE_DIR = Path("data/train_images")
LABELS_CSV = Path("train_labels.csv")

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)

# Select 3 example images from different classes
NUM_EXAMPLES = 3

# ==========================================
# Same augmentations used during training
# ==========================================

random_crop = transforms.RandomCrop(
    32,
    padding=4
)

to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(MEAN, STD)

# Use p=1.0 for illustration so the erased
# region is guaranteed to appear.
# Actual training uses p=0.5.
random_erasing = transforms.RandomErasing(
    p=1.0,
    scale=(0.08, 0.15),
    ratio=(1.0, 1.0),
    value=0
)


def denormalize(tensor):
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)

    return (tensor * std + mean).clamp(0, 1)


# ==========================================
# Select real training images
# ==========================================

df = pd.read_csv(LABELS_CSV)

# One randomly selected image per class
examples = (
    df.groupby("label")
    .sample(n=1, random_state=SEED)
    .sort_values("label")
    .head(NUM_EXAMPLES)
)

fig, axes = plt.subplots(
    NUM_EXAMPLES,
    4,
    figsize=(12, 3 * NUM_EXAMPLES),
    squeeze=False
)

titles = [
    "Original",
    "Random Crop",
    "Horizontal Flip",
    "Random Erasing"
]

for col, title in enumerate(titles):
    axes[0, col].set_title(
        title,
        fontsize=13,
        fontweight="bold"
    )


# ==========================================
# Generate example augmentations
# ==========================================

for row_idx, (_, row) in enumerate(
    examples.iterrows()
):
    image_path = IMAGE_DIR / str(row["filename"])

    with Image.open(image_path) as img:
        original = img.convert("RGB")

    # 1. Original
    axes[row_idx, 0].imshow(original)

    # 2. Random crop with padding=4
    cropped = random_crop(original)
    axes[row_idx, 1].imshow(cropped)

    # 3. Horizontal flip
    # Represents the flipped orientation in
    # an alternating-flip training epoch.
    flipped = TF.hflip(original)
    axes[row_idx, 2].imshow(flipped)

    # 4. Random Erasing
    # Applied after tensor conversion and
    # normalization, as in your training code.
    tensor = normalize(to_tensor(original))
    erased = random_erasing(tensor)

    erased_display = denormalize(erased)
    erased_display = erased_display.permute(
        1, 2, 0
    ).numpy()

    axes[row_idx, 3].imshow(erased_display)

    # Class label shown on the left
    axes[row_idx, 0].set_ylabel(
        f"Class {row['label']}",
        fontsize=11,
        fontweight="bold"
    )

    for ax in axes[row_idx]:
        ax.set_xticks([])
        ax.set_yticks([])

fig.suptitle(
    "CIFAR-10 Training Data Augmentation Examples",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

output_path = (
    OUTPUT_DIR / "cifar10_augmentation_examples.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("Figure saved to:", output_path)
