
from pathlib import Path
import argparse
import csv
import random

import numpy as np
import torch
import torch.nn as nn

from model_v13_preact import PreActDeepSEResNetDropout
from kaggle_dataset_alternating import get_alternating_loader


# ==========================================
# Configuration
# ==========================================

SEED = 42

PREVIOUS_EPOCHS = 250
EXTRA_EPOCHS = 30

BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4

SOURCE_CHECKPOINT = Path(
    "experiments/kaggle_v13_preact_alt_full50k/"
    "model_epoch_250.pth"
)

OUTPUT_DIR = Path(
    "experiments/kaggle_v13_preact_finetuned_280"
)

LAST_PATH = OUTPUT_DIR / "last_checkpoint.pth"
FINAL_PATH = OUTPUT_DIR / "final_model.pth"
HISTORY_PATH = OUTPUT_DIR / "training_history.csv"

FIELDS = [
    "epoch",
    "train_loss",
    "train_accuracy",
    "learning_rate"
]


# ==========================================
# Reproducibility
# ==========================================

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ==========================================
# Train one epoch
# ==========================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(
            device, non_blocking=True
        )
        labels = labels.to(
            device, non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda")
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()

        total += batch_size

    return (
        total_loss / total,
        100.0 * correct / total
    )


# ==========================================
# Save history
# ==========================================

def save_history(history):
    with open(HISTORY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=FIELDS
        )
        writer.writeheader()
        writer.writerows(history)


# ==========================================
# Fine-tune V13
# ==========================================

def finetune_v13(resume=False):
    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )
        torch.backends.cudnn.benchmark = True

    # Protect existing fine-tuning results
    if not resume and (
        LAST_PATH.exists() or FINAL_PATH.exists()
    ):
        raise FileExistsError(
            f"Checkpoints already exist in {OUTPUT_DIR}. "
            "Use --resume or choose a new directory."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Same training data and alternating flips
    train_loader, epoch_value = get_alternating_loader(
        batch_size=BATCH_SIZE,
        num_workers=4
    )

    print(
        "Training images:",
        len(train_loader.dataset)
    )

    # Exact same architecture as V13
    model = PreActDeepSEResNetDropout(
        num_classes=10,
        base_width=39,
        blocks=(3, 3, 3, 2),
        dropout=0.15
    ).to(device)

    num_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Parameters: {num_params:,}")

    assert num_params <= 5_000_000

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    # New optimizer for fine-tuning
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
        nesterov=True
    )

    # New 30-epoch cosine schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EXTRA_EPOCHS
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda")
    )

    start_epoch = 0
    history = []

    # ======================================
    # Resume an interrupted fine-tuning run
    # ======================================

    if resume:
        checkpoint = torch.load(
            LAST_PATH,
            map_location=device,
            weights_only=True
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

        start_epoch = checkpoint["local_epoch"]
        history = checkpoint["history"]

        print(
            f"Resuming from global epoch "
            f"{PREVIOUS_EPOCHS + start_epoch}"
        )

    # ======================================
    # Start from original epoch-250 weights
    # ======================================

    else:
        checkpoint = torch.load(
            SOURCE_CHECKPOINT,
            map_location=device,
            weights_only=True
        )

        assert checkpoint["epoch"] == PREVIOUS_EPOCHS, (
            f"Expected epoch {PREVIOUS_EPOCHS}, "
            f"found {checkpoint['epoch']}"
        )

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )

        print("\nOriginal V13 weights loaded!")
        print("Checkpoint:", SOURCE_CHECKPOINT)
        print("Starting from epoch:", checkpoint["epoch"])

    # ======================================
    # Fine-tuning loop
    # ======================================

    for local_epoch in range(
        start_epoch, EXTRA_EPOCHS
    ):
        global_epoch = (
            PREVIOUS_EPOCHS + local_epoch + 1
        )

        # Keep alternating-flip parity consistent
        # with the original 250-epoch training run.
        epoch_value.value = global_epoch - 1

        lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device
        )

        scheduler.step()

        history.append({
            "epoch": global_epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "learning_rate": lr
        })

        print(
            f"Epoch [{global_epoch}/280] "
            f"Train: {train_acc:.2f}% "
            f"Loss: {train_loss:.4f} "
            f"LR: {lr:.6f}"
        )

        # Resumable checkpoint
        torch.save({
            "epoch": global_epoch,
            "local_epoch": local_epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "history": history,
            "num_parameters": num_params
        }, LAST_PATH)

        save_history(history)

        # Keep intermediate snapshots
        if global_epoch in (265, 280):
            snapshot_path = OUTPUT_DIR / (
                f"model_epoch_{global_epoch}.pth"
            )

            torch.save({
                "epoch": global_epoch,
                "model_state_dict": model.state_dict(),
                "num_parameters": num_params
            }, snapshot_path)

            print(
                "  Saved snapshot:",
                snapshot_path
            )

    # ======================================
    # Save final model
    # ======================================

    torch.save({
        "epoch": PREVIOUS_EPOCHS + EXTRA_EPOCHS,
        "model_state_dict": model.state_dict(),
        "num_parameters": num_params
    }, FINAL_PATH)

    print("\nFine-tuning completed!")
    print("Final epoch:", 280)
    print("Final checkpoint:", FINAL_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resume",
        action="store_true"
    )

    args = parser.parse_args()

    finetune_v13(resume=args.resume)
