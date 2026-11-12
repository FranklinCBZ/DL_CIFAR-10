
from pathlib import Path
import argparse
import csv
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model import CIFARResNet
from model_se import SEResNet
from kaggle_dataset import get_kaggle_loaders


SEED = 42
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.1

#OUTPUT_DIR = Path("experiments/kaggle_v2")
#OUTPUT_DIR = Path("experiments/kaggle_v3_erasing")
OUTPUT_DIR = Path("experiments/kaggle_v3_se")
BEST_PATH = OUTPUT_DIR / "best_model.pth"
LAST_PATH = OUTPUT_DIR / "last_checkpoint.pth"
HISTORY_PATH = OUTPUT_DIR / "training_history.csv"

FIELDS = [
    "epoch", "train_loss", "train_accuracy",
    "val_loss", "val_accuracy", "learning_rate"
]


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def run_epoch(model, loader, criterion, device,
              optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)

    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda")
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        n = labels.size(0)
        loss_sum += loss.item() * n
        correct += (
            outputs.argmax(1) == labels
        ).sum().item()
        total += n

    return loss_sum / total, 100 * correct / total


def save_history(history):
    with open(HISTORY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(history)


def save_plots(history):
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        epochs, [h["train_accuracy"] for h in history],
        label="Train"
    )
    axes[0].plot(
        epochs, [h["val_accuracy"] for h in history],
        label="Validation"
    )
    axes[0].set_title("Accuracy")
    axes[0].set_ylabel("Accuracy (%)")

    axes[1].plot(
        epochs, [h["train_loss"] for h in history],
        label="Train"
    )
    axes[1].plot(
        epochs, [h["val_loss"] for h in history],
        label="Validation"
    )
    axes[1].set_title("Loss")
    axes[1].set_ylabel("Cross-entropy loss")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "training_analysis.png",
        dpi=200
    )
    plt.close(fig)


def main(resume=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader = get_kaggle_loaders(
        batch_size=BATCH_SIZE,
        num_workers=4
    )

    #model = CIFARResNet(base_width=40).to(device)
    model = SEResNet(base_width=40).to(device)

    num_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    print(f"Parameters: {num_params:,}")
    assert num_params <= 5_000_000

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda")
    )

    start_epoch = 0
    best_accuracy = 0.0
    history = []

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

        start_epoch = checkpoint["epoch"]
        best_accuracy = checkpoint["best_accuracy"]
        history = checkpoint["history"]

        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, scaler=scaler
        )

        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step()

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": lr
        })

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train: {train_acc:.2f}% "
            f"Val: {val_acc:.2f}% "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "best_accuracy": best_accuracy,
                "num_parameters": num_params
            }, BEST_PATH)

            print(
                f"  New best: {best_accuracy:.2f}%"
            )

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_accuracy": best_accuracy,
            "history": history
        }, LAST_PATH)

        save_history(history)
        save_plots(history)

    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Best checkpoint: {BEST_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", action="store_true"
    )
    args = parser.parse_args()
    main(resume=args.resume)
