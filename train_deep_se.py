
from pathlib import Path
import argparse
import csv
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model import CIFARResNet
from model_se import DeepSEResNet
from kaggle_dataset import get_kaggle_loaders


SEED = 42
EPOCHS = 200
MIN_EPOCHS = 100
PATIENCE = 25
MIN_DELTA = 0.02

BATCH_SIZE = 128
LEARNING_RATE = 0.1


OUTPUT_DIR = Path("experiments/kaggle_v6_deep_se")
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

    model = DeepSEResNet(
        num_classes=10,
        base_width=38,
        blocks=(3, 3, 3, 2)
    ).to(device)

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
    best_meaningful_accuracy = 0.0
    epochs_without_improvement = 0
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

        best_meaningful_accuracy = checkpoint.get(
            "best_meaningful_accuracy",
            best_accuracy
        )

        epochs_without_improvement = checkpoint.get(
            "epochs_without_improvement",
            0
        )

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


        # =====================================
        # Save best model
        # =====================================

        if val_acc > best_accuracy:
            best_accuracy = val_acc

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "best_accuracy": best_accuracy,
                "num_parameters": num_params
            }, BEST_PATH)

            print(
                f"  New best model: {best_accuracy:.2f}%"
            )

        # =====================================
        # Early stopping
        # =====================================

        if val_acc > best_meaningful_accuracy + MIN_DELTA:

            best_meaningful_accuracy = val_acc
            epochs_without_improvement = 0

            print(
                f"  Meaningful improvement: "
                f"{val_acc:.2f}%"
            )

        else:
            epochs_without_improvement += 1

        print(
            f"  Early stopping counter: "
            f"{epochs_without_improvement}/{PATIENCE}"
        )

        # =====================================
        # Save resumable checkpoint
        # =====================================

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_accuracy": best_accuracy,
            "best_meaningful_accuracy": best_meaningful_accuracy,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "num_parameters": num_params
        }, LAST_PATH)

        # Save results
        save_history(history)
        save_plots(history)

        # =====================================
        # Stop if validation has plateaued
        # =====================================

        if (
            epoch + 1 >= MIN_EPOCHS
            and epochs_without_improvement >= PATIENCE
        ):
            print("\nEarly stopping triggered!")
            print(f"Stopped at epoch: {epoch + 1}")
            print(f"Best validation accuracy: {best_accuracy:.2f}%")
            break


    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Best checkpoint: {BEST_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", action="store_true"
    )
    args = parser.parse_args()
    main(resume=args.resume)
