
import os
import csv
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model import CIFARResNet
from dataset import get_dataloaders


# -----------------------------
# Configuration
# -----------------------------
SEED = 42
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 4

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# -----------------------------
# Dataset
# -----------------------------
train_loader, val_loader = get_dataloaders(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)


# -----------------------------
# Model
# -----------------------------
model = CIFARResNet(base_width=40).to(device)

num_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print(f"Trainable parameters: {num_params:,}")

assert num_params <= 5_000_000, "Parameter limit exceeded!"


# -----------------------------
# Optimizer and scheduler
# -----------------------------
criterion = nn.CrossEntropyLoss()
#criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    nesterov=True
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

scaler = torch.amp.GradScaler(
    "cuda", enabled=(device.type == "cuda")
)


# -----------------------------
# Training
# -----------------------------
def train_one_epoch():
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

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

    return total_loss / total, 100 * correct / total


# -----------------------------
# Validation
# -----------------------------
@torch.no_grad()
def validate():
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda")
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        total += batch_size

    return total_loss / total, 100 * correct / total


# -----------------------------
# Main training loop
# -----------------------------
def main():
    best_accuracy = 0.0
    history = []

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch()
        val_loss, val_acc = validate()

        scheduler.step()

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_accuracy:
            best_accuracy = val_acc

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_accuracy": best_accuracy,
                "num_parameters": num_params
            }, "checkpoints/best_model.pth")

            print(
                f"  Best model saved! "
                f"Accuracy: {best_accuracy:.2f}%"
            )

        # Save metrics after each epoch
        with open(
            "results/training_history.csv",
            "w",
            newline=""
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=history[0].keys()
            )
            writer.writeheader()
            writer.writerows(history)

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Trainable parameters: {num_params:,}")

    # Generate plots
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        [row["train_accuracy"] for row in history],
        label="Training"
    )
    plt.plot(
        epochs,
        [row["val_accuracy"] for row in history],
        label="Validation"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/accuracy.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        [row["train_loss"] for row in history],
        label="Training"
    )
    plt.plot(
        epochs,
        [row["val_loss"] for row in history],
        label="Validation"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/loss.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
