
from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history",
        type=str,
        default="experiments/kaggle_v11_v7_full50k/training_history.csv"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="V11"
    )
    args = parser.parse_args()

    history_path = Path(args.history)
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    df = pd.read_csv(history_path)

    print("Columns:", df.columns.tolist())
    print("Total epochs:", len(df))

    epochs = df["epoch"]

    # ===================================
    # Accuracy graph
    # ===================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        df["train_accuracy"],
        label="Training Accuracy",
        linewidth=2
    )

    if "val_accuracy" in df.columns:
        plt.plot(
            epochs,
            df["val_accuracy"],
            label="Validation Accuracy",
            linewidth=2
        )

        best_idx = df["val_accuracy"].idxmax()
        best_epoch = df.loc[best_idx, "epoch"]
        best_acc = df.loc[best_idx, "val_accuracy"]

        plt.scatter(
            best_epoch,
            best_acc,
            color="red",
            s=70,
            zorder=5,
            label=f"Best Val: {best_acc:.2f}%"
        )

    plt.title(f"{args.name} ResNet - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    acc_path = output_dir / f"{args.name}_accuracy.png"
    plt.savefig(acc_path, dpi=300)
    plt.close()

    # ===================================
    # Loss graph
    # ===================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        df["train_loss"],
        label="Training Loss",
        linewidth=2
    )

    if "val_loss" in df.columns:
        plt.plot(
            epochs,
            df["val_loss"],
            label="Validation Loss",
            linewidth=2
        )

    plt.title(f"{args.name} ResNet - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    loss_path = output_dir / f"{args.name}_loss.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()

    # ===================================
    # Summary
    # ===================================

    print("\nGraphs generated:")
    print(acc_path)
    print(loss_path)

    print(
        "Final training accuracy:",
        f"{df['train_accuracy'].iloc[-1]:.2f}%"
    )

    if "val_accuracy" in df.columns:
        print(
            "Best validation accuracy:",
            f"{df['val_accuracy'].max():.2f}%"
        )


if __name__ == "__main__":
    main()
