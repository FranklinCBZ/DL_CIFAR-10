
import pandas as pd
import matplotlib.pyplot as plt

# Load training history
df = pd.read_csv("results/training_history.csv")

# Find best validation epoch
best_idx = df["val_accuracy"].idxmax()
best_epoch = int(df.loc[best_idx, "epoch"])
best_acc = df.loc[best_idx, "val_accuracy"]

# Create plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(
    df["epoch"],
    df["train_accuracy"],
    label="Training Accuracy",
    linewidth=2
)

axes[0].plot(
    df["epoch"],
    df["val_accuracy"],
    label="Validation Accuracy",
    linewidth=2
)

axes[0].axvline(
    best_epoch,
    linestyle="--",
    color="gray",
    label=f"Best Epoch: {best_epoch}"
)

axes[0].set_title("Training vs Validation Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy (%)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Loss
axes[1].plot(
    df["epoch"],
    df["train_loss"],
    label="Training Loss",
    linewidth=2
)

axes[1].plot(
    df["epoch"],
    df["val_loss"],
    label="Validation Loss",
    linewidth=2
)

axes[1].axvline(
    best_epoch,
    linestyle="--",
    color="gray"
)

axes[1].set_title("Training vs Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle(
    f"Modified ResNet-18 | Best Validation: "
    f"{best_acc:.2f}% (Epoch {best_epoch})"
)

plt.tight_layout()
plt.savefig(
    "results/training_analysis.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print(f"Best validation epoch: {best_epoch}")
print(f"Best validation accuracy: {best_acc:.2f}%")
print(
    f"Final train-validation gap: "
    f"{df['train_accuracy'].iloc[-1] - df['val_accuracy'].iloc[-1]:.2f}%"
)
