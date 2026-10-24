"""Training Script for SignBridge SignCNN.

Trains a robust Convolutional Neural Network on the ASL alphabet dataset with:
- PyTorch with Apple Silicon MPS / CUDA / CPU acceleration
- Non-chirality-altering data augmentation
- Learning rate scheduling and Early Stopping
- Checkpointing of the best performing model
- Generation of training curves and confusion matrix
- Model and metadata export for the inference backend
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.architecture import SignCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Select best available hardware acceleration device."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS (Metal Performance Shaders) acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using NVIDIA CUDA acceleration: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU acceleration")
    return device


def get_transforms(img_size: int = 64) -> Tuple[transforms.Compose, transforms.Compose]:
    """Build train and evaluation data transforms.
    
    CRITICAL: Never use RandomHorizontalFlip for sign language because
    mirroring changes chiral hand signs (e.g., left vs right hand orientation).
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.06, 0.06),
            scale=(0.94, 1.06),
        ),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, eval_transform


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch, returning average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate model on validation loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def plot_metrics(
    history: Dict[str, List[float]],
    output_path: Path
):
    """Plot and save training and validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r--s", label="Val Loss", linewidth=2)
    ax1.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Accuracy curve
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Acc", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "g--s", label="Val Acc", linewidth=2)
    ax2.set_title("Training & Validation Accuracy (%)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    logger.info(f"Saved training curves to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train SignBridge SignCNN model.")
    parser.add_argument("--data-dir", type=str, default="ml/dataset/split", help="Path to split dataset folder")
    parser.add_argument("--output-dir", type=str, default="ml/weights", help="Directory to save model and metrics")
    parser.add_argument("--img-size", type=int, default=64, help="Input image dimension (64x64)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for regularization")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Training data not found at {data_dir}. Run prepare_dataset.py first.")

    train_tf, eval_tf = get_transforms(args.img_size)

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=eval_tf)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    logger.info(f"Loaded {len(train_dataset)} training images, {len(val_dataset)} validation images across {num_classes} classes")

    # Save class indices mapping
    class_indices = {i: name for i, name in enumerate(class_names)}
    with open(output_dir / "class_indices.json", "w") as f:
        json.dump(class_indices, f, indent=2)

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )

    # Initialize model
    model = SignCNN(num_classes=num_classes, in_channels=3, img_size=args.img_size).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Initialized SignCNN with {total_params:,} trainable parameters")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0
    start_time = time.time()

    logger.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        lr_current = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch:02d}/{args.epochs:02d} [{epoch_time:.1f}s] - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}% (lr={lr_current:.6f})"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "num_classes": num_classes,
                "class_names": class_names,
                "img_size": args.img_size,
                "in_channels": 3,
                "model_type": "SignCNN",
            }
            torch.save(checkpoint, str(output_dir / "sign_model.pt"))
            logger.info(f" Saved best checkpoint (Val Acc: {val_acc*100:.2f}%) to {output_dir / 'sign_model.pt'}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping triggered after {args.patience} epochs without improvement.")
                break

    total_training_time = time.time() - start_time
    logger.info(f"Training completed in {total_training_time:.1f}s. Best Validation Accuracy: {best_val_acc*100:.2f}%")

    # Plot loss and accuracy curves
    plot_metrics(history, output_dir / "training_curves.png")

    # Save model config metadata
    model_config = {
        "model_type": "SignCNN",
        "num_classes": num_classes,
        "class_names": class_names,
        "img_size": args.img_size,
        "in_channels": 3,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "best_val_acc": float(best_val_acc),
        "epochs_trained": len(history["train_loss"]),
        "total_training_time_sec": float(total_training_time),
        "parameter_count": total_params,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(output_dir / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    logger.info(f"Saved model configuration to {output_dir / 'model_config.json'}")


if __name__ == "__main__":
    main()
