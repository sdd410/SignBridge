"""Model Evaluation Script for SignBridge.

Evaluates trained SignCNN model on the test split, computing:
- Top-1 and Top-3 accuracy
- Per-class precision, recall, F1 score
- Macro and weighted average metrics
- Confusion matrix visualization (PNG) and matrix data (JSON)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.architecture import SignCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: Path):
    """Plot and save confusion matrix as a high-resolution heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="SignBridge CNN Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val > 0:
                ax.text(j, i, format(val, "d"),
                        ha="center", va="center",
                        color="white" if val > thresh else "black",
                        fontsize=7)

    fig.tight_layout()
    plt.savefig(str(output_path), dpi=180)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {output_path}")


def evaluate_model(
    checkpoint_path: Path,
    test_dir: Path,
    output_dir: Path,
    device: torch.device
) -> Dict[str, Any]:
    """Run full evaluation on the test split."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found at {test_dir}")

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    num_classes = checkpoint["num_classes"]
    class_names = checkpoint["class_names"]
    img_size = checkpoint.get("img_size", 64)

    # Initialize model and load weights
    model = SignCNN(num_classes=num_classes, in_channels=3, img_size=img_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(str(test_dir), transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    all_preds = []
    all_targets = []
    all_top3 = []

    logger.info(f"Running evaluation on {len(test_dataset)} test samples...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=-1)

            top1 = torch.argmax(probs, dim=-1).cpu().numpy()
            top3 = torch.topk(probs, k=min(3, num_classes), dim=-1).indices.cpu().numpy()

            all_preds.extend(top1)
            all_targets.extend(labels.numpy())
            all_top3.extend(top3)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate Top-1 and Top-3 accuracy
    top1_accuracy = float(np.mean(all_preds == all_targets))
    top3_correct = sum(target in top3_row for target, top3_row in zip(all_targets, all_top3))
    top3_accuracy = float(top3_correct / len(all_targets))

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))
    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")

    # Classification report
    clf_report = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    metrics = {
        "top1_accuracy": round(top1_accuracy, 4),
        "top3_accuracy": round(top3_accuracy, 4),
        "macro_avg_f1": round(clf_report["macro avg"]["f1-score"], 4),
        "weighted_avg_f1": round(clf_report["weighted avg"]["f1-score"], 4),
        "total_test_samples": len(all_targets),
        "class_count": num_classes,
        "classes": class_names,
        "per_class_metrics": {
            cls: {
                "precision": round(clf_report[cls]["precision"], 4),
                "recall": round(clf_report[cls]["recall"], 4),
                "f1_score": round(clf_report[cls]["f1-score"], 4),
                "support": clf_report[cls]["support"],
            }
            for cls in class_names if cls in clf_report
        },
        "confusion_matrix": cm.tolist()
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Evaluation complete:")
    logger.info(f"  Top-1 Accuracy: {top1_accuracy * 100:.2f}%")
    logger.info(f"  Top-3 Accuracy: {top3_accuracy * 100:.2f}%")
    logger.info(f"  Macro F1-Score: {metrics['macro_avg_f1']:.4f}")
    logger.info(f"Saved full metrics report to {metrics_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate SignBridge SignCNN model.")
    parser.add_argument("--checkpoint", type=str, default="ml/weights/sign_model.pt", help="Path to checkpoint")
    parser.add_argument("--test-dir", type=str, default="ml/dataset/split/test", help="Path to test split")
    parser.add_argument("--output-dir", type=str, default="ml/weights", help="Directory to save evaluation results")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    evaluate_model(Path(args.checkpoint), Path(args.test_dir), Path(args.output_dir), device)


if __name__ == "__main__":
    main()
