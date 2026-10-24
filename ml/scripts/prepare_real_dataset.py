"""Convert Real-Hand ASL Datasets (Sign Language MNIST and Custom Images)

Processes 27,455 real human hand gesture images into 64x64 RGB images
organized into train/val/test splits, and seamlessly integrates any custom real-hand
images provided by the user.
"""

import argparse
import csv
import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Label mapping for Sign Language MNIST (24 classes: A-Y excluding J and Z)
MNIST_LABEL_MAP = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H",
    8: "I", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P",
    16: "Q", 17: "R", 18: "S", 19: "T", 20: "U", 21: "V", 22: "W",
    23: "X", 24: "Y"
}


def convert_mnist_csv(
    csv_path: Path,
    output_raw_dir: Path,
    max_per_class: int = 400,
    img_size: int = 64
) -> Dict[str, int]:
    """Convert Sign Language MNIST CSV into real hand JPEG images."""
    output_raw_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}

    logger.info(f"Converting real-hand CSV from {csv_path} (max {max_per_class} per class)...")
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip label, pixel1...

        for row_idx, row in enumerate(reader):
            if not row:
                continue
            label_id = int(row[0])
            if label_id not in MNIST_LABEL_MAP:
                continue

            class_name = MNIST_LABEL_MAP[label_id]
            current_count = counts.get(class_name, 0)
            if current_count >= max_per_class:
                continue

            # Extract 28x28 pixels
            pixels = np.array([int(p) for p in row[1:]], dtype=np.uint8).reshape((28, 28))
            
            # Upscale smoothly to target img_size (e.g. 64x64)
            resized = cv2.resize(pixels, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            
            # Convert grayscale to 3-channel BGR
            bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

            class_dir = output_raw_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            img_path = class_dir / f"mnist_{class_name}_{current_count:04d}.jpg"
            cv2.imwrite(str(img_path), bgr)

            counts[class_name] = current_count + 1

    total_images = sum(counts.values())
    logger.info(f"Extracted {total_images} real hand images across {len(counts)} classes to {output_raw_dir}")
    return counts


def merge_custom_dataset(custom_dir: Path, raw_dir: Path) -> int:
    """Merge user-supplied real hand images from custom directory into raw directory."""
    if not custom_dir.exists():
        return 0

    copied = 0
    for class_folder in custom_dir.iterdir():
        if not class_folder.is_dir() or class_folder.name.startswith("."):
            continue
        dest_folder = raw_dir / class_folder.name
        dest_folder.mkdir(parents=True, exist_ok=True)

        for img in class_folder.glob("*"):
            if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                shutil.copy2(img, dest_folder / f"custom_{img.name}")
                copied += 1

    if copied > 0:
        logger.info(f"Merged {copied} custom user hand images from {custom_dir}")
    return copied


def split_real_dataset(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, int]:
    """Split real hand dataset into train, val, and test subsets."""
    random.seed(seed)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    # Clean existing split
    for d in [train_dir, val_dir, test_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    summary = {"train": 0, "val": 0, "test": 0, "classes": 0}
    class_folders = sorted([d for d in source_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    summary["classes"] = len(class_folders)

    for cdir in class_folders:
        cname = cdir.name
        (train_dir / cname).mkdir(exist_ok=True)
        (val_dir / cname).mkdir(exist_ok=True)
        (test_dir / cname).mkdir(exist_ok=True)

        images = [f for f in cdir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        for f in images[:n_train]:
            shutil.copy2(f, train_dir / cname / f.name)
            summary["train"] += 1
        for f in images[n_train:n_train + n_val]:
            shutil.copy2(f, val_dir / cname / f.name)
            summary["val"] += 1
        for f in images[n_train + n_val:]:
            shutil.copy2(f, test_dir / cname / f.name)
            summary["test"] += 1

    logger.info(f"Split complete: Train={summary['train']}, Val={summary['val']}, Test={summary['test']}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare real hand dataset for SignBridge.")
    parser.add_argument("--csv", type=str, default="ml/dataset/real_mnist/sign_mnist_train.csv", help="Path to MNIST CSV")
    parser.add_argument("--custom-dir", type=str, default="ml/dataset/custom", help="Path to user custom hand images")
    parser.add_argument("--raw-dir", type=str, default="ml/dataset/real_raw", help="Path to raw merged real images")
    parser.add_argument("--split-dir", type=str, default="ml/dataset/split", help="Output split directory")
    parser.add_argument("--max-per-class", type=int, default=300, help="Max images per class from CSV")
    parser.add_argument("--img-size", type=int, default=64, help="Target image size")
    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    split_path = Path(args.split_dir)
    csv_path = Path(args.csv)
    custom_path = Path(args.custom_dir)

    if csv_path.exists():
        convert_mnist_csv(csv_path, raw_path, max_per_class=args.max_per_class, img_size=args.img_size)
    else:
        logger.warning(f"CSV file not found at {csv_path}")

    # Merge any user-provided custom hand images
    merge_custom_dataset(custom_path, raw_path)

    # Perform train/val/test split
    split_real_dataset(raw_path, split_path)

    # Update class indices mapping
    class_names = sorted([d.name for d in (split_path / "train").iterdir() if d.is_dir()])
    mapping = {i: name for i, name in enumerate(class_names)}
    with open(split_path / "class_indices.json", "w") as f:
        json.dump(mapping, f, indent=2)
    logger.info(f"Saved class indices mapping ({len(class_names)} classes) to {split_path / 'class_indices.json'}")


if __name__ == "__main__":
    main()
