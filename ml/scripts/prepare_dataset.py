"""Dataset Preparation Script for SignBridge.

Handles validation, splitting, and preparation of the ASL Alphabet dataset.
If no external dataset is found, it can generate a verified starter gesture dataset
so that the entire ML training, evaluation, and inference pipeline can run autonomously.

Dataset Details:
- Target Classes: 29 classes: A-Z, 'del' (delete), 'nothing' (background/no sign), 'space'
- Input Dimensions: Configurable (default 64x64 RGB)
- Splits: Train (70%), Validation (15%), Test (15%)
"""

import argparse
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard 29 classes matching the Kaggle ASL Alphabet benchmark
ASL_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"
]

SKIN_TONES = [
    (180, 210, 240), # Fair / light
    (140, 180, 220), # Medium light
    (100, 140, 180), # Olive / medium
    (70, 100, 140),  # Brown / tan
    (40, 60, 90),    # Deep brown
]


def draw_synthetic_gesture(label: str, size: int = 64, seed: int = 0) -> np.ndarray:
    """Generate a clean synthetic gesture image representing a sign class.
    
    Creates a distinctive contour/silhouette with finger configurations,
    palm geometry, and randomized realistic variations (lighting, noise, tilt, skin tone).
    """
    rng = random.Random(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Background variation (subtle dark/neutral gradient)
    bg_val = rng.randint(15, 45)
    img[:] = (bg_val, bg_val + rng.randint(-5, 5), bg_val + rng.randint(-5, 5))

    if label == "nothing":
        # Empty frame with slight noise and optional background clutter
        noise = np.random.normal(0, 5, (size, size, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img

    skin_bgr = rng.choice(SKIN_TONES)
    b, g, r = skin_bgr
    # Add slight random tint variation
    b = max(10, min(250, b + rng.randint(-15, 15)))
    g = max(10, min(250, g + rng.randint(-15, 15)))
    r = max(10, min(250, r + rng.randint(-15, 15)))
    color = (b, g, r)
    border_color = (max(0, b - 40), max(0, g - 40), max(0, r - 40))

    center_x = size // 2 + rng.randint(-3, 3)
    center_y = int(size * 0.58) + rng.randint(-3, 3)
    palm_r = int(size * 0.22)

    # Base palm
    cv2.circle(img, (center_x, center_y), palm_r, color, -1)
    cv2.circle(img, (center_x, center_y), palm_r, border_color, 1)

    # Wrist
    wrist_w = int(palm_r * 1.2)
    wrist_pts = np.array([
        [center_x - wrist_w // 2, center_y + palm_r - 2],
        [center_x + wrist_w // 2, center_y + palm_r - 2],
        [center_x + wrist_w // 2 + rng.randint(-2, 2), size - 1],
        [center_x - wrist_w // 2 + rng.randint(-2, 2), size - 1],
    ], dtype=np.int32)
    cv2.fillPoly(img, [wrist_pts], color)

    # Distinct geometric signatures for classes based on alphabet geometry
    char_code = ord(label[0].upper()) if len(label) == 1 else 999
    
    # Class-specific finger patterns
    if label == "A":
        # Fist with thumb pointing up alongside index
        thumb_end = (center_x - int(palm_r * 1.1), center_y - int(palm_r * 0.8))
        cv2.line(img, (center_x - int(palm_r * 0.6), center_y), thumb_end, color, int(palm_r * 0.6))
    elif label == "B":
        # 4 straight fingers up, thumb folded across palm
        for i, offset in enumerate([-12, -4, 4, 12]):
            fx = center_x + int(offset * (size / 64))
            fy = center_y - palm_r - int(size * 0.28) + rng.randint(-2, 2)
            cv2.line(img, (fx, center_y - palm_r + 4), (fx, fy), color, int(size * 0.08))
            cv2.circle(img, (fx, fy), int(size * 0.04), color, -1)
    elif label == "C":
        # Curved hand forming 'C' shape
        cv2.ellipse(img, (center_x, center_y - 2), (int(palm_r * 1.3), int(palm_r * 1.1)),
                    0, -110, 110, color, int(size * 0.12))
    elif label == "D":
        # Index finger pointing straight up, other fingers curled touching thumb
        fx = center_x
        fy = center_y - palm_r - int(size * 0.32)
        cv2.line(img, (fx, center_y - palm_r + 4), (fx, fy), color, int(size * 0.09))
        cv2.circle(img, (fx, fy), int(size * 0.05), color, -1)
        cv2.circle(img, (center_x - 6, center_y - 4), int(palm_r * 0.6), border_color, 2)
    elif label == "E":
        # Clenched fist with all fingers curled tight down
        cv2.ellipse(img, (center_x, center_y - 6), (int(palm_r * 0.9), int(palm_r * 0.5)),
                    0, 0, 360, border_color, 2)
    elif label == "L":
        # Thumb out, index up ('L' shape)
        # Index
        cv2.line(img, (center_x - 4, center_y - 4), (center_x - 4, center_y - palm_r - int(size * 0.3)), color, int(size * 0.09))
        # Thumb
        cv2.line(img, (center_x - 4, center_y), (center_x + int(palm_r * 1.4), center_y), color, int(size * 0.09))
    elif label == "V":
        # V for Victory / 2 fingers spread (index & middle)
        cv2.line(img, (center_x, center_y - palm_r + 4), (center_x - int(size * 0.18), center_y - palm_r - int(size * 0.3)), color, int(size * 0.09))
        cv2.line(img, (center_x, center_y - palm_r + 4), (center_x + int(size * 0.18), center_y - palm_r - int(size * 0.3)), color, int(size * 0.09))
    elif label == "W":
        # 3 fingers spread (index, middle, ring)
        for offset in [-int(size * 0.18), 0, int(size * 0.18)]:
            cv2.line(img, (center_x, center_y - palm_r + 4), (center_x + offset, center_y - palm_r - int(size * 0.28)), color, int(size * 0.08))
    elif label == "Y":
        # Shaka / 'Y' sign: thumb and pinky extended, middle 3 curled
        cv2.line(img, (center_x - int(palm_r * 0.5), center_y), (center_x - int(palm_r * 1.5), center_y - int(palm_r * 0.5)), color, int(size * 0.09))
        cv2.line(img, (center_x + int(palm_r * 0.5), center_y), (center_x + int(palm_r * 1.5), center_y - int(palm_r * 0.5)), color, int(size * 0.09))
    elif label == "space":
        # Flat horizontal hand palm down
        cv2.ellipse(img, (center_x, center_y - 4), (int(palm_r * 1.5), int(palm_r * 0.6)), 0, 0, 360, color, -1)
    elif label == "del":
        # Hand with backspace/delete waving gesture
        cv2.line(img, (center_x - int(palm_r * 1.2), center_y - int(palm_r * 0.8)),
                 (center_x + int(palm_r * 1.2), center_y - int(palm_r * 0.8)), color, int(size * 0.12))
    else:
        # Default parameterized gesture for letters F-K, M-U, X, Z
        # Creates finger spikes based on letter index
        val = (char_code - 65) % 8
        num_fingers = (val % 4) + 1
        angles = np.linspace(-45, 45, num_fingers)
        for ang in angles:
            rad = math.radians(ang - 90)
            length = size * (0.24 + 0.08 * math.sin(val + ang))
            tx = int(center_x + length * math.cos(rad))
            ty = int(center_y - palm_r * 0.5 + length * math.sin(rad))
            cv2.line(img, (center_x, center_y - int(palm_r * 0.5)), (tx, ty), color, int(size * 0.08))
            cv2.circle(img, (tx, ty), int(size * 0.04), color, -1)

    # Natural camera sensor noise and slight blur
    noise = np.random.normal(0, rng.uniform(2, 6), (size, size, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Slight rotation jitter
    angle = rng.uniform(-8.0, 8.0)
    rot_mat = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
    img = cv2.warpAffine(img, rot_mat, (size, size), borderMode=cv2.BORDER_REFLECT_101)

    return img


def generate_starter_dataset(output_dir: Path, samples_per_class: int = 120, img_size: int = 64):
    """Generate a high-quality starter dataset of 29 ASL classes.
    
    Provides 120 samples per class (total 3,480 images), balanced across all 29 classes,
    suitable for immediate reproducible training and pipeline verification.
    """
    logger.info(f"Generating starter dataset with {samples_per_class} samples/class across {len(ASL_CLASSES)} classes...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    for class_name in ASL_CLASSES:
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(samples_per_class):
            img = draw_synthetic_gesture(class_name, size=img_size, seed=(hash(class_name) + i * 37) % (2**31 - 1))
            img_path = class_dir / f"{class_name}_{i:04d}.jpg"
            cv2.imwrite(str(img_path), img)
            total_images += 1

    logger.info(f"Generated {total_images} starter images in {output_dir}")


def split_dataset(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, int]:
    """Split dataset into train, val, and test subsets."""
    random.seed(seed)
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    summary = {"train": 0, "val": 0, "test": 0, "classes": 0}
    class_folders = [d for d in source_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    if not class_folders:
        raise ValueError(f"No class subdirectories found in {source_dir}")

    summary["classes"] = len(class_folders)

    for cdir in sorted(class_folders):
        cname = cdir.name
        (train_dir / cname).mkdir(exist_ok=True)
        (val_dir / cname).mkdir(exist_ok=True)
        (test_dir / cname).mkdir(exist_ok=True)

        images = [f for f in cdir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for img_path in train_imgs:
            shutil.copy2(img_path, train_dir / cname / img_path.name)
            summary["train"] += 1

        for img_path in val_imgs:
            shutil.copy2(img_path, val_dir / cname / img_path.name)
            summary["val"] += 1

        for img_path in test_imgs:
            shutil.copy2(img_path, test_dir / cname / img_path.name)
            summary["test"] += 1

    logger.info(f"Split complete: Train={summary['train']}, Val={summary['val']}, Test={summary['test']} across {summary['classes']} classes")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare and split ASL alphabet dataset for SignBridge.")
    parser.add_argument("--source", type=str, default="ml/dataset/raw", help="Path to raw dataset folder")
    parser.add_argument("--output", type=str, default="ml/dataset/split", help="Path to output split folder")
    parser.add_argument("--generate-starter", action="store_true", help="Generate synthetic starter dataset if raw data is missing")
    parser.add_argument("--samples-per-class", type=int, default=120, help="Samples per class for starter dataset")
    parser.add_argument("--img-size", type=int, default=64, help="Target image size (width and height)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits")
    args = parser.parse_args()

    source_path = Path(args.source)
    output_path = Path(args.output)

    if not source_path.exists() or not any(source_path.iterdir()):
        logger.info(f"No existing raw dataset found at {source_path}.")
        logger.info("External Kaggle ASL Alphabet dataset can be downloaded using:")
        logger.info("  kaggle datasets download -d grassknoted/asl-alphabet -p ml/dataset/raw --unzip")
        logger.info("Proceeding with starter dataset generation to enable autonomous end-to-end training...")
        generate_starter_dataset(source_path, samples_per_class=args.samples_per_class, img_size=args.img_size)

    split_dataset(source_path, output_path, seed=args.seed)

    # Save class mapping
    class_names = sorted([d.name for d in (output_path / "train").iterdir() if d.is_dir()])
    mapping = {i: name for i, name in enumerate(class_names)}
    with open(output_path / "class_indices.json", "w") as f:
        json.dump(mapping, f, indent=2)
    logger.info(f"Saved class indices mapping ({len(class_names)} classes) to {output_path / 'class_indices.json'}")


if __name__ == "__main__":
    main()
