# SignBridge Machine Learning Pipeline

## Overview

The SignBridge ML pipeline trains and evaluates a Convolutional Neural Network (**SignCNN**) for American Sign Language (ASL) alphabet classification. The architecture is lightweight (1.48M parameters), specifically optimized for low-latency real-time inference on CPU and Apple Silicon MPS hardware.

## Supported Classes (29 Total)

- **Alphabet (A–Z)**: 26 static hand gestures
- **del**: Backspace / delete character action
- **space**: Word separator action
- **nothing**: Background / neutral / no hand in frame

## Downloading Full External Datasets

To train on the complete 87,000-image Kaggle ASL Alphabet benchmark dataset (by Akash Nagaraj / grassknoted):

```bash
# 1. Install Kaggle CLI
pip install kaggle

# 2. Download and extract raw dataset
kaggle datasets download -d grassknoted/asl-alphabet -p ml/dataset/raw --unzip
```

Alternatively, if no external dataset is downloaded, `prepare_dataset.py` automatically generates a balanced starter gesture dataset with 29 classes to enable autonomous end-to-end training and continuous integration.

## Dataset Preparation & Splitting

```bash
python ml/scripts/prepare_dataset.py --samples-per-class 120 --img-size 64
```
This splits the dataset into:
- **Train Split (70%)**
- **Validation Split (15%)**
- **Test Split (15%)**

## Model Training

```bash
python ml/scripts/train.py --epochs 15 --batch-size 32 --lr 0.001
```

Features:
- Hardware acceleration (Apple Silicon MPS / CUDA / CPU)
- Chirality-preserving data augmentations (RandomAffine ±10°, slight scale/jitter; NO horizontal flip)
- Learning rate scheduling with `ReduceLROnPlateau`
- Early stopping & best checkpoint serialization to `ml/weights/sign_model.pt`
- Automatic generation of training loss and accuracy curves (`training_curves.png`)

## Model Evaluation

```bash
python ml/scripts/evaluate.py --checkpoint ml/weights/sign_model.pt --test-dir ml/dataset/split/test
```

Outputs:
- Top-1 and Top-3 accuracy
- Per-class precision, recall, and F1 scores
- High-resolution confusion matrix heatmap (`ml/weights/confusion_matrix.png`)
- Serialized metrics JSON (`ml/weights/metrics.json`)
