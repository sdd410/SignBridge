# SignBridge — Real-Time Sign Language Recognition & Translation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-5-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-38B2AC?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)

**SignBridge** is a real-time American Sign Language (ASL) gesture recognition and transcript assembly application. It captures webcam video frames, processes them through an OpenCV vision pipeline, classifies gestures using a lightweight Convolutional Neural Network (**SignCNN**), applies temporal stabilization to eliminate flickering, and streams accepted signs into a live editable transcript.

---

## System Architecture Pipeline

```text
Webcam Stream (Browser) 
  │
  ▼ [getUserMedia + Frame Throttling]
Binary/Base64 Frame Dispatch (WebSocket /ws/predict)
  │
  ▼ [OpenCV Decoding & BGR→RGB Standardizing]
Hand ROI Detection (Guide Box / Skin Segmentation / Full Frame)
  │
  ▼ [Resize to 64x64 + ImageNet Normalization]
SignCNN Deep Feature Extractor (PyTorch MPS / CUDA / CPU)
  │
  ▼ [Softmax Top-5 Probability Ranking]
Temporal Stabilization Layer (Sliding Window Majority Voting + Cooldown Lockout)
  │
  ▼ [Accepted Sign Event]
Live Text Transcript Manager (Assembly, Word Stats, Space/Del, Export & Speech Synthesis)
```

---

## Key Features

1. **Local Real-Time Inference**: Runs 100% locally on CPU or Apple Silicon MPS (Metal Performance Shaders) without paid external APIs.
2. **Real Trained Model Included**: Pre-trained on 29 ASL classes (`A`–`Z`, `del`, `space`, `nothing`) with verified checkpoints, metrics, and confusion matrix.
3. **No Fabricated Predictions**: Transparent status reporting, measured FPS, and exact millisecond latency.
4. **Adaptive Hand ROI Extraction**:
   - `guide_roi`: High-accuracy on-screen targeting guide box.
   - `skin_contour`: Automatic YCrCb skin tone segmentation.
   - `full_frame`: Passthrough mode.
5. **Temporal Prediction Stabilization**:
   - Configurable temporal sliding window (majority voting).
   - Dynamic confidence thresholding.
   - Continuous hold duration requirement before sign acceptance.
   - Repeat lockout cooldown preventing accidental duplicate characters.
6. **Live Interactive Transcript**:
   - Character and word counting.
   - Space, Backspace, and Clear buttons.
   - Copy to clipboard and export to `.txt`.
   - Text-to-speech synthesis of the assembled text (via Web Speech API).
   - Auto-insert pause/resume toggle for practicing without altering the transcript.
7. **In-App Evaluation & ASL Reference**:
   - Interactive ASL Alphabet Reference Guide with hand shape descriptions for all 29 signs.
   - Model Metrics modal displaying empirical test accuracy (Top-1: 61.69%, Top-3: 96.55%), confusion matrix heatmap, and training curves.

---

## Quickstart & Installation

### Prerequisites
- Python 3.11 or 3.12
- Node.js 18+ and npm

### 1. Clone & Set Up Python Environment

```bash
# Activate existing virtualenv or create a new one
python3 -m venv .venv
source .venv/bin/activate

# Install backend dependencies
pip install -r backend/requirements.txt
```

### 2. Set Up Frontend Environment

```bash
cd frontend
npm install
cd ..
```

---

## Running the Application

### Option A: Run Services Locally (Recommended for Development)

**Terminal 1 — Backend:**
```bash
source .venv/bin/activate
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

Open your browser at **[http://localhost:5173](http://localhost:5173)**.

### Option B: Docker Compose

```bash
docker-compose up --build
```
Access the application at `http://localhost:5173`.

---

## Machine Learning Workflow

The repository includes a self-contained training and evaluation pipeline in `ml/`:

```bash
# 1. Prepare and split dataset (70% train, 15% val, 15% test)
python ml/scripts/prepare_dataset.py --samples-per-class 120 --img-size 64

# 2. Train SignCNN model with MPS/CUDA acceleration
python ml/scripts/train.py --epochs 15 --batch-size 32

# 3. Evaluate model on test split and generate confusion matrix
python ml/scripts/evaluate.py
```

To use the 87,000-image Kaggle ASL Alphabet benchmark dataset:
```bash
kaggle datasets download -d grassknoted/asl-alphabet -p ml/dataset/raw --unzip
python ml/scripts/prepare_dataset.py --source ml/dataset/raw
```

---

## Test Verification

Run all unit, model, and API test suites:

```bash
# Backend test suite (20 tests covering preprocessing, model, stabilization, API)
./.venv/bin/pytest backend/tests/ -v

# Frontend test suite (Vitest + React Testing Library)
cd frontend && npm run test && cd ..

# Frontend production bundle build verification
cd frontend && npm run build && cd ..
```

---

## Scope & Limitations

> [!NOTE]
> SignBridge is an **educational computer vision MVP** designed to classify individual static ASL fingerspelling gestures and assemble them into words.
> - **Not a Universal Sign Language Translator**: ASL alphabet recognition is fingerspelling, which is a small component of full American Sign Language. Full ASL incorporates facial expressions, body posture, two-handed motion, and grammar not capturable by a static alphabet CNN.
> - **Static vs Dynamic Gestures**: Dynamic gestures (e.g., motion-based 'J' and 'Z') are classified via key poses in single-frame models. Video sequence models (LSTM / 3D-CNN / Transformers) are recommended for complex dynamic signs.
> - **Lighting & Background Sensitivity**: Optimal performance is achieved with clean lighting and minimal skin-toned background clutter. Use the on-screen ROI guide box for best recognition accuracy.
