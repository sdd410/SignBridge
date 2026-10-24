"""Tests for SignCNN architecture and ModelLoader."""

import json
from pathlib import Path
import pytest
import torch
import numpy as np

from ml.models.architecture import SignCNN
from backend.app.inference.model_loader import ModelLoader
from backend.app.inference.predictor import Predictor


def test_sign_cnn_forward():
    model = SignCNN(num_classes=29, in_channels=3, img_size=64)
    model.eval()

    dummy_input = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = model.predict_proba(dummy_input)

    assert logits.shape == (2, 29)
    assert probs.shape == (2, 29)
    # Check probabilities sum to 1
    sums = probs.sum(dim=-1).numpy()
    np.testing.assert_allclose(sums, [1.0, 1.0], atol=1e-5)


def test_model_loader_singleton():
    loader1 = ModelLoader.get_instance()
    loader2 = ModelLoader.get_instance()
    assert loader1 is loader2
    assert loader1.is_ready()
    assert loader1.num_classes >= 24
    assert "A" in loader1.classes


def test_predictor_inference():
    loader = ModelLoader.get_instance()
    predictor = Predictor(loader)

    # Create dummy 64x64 BGR frame
    dummy_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    res = predictor.predict(dummy_frame)

    assert res["error"] is None
    assert res["label"] in loader.classes
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["top_k"]) == 5
    assert res["latency_ms"] >= 0.0
