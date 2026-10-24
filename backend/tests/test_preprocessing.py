"""Tests for Vision Preprocessing and Hand Detection."""

import base64
import cv2
import numpy as np
import pytest
import torch

from backend.app.vision.preprocessing import decode_image, crop_roi, preprocess_for_model
from backend.app.vision.hand_detection import HandDetector


def test_decode_raw_bytes():
    # Create test 100x100 BGR image
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy[:, :] = (120, 150, 180)
    _, buf = cv2.imencode(".jpg", dummy)
    raw_bytes = buf.tobytes()

    decoded = decode_image(raw_bytes)
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == (100, 100, 3)


def test_decode_base64_string():
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", dummy)
    b64_str = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")

    decoded = decode_image(b64_str)
    assert decoded.shape == (64, 64, 3)


def test_decode_invalid_data():
    with pytest.raises(Exception):
        decode_image(b"not_an_image_string")


def test_crop_roi_within_bounds():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cropped = crop_roi(img, (20, 30, 50, 50))
    assert cropped.shape == (50, 50, 3)


def test_crop_roi_clamping():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Bbox extending beyond boundaries
    cropped = crop_roi(img, (80, 80, 50, 50))
    assert cropped.shape[0] == 20
    assert cropped.shape[1] == 20


def test_preprocess_for_model():
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    tensor = preprocess_for_model(img, target_size=64)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 64, 64)
    assert tensor.dtype == torch.float32
    # Ensure no NaN or inf
    assert not torch.isnan(tensor).any()
    assert not torch.isinf(tensor).any()


def test_hand_detector_guide_roi():
    detector = HandDetector(mode="guide_roi")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cropped, bbox, norm_bbox = detector.get_roi(frame)

    assert len(bbox) == 4
    assert len(norm_bbox) == 4
    x, y, w, h = bbox
    assert w == h
    assert cropped.shape[0] == h
    assert cropped.shape[1] == w
    assert x >= 0 and y >= 0
    assert 0.0 <= norm_bbox[0] <= 1.0
