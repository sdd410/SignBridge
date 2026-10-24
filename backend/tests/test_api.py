"""Tests for FastAPI endpoints."""

import base64
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "uptime_sec" in data
    assert data["model_ready"] is True


def test_model_status():
    res = client.get("/api/model/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "LOADED"
    assert data["num_classes"] >= 24
    assert len(data["classes"]) >= 24


def test_model_metrics():
    res = client.get("/api/model/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "top1_accuracy" in data
    assert "top3_accuracy" in data
    assert "confusion_matrix" in data


def test_settings_get_and_patch():
    # Get settings
    get_res = client.get("/api/settings")
    assert get_res.status_code == 200
    curr = get_res.json()

    # Patch settings
    patch_res = client.patch("/api/settings", json={
        "confidence_threshold": 0.72,
        "smoothing_window_size": 8
    })
    assert patch_res.status_code == 200
    updated = patch_res.json()
    assert updated["confidence_threshold"] == 0.72
    assert updated["smoothing_window_size"] == 8


def test_transcript_api():
    # Clear
    client.post("/api/transcripts", json={"action": "clear"})
    res = client.get("/api/transcripts")
    assert res.json()["text"] == ""

    # Append
    client.post("/api/transcripts", json={"action": "append", "text": "A"})
    client.post("/api/transcripts", json={"action": "append", "text": "S"})
    client.post("/api/transcripts", json={"action": "append", "text": "L"})
    assert client.get("/api/transcripts").json()["text"] == "ASL"

    # Space
    client.post("/api/transcripts", json={"action": "space"})
    assert client.get("/api/transcripts").json()["text"] == "ASL "

    # Backspace
    client.post("/api/transcripts", json={"action": "backspace"})
    assert client.get("/api/transcripts").json()["text"] == "ASL"


def test_predict_endpoint():
    # Generate dummy frame
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    b64_str = base64.b64encode(buf.tobytes()).decode("utf-8")

    res = client.post("/api/predict", json={"image": b64_str})
    assert res.status_code == 200
    data = res.json()
    assert "label" in data
    assert "confidence" in data
    assert "top_k" in data
    assert len(data["top_k"]) == 5
    assert "latency_ms" in data
    assert "roi_bbox" in data
