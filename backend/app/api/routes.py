"""REST API Routes for SignBridge."""

import json
import logging
import time
from typing import Dict, Any, List
from fastapi import APIRouter, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse

from backend.app.config.settings import settings
from backend.app.inference.model_loader import ModelLoader
from backend.app.inference.predictor import Predictor
from backend.app.inference.stabilization import TemporalStabilizer
from backend.app.schemas.models import (
    ModelStatusResponse,
    ModelMetricsResponse,
    SettingsResponse,
    SettingsUpdate,
    TranscriptAction,
    TranscriptResponse,
    PredictionResponse,
)
from backend.app.transcript.manager import transcript_manager
from backend.app.vision.hand_detection import HandDetector
from backend.app.vision.preprocessing import decode_image

logger = logging.getLogger(__name__)

router = APIRouter()

# Shared singletons
model_loader = ModelLoader.get_instance()
predictor = Predictor(model_loader)
stabilizer = TemporalStabilizer(
    window_size=settings.smoothing_window_size,
    confidence_threshold=settings.confidence_threshold,
    stability_duration_ms=settings.stability_duration_ms,
    repeat_lockout_ms=settings.repeat_lockout_ms
)
hand_detector = HandDetector(mode="guide_roi")

recent_predictions: List[Dict[str, Any]] = []
SERVER_START_TIME = time.time()


@router.get("/health")
def get_health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "uptime_sec": round(time.time() - SERVER_START_TIME, 1),
        "model_ready": model_loader.is_ready(),
    }


@router.get("/camera/config")
def get_camera_config():
    """Default camera parameters for frontend stream negotiation."""
    return {
        "target_fps": settings.target_fps,
        "default_width": 640,
        "default_height": 480,
        "model_input_size": settings.input_image_size,
        "supported_roi_modes": ["guide_roi", "skin_contour", "full_frame"],
        "active_roi_mode": hand_detector.mode,
    }


@router.get("/model/status", response_model=ModelStatusResponse)
def get_model_status():
    """Model load status, device, and class count."""
    return model_loader.get_status_dict()


@router.get("/model/classes")
def get_model_classes():
    """List all supported ASL gesture classes."""
    return {
        "count": model_loader.num_classes,
        "classes": model_loader.classes,
        "class_indices": model_loader.class_indices,
    }


@router.get("/model/metrics")
def get_model_metrics():
    """Evaluation metrics, accuracy, and confusion matrix."""
    if not settings.metrics_path.exists():
        raise HTTPException(status_code=404, detail="Model metrics file not found. Run evaluation first.")
    with open(settings.metrics_path, "r") as f:
        metrics_data = json.load(f)
    return metrics_data


@router.get("/settings", response_model=SettingsResponse)
def get_settings():
    """Retrieve active pipeline and stabilization settings."""
    return SettingsResponse(
        confidence_threshold=stabilizer.confidence_threshold,
        smoothing_window_size=stabilizer.window_size,
        stability_duration_ms=stabilizer.stability_duration_ms,
        repeat_lockout_ms=stabilizer.repeat_lockout_ms,
        auto_insert_transcript=settings.auto_insert_transcript,
        roi_mode=hand_detector.mode,
        target_fps=settings.target_fps,
        input_image_size=settings.input_image_size
    )


@router.patch("/settings", response_model=SettingsResponse)
def update_settings(update: SettingsUpdate):
    """Update runtime stabilization and detection parameters."""
    if update.confidence_threshold is not None:
        settings.confidence_threshold = update.confidence_threshold
    if update.smoothing_window_size is not None:
        settings.smoothing_window_size = update.smoothing_window_size
    if update.stability_duration_ms is not None:
        settings.stability_duration_ms = update.stability_duration_ms
    if update.repeat_lockout_ms is not None:
        settings.repeat_lockout_ms = update.repeat_lockout_ms
    if update.auto_insert_transcript is not None:
        settings.auto_insert_transcript = update.auto_insert_transcript
    if update.target_fps is not None:
        settings.target_fps = update.target_fps
    if update.roi_mode is not None:
        hand_detector.set_mode(update.roi_mode)

    stabilizer.update_settings(
        window_size=update.smoothing_window_size,
        confidence_threshold=update.confidence_threshold,
        stability_duration_ms=update.stability_duration_ms,
        repeat_lockout_ms=update.repeat_lockout_ms
    )

    return get_settings()


@router.post("/predict", response_model=PredictionResponse)
def predict_frame(payload: Dict[str, Any] = Body(...)):
    """Run prediction on a single base64-encoded frame."""
    image_b64 = payload.get("image")
    if not image_b64:
        raise HTTPException(status_code=400, detail="Missing 'image' parameter in request body.")

    try:
        frame = decode_image(image_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decoding failed: {e}")

    # 1. Extract Hand ROI
    client_roi = payload.get("roi")
    roi_image, bbox, norm_bbox = hand_detector.get_roi(frame, client_norm_roi=client_roi)

    # 2. Run CNN inference
    pred_res = predictor.predict(roi_image)
    if pred_res.get("error"):
        raise HTTPException(status_code=503, detail=pred_res["error"])

    # 3. Apply temporal stabilization
    stab_res = stabilizer.process(pred_res["label"], pred_res["confidence"])

    # 4. Handle transcript insertion
    if stab_res["is_accepted"] and settings.auto_insert_transcript and stab_res["accepted_char"]:
        transcript_manager.append_char(
            stab_res["accepted_char"],
            confidence=stab_res["smoothed_confidence"],
            source="auto_recognition"
        )

    t_state = transcript_manager.get_state()

    result = PredictionResponse(
        label=stab_res["stabilized_label"],
        confidence=stab_res["smoothed_confidence"],
        is_accepted=stab_res["is_accepted"],
        is_stable=stab_res["is_stable"],
        stability_progress=stab_res["stability_progress"],
        top_k=pred_res["top_k"],
        latency_ms=pred_res["latency_ms"],
        timestamp=pred_res["timestamp"],
        accepted_char=stab_res["accepted_char"],
        transcript=t_state["text"],
        roi_bbox=list(bbox),
        norm_roi_bbox=list(norm_bbox)
    )

    recent_predictions.append(result.model_dump())
    if len(recent_predictions) > 100:
        recent_predictions.pop(0)

    return result


@router.get("/predictions/recent")
def get_recent_predictions(limit: int = 20):
    """Return recently logged prediction results."""
    return recent_predictions[-limit:]


@router.get("/transcripts", response_model=TranscriptResponse)
def get_transcript():
    """Get active transcript text and word count."""
    return transcript_manager.get_state()


@router.get("/dataset/summary")
def get_dataset_summary():
    """Return counts of available training images per class."""
    custom_dir = settings.weights_dir.parent / "dataset" / "custom"
    real_raw_dir = settings.weights_dir.parent / "dataset" / "real_raw"
    split_train_dir = settings.weights_dir.parent / "dataset" / "split" / "train"

    summary: Dict[str, Dict[str, int]] = {}
    
    if split_train_dir.exists():
        for d in sorted(split_train_dir.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                count = len(list(d.glob("*.jpg")) + list(d.glob("*.png")))
                summary[d.name] = {"train_samples": count, "custom_samples": 0}

    if custom_dir.exists():
        for d in sorted(custom_dir.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                count = len(list(d.glob("*.jpg")) + list(d.glob("*.png")))
                if d.name in summary:
                    summary[d.name]["custom_samples"] = count
                else:
                    summary[d.name] = {"train_samples": 0, "custom_samples": count}

    total_samples = sum(item["train_samples"] + item["custom_samples"] for item in summary.values())
    return {
        "classes": sorted(list(summary.keys())),
        "total_classes": len(summary),
        "total_samples": total_samples,
        "class_breakdown": summary,
        "custom_dir": str(custom_dir),
    }


@router.post("/dataset/capture")
def capture_dataset_sample(payload: Dict[str, Any] = Body(...)):
    """Save a real webcam frame into the custom dataset directory for training."""
    label = payload.get("label", "").strip().upper()
    image_b64 = payload.get("image")
    if not label or not image_b64:
        raise HTTPException(status_code=400, detail="Missing 'label' or 'image' parameter.")

    try:
        frame = decode_image(image_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decoding failed: {e}")

    # Extract ROI or use full frame
    client_roi = payload.get("roi")
    roi_image, _, _ = hand_detector.get_roi(frame, client_norm_roi=client_roi)

    custom_dir = settings.weights_dir.parent / "dataset" / "custom" / label
    custom_dir.mkdir(parents=True, exist_ok=True)

    filename = f"hand_{int(time.time() * 1000)}.jpg"
    filepath = custom_dir / filename
    cv2.imwrite(str(filepath), roi_image)

    count = len(list(custom_dir.glob("*.jpg")))
    return {"status": "saved", "label": label, "filename": filename, "class_total": count}


@router.post("/dataset/train")
def trigger_training():
    """Trigger background retraining on real and custom hand datasets."""
    import subprocess
    cmd = [
        str(settings.weights_dir.parent.parent / ".venv" / "bin" / "python"),
        "ml/scripts/prepare_real_dataset.py",
        "--max-per-class", "350"
    ]
    try:
        subprocess.run(cmd, check=True)
        # Reload model into singleton
        model_loader.load()
        return {"status": "success", "message": "Model reloaded with updated real-hand weights"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {e}")

def get_transcript():
    """Get active transcript text and word count."""
    return transcript_manager.get_state()


@router.post("/transcripts", response_model=TranscriptResponse)
def mutate_transcript(action: TranscriptAction):
    """Perform action on transcript: 'append', 'space', 'backspace', 'clear', 'set'."""
    cmd = action.action.lower()
    if cmd == "append":
        if action.text:
            transcript_manager.append_char(action.text, source="manual")
    elif cmd == "space":
        transcript_manager.space()
    elif cmd == "backspace":
        transcript_manager.backspace()
    elif cmd == "clear":
        transcript_manager.clear()
    elif cmd == "set":
        transcript_manager.set_text(action.text or "")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported transcript action '{cmd}'.")

    return transcript_manager.get_state()
