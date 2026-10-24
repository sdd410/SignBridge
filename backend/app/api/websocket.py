"""Real-Time WebSocket Inference Endpoint for SignBridge."""

import json
import logging
import time
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.app.config.settings import settings
from backend.app.inference.model_loader import ModelLoader
from backend.app.inference.predictor import Predictor
from backend.app.inference.stabilization import TemporalStabilizer
from backend.app.transcript.manager import transcript_manager
from backend.app.vision.hand_detection import HandDetector
from backend.app.vision.preprocessing import decode_image

logger = logging.getLogger(__name__)

ws_router = APIRouter()


@ws_router.websocket("/ws/predict")
async def websocket_predict_endpoint(websocket: WebSocket):
    """High-throughput WebSocket streaming endpoint for webcam frames."""
    await websocket.accept()
    logger.info("WebSocket client connected to /ws/predict")

    model_loader = ModelLoader.get_instance()
    predictor = Predictor(model_loader)
    stabilizer = TemporalStabilizer(
        window_size=settings.smoothing_window_size,
        confidence_threshold=settings.confidence_threshold,
        stability_duration_ms=settings.stability_duration_ms,
        repeat_lockout_ms=settings.repeat_lockout_ms
    )
    hand_detector = HandDetector(mode="guide_roi")

    last_frame_time = time.time()
    fps_counter = 0.0

    try:
        while True:
            # Receive either binary frame or JSON message
            message = await websocket.receive()
            recv_time = time.time()

            # Handle disconnection frames
            if message.get("type") == "websocket.disconnect":
                logger.info("WebSocket client initiated disconnect")
                break

            frame = None
            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                if len(raw_bytes) > settings.max_frame_bytes:
                    await websocket.send_json({"error": "Frame exceeded maximum allowed size."})
                    continue
                try:
                    frame = decode_image(raw_bytes)
                except Exception as e:
                    await websocket.send_json({"error": f"Image decode failure: {str(e)}"})
                    continue

            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                except Exception:
                    await websocket.send_json({"error": "Malformed JSON payload."})
                    continue

                # Handle ping/config events
                msg_type = data.get("type")
                if msg_type == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": time.time()})
                    continue

                if msg_type == "config":
                    if "roi_mode" in data:
                        hand_detector.set_mode(data["roi_mode"])
                    if "confidence_threshold" in data:
                        stabilizer.update_settings(confidence_threshold=float(data["confidence_threshold"]))
                    if "smoothing_window_size" in data:
                        stabilizer.update_settings(window_size=int(data["smoothing_window_size"]))
                    if "stability_duration_ms" in data:
                        stabilizer.update_settings(stability_duration_ms=int(data["stability_duration_ms"]))
                    if "repeat_lockout_ms" in data:
                        stabilizer.update_settings(repeat_lockout_ms=int(data["repeat_lockout_ms"]))
                    await websocket.send_json({"type": "config_ack", "status": "updated"})
                    continue

                image_data = data.get("image")
                if not image_data:
                    await websocket.send_json({"error": "Missing 'image' in payload."})
                    continue

                try:
                    frame = decode_image(image_data)
                except Exception as e:
                    await websocket.send_json({"error": f"Image decode failure: {str(e)}"})
                    continue

            if frame is None:
                continue

            # Calculate incoming FPS
            now = time.time()
            dt = now - last_frame_time
            last_frame_time = now
            if dt > 0:
                current_fps = 1.0 / dt
                fps_counter = 0.85 * fps_counter + 0.15 * current_fps  # Exponential smoothing

            # 1. Extract Hand ROI
            roi_image, bbox = hand_detector.get_roi(frame)

            # 2. Run CNN Inference
            pred_res = predictor.predict(roi_image)

            # 3. Apply Temporal Stabilization
            stab_res = stabilizer.process(pred_res["label"], pred_res["confidence"])

            # 4. Handle Transcript Insertion
            accepted_char = stab_res["accepted_char"]
            if stab_res["is_accepted"] and settings.auto_insert_transcript and accepted_char:
                transcript_manager.append_char(
                    accepted_char,
                    confidence=stab_res["smoothed_confidence"],
                    source="realtime_recognition"
                )

            current_transcript = transcript_manager.get_state()

            # 5. Send Real-Time Prediction Event
            response = {
                "type": "prediction",
                "label": stab_res["stabilized_label"],
                "raw_label": pred_res["label"],
                "confidence": stab_res["smoothed_confidence"],
                "raw_confidence": pred_res["confidence"],
                "is_accepted": stab_res["is_accepted"],
                "is_stable": stab_res["is_stable"],
                "stability_progress": stab_res["stability_progress"],
                "top_k": pred_res["top_k"],
                "latency_ms": pred_res["latency_ms"],
                "fps": round(fps_counter, 1),
                "timestamp": now,
                "accepted_char": accepted_char,
                "transcript": current_transcript["text"],
                "roi_bbox": list(bbox),
            }

            await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket unhandled error: {e}", exc_info=True)
    finally:
        logger.info("Cleaned up WebSocket connection")
