"""CNN Predictor for SignBridge.

Performs forward inference, probability extraction, and top-k ranking.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from backend.app.config.settings import settings
from backend.app.inference.model_loader import ModelLoader
from backend.app.schemas.models import ClassPrediction
from backend.app.vision.preprocessing import preprocess_for_model

logger = logging.getLogger(__name__)


class Predictor:
    """Performs real-time frame inference on the loaded SignCNN model."""

    def __init__(self, model_loader: Optional[ModelLoader] = None):
        self.loader = model_loader or ModelLoader.get_instance()

    def predict(self, roi_image: np.ndarray, top_k: int = 5) -> Dict:
        """Run inference on cropped hand ROI image.
        
        Returns dictionary with top prediction, top-k list, and latency.
        """
        if not self.loader.is_ready():
            return {
                "label": "MODEL_NOT_READY",
                "confidence": 0.0,
                "top_k": [],
                "latency_ms": 0.0,
                "timestamp": time.time(),
                "error": self.loader.error_message or "Model is not loaded."
            }

        start_time = time.perf_counter()

        # 1. Preprocess image into tensor
        tensor = preprocess_for_model(
            roi_image,
            target_size=self.loader.img_size
        ).to(self.loader.device)

        # 2. Forward pass with Softmax
        with torch.no_grad():
            logits = self.loader.model(tensor)
            probs = F.softmax(logits, dim=-1)[0]

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # 3. Top-k predictions
        k = min(top_k, self.loader.num_classes)
        top_probs, top_indices = torch.topk(probs, k=k)

        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        predictions: List[ClassPrediction] = []
        for prob, idx in zip(top_probs, top_indices):
            label = self.loader.class_indices.get(int(idx), f"Class_{idx}")
            predictions.append(
                ClassPrediction(
                    label=label,
                    confidence=float(round(float(prob), 4)),
                    index=int(idx)
                )
            )

        best_prediction = predictions[0]

        return {
            "label": best_prediction.label,
            "confidence": best_prediction.confidence,
            "top_k": [p.model_dump() for p in predictions],
            "latency_ms": round(latency_ms, 2),
            "timestamp": time.time(),
            "error": None
        }
