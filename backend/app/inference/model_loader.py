"""Thread-Safe Model Loader for SignBridge.

Loads and caches the trained PyTorch SignCNN model and associated class mappings.
Provides structured readiness and metadata inspection.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch

# Ensure project root is available in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config.settings import settings
from ml.models.architecture import SignCNN

logger = logging.getLogger(__name__)


class ModelLoader:
    """Singleton model management class for SignBridge."""

    _instance: Optional["ModelLoader"] = None

    def __init__(self):
        self.model: Optional[SignCNN] = None
        self.device: torch.device = self._select_device()
        self.class_indices: Dict[int, str] = {}
        self.classes: List[str] = []
        self.num_classes: int = 0
        self.img_size: int = settings.input_image_size
        self.status: str = "INITIALIZING"  # "LOADED", "NOT_FOUND", "ERROR"
        self.error_message: Optional[str] = None
        self.val_accuracy: Optional[float] = None
        self.parameter_count: int = 0
        self.checkpoint_metadata: Dict = {}

    @classmethod
    def get_instance(cls) -> "ModelLoader":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.load()
        return cls._instance

    def _select_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def load(self, checkpoint_path: Optional[Path] = None, classes_path: Optional[Path] = None):
        """Load checkpoint and class mappings from disk."""
        ckpt_path = checkpoint_path or settings.model_checkpoint_path
        cls_path = classes_path or settings.class_indices_path

        logger.info(f"Attempting to load model checkpoint from: {ckpt_path}")

        if not ckpt_path.exists():
            self.status = "NOT_FOUND"
            self.error_message = f"Model checkpoint not found at {ckpt_path}. Run ml/scripts/train.py to generate it."
            logger.warning(self.error_message)
            return

        try:
            # Load class mapping
            if cls_path.exists():
                with open(cls_path, "r") as f:
                    raw_classes = json.load(f)
                    # Convert keys to int
                    self.class_indices = {int(k): v for k, v in raw_classes.items()}
                    self.classes = [self.class_indices[i] for i in sorted(self.class_indices.keys())]
                    self.num_classes = len(self.classes)
            else:
                logger.warning(f"Class indices file not found at {cls_path}, will attempt to read from checkpoint")

            # Load model checkpoint
            checkpoint = torch.load(str(ckpt_path), map_location=self.device)
            self.checkpoint_metadata = checkpoint

            num_classes = checkpoint.get("num_classes", self.num_classes)
            if not self.classes and "class_names" in checkpoint:
                self.classes = checkpoint["class_names"]
                self.class_indices = {i: name for i, name in enumerate(self.classes)}
                self.num_classes = len(self.classes)

            self.img_size = checkpoint.get("img_size", settings.input_image_size)
            self.val_accuracy = checkpoint.get("val_acc")

            # Instantiate and load weights
            self.model = SignCNN(
                num_classes=self.num_classes,
                in_channels=checkpoint.get("in_channels", 3),
                img_size=self.img_size
            ).to(self.device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            self.parameter_count = sum(p.numel() for p in self.model.parameters())
            self.status = "LOADED"
            self.error_message = None

            logger.info(
                f"Successfully loaded {checkpoint.get('model_type', 'SignCNN')} on {self.device} "
                f"({self.num_classes} classes, {self.parameter_count:,} params, "
                f"Val Acc: {self.val_accuracy * 100:.2f}% if self.val_accuracy else 'N/A')"
            )

        except Exception as e:
            self.status = "ERROR"
            self.error_message = f"Failed to load model: {str(e)}"
            logger.error(self.error_message, exc_info=True)

    def is_ready(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self.status == "LOADED" and self.model is not None

    def get_status_dict(self) -> Dict:
        """Return status dictionary for API inspection."""
        return {
            "status": self.status,
            "model_type": "SignCNN" if self.is_ready() else None,
            "num_classes": self.num_classes,
            "classes": self.classes,
            "parameter_count": self.parameter_count,
            "device": str(self.device),
            "checkpoint_path": str(settings.model_checkpoint_path),
            "val_accuracy": round(self.val_accuracy, 4) if self.val_accuracy else None,
            "message": self.error_message,
        }
