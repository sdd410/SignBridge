"""Vision Preprocessing Module for SignBridge.

Handles frame decoding, image resizing, color conversion, ROI cropping,
and PyTorch tensor normalization.
"""

import base64
import logging
from typing import Optional, Tuple, Union
import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Standard ImageNet normalization matching training
DEFAULT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DEFAULT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def decode_image(data: Union[bytes, str]) -> np.ndarray:
    """Decode raw image bytes or base64 string into OpenCV BGR numpy array."""
    if isinstance(data, str):
        if "," in data:
            # Strip data:image/jpeg;base64,... header if present
            data = data.split(",", 1)[1]
        try:
            raw_bytes = base64.b64decode(data)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 string: {e}")
    elif isinstance(data, bytes):
        raw_bytes = data
    else:
        raise TypeError(f"Expected bytes or str, got {type(data)}")

    np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None or image.size == 0:
        raise ValueError("Decoded image is empty or invalid format")

    return image


def crop_roi(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop image to bounding box [x, y, w, h] with boundary clamping."""
    h, w = image.shape[:2]
    x, y, bw, bh = bbox

    # Clamp coordinates to image boundaries
    x1 = max(0, min(w - 1, x))
    y1 = max(0, min(h - 1, y))
    x2 = max(x1 + 1, min(w, x + bw))
    y2 = max(y1 + 1, min(h, y + bh))

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        logger.warning(f"Invalid crop size for bbox {bbox}, returning original image")
        return image
    return cropped


def preprocess_for_model(
    image: np.ndarray,
    target_size: int = 64,
    mean: np.ndarray = DEFAULT_MEAN,
    std: np.ndarray = DEFAULT_STD
) -> torch.Tensor:
    """Preprocess OpenCV BGR image into normalized PyTorch tensor (1, 3, target_size, target_size)."""
    # 1. Resize to target dimension
    if image.shape[0] != target_size or image.shape[1] != target_size:
        resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    else:
        resized = image

    # 2. Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # 3. Normalize to [0.0, 1.0]
    normalized = rgb.astype(np.float32) / 255.0

    # 4. Standardize with mean and std
    standardized = (normalized - mean) / std

    # 5. Convert HWC -> CHW and add batch dimension -> (1, 3, H, W)
    chw = np.transpose(standardized, (2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0).float()

    return tensor
