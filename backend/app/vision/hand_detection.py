"""Hand Region-of-Interest (ROI) Detection Module for SignBridge.

Supports configurable detection strategies:
- Guide ROI: Fixed bounding box matching on-screen UI guide (default, highly reliable)
- Skin Contour: Color-based segmentation detecting hand contours
- Full Frame: Passthrough for pre-cropped or isolated inputs
"""

import logging
from typing import Tuple, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class HandDetector:
    """Configurable Hand ROI Extractor."""

    def __init__(self, mode: str = "guide_roi"):
        self.mode = mode

    def set_mode(self, mode: str):
        if mode in ["guide_roi", "skin_contour", "full_frame"]:
            self.mode = mode
        else:
            logger.warning(f"Unknown mode '{mode}', defaulting to 'guide_roi'")
            self.mode = "guide_roi"

    def get_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Extract hand region from video frame based on configured strategy.
        
        Returns:
            cropped_roi: Cropped BGR image
            bbox: Tuple of (x, y, w, h) in frame coordinates
        """
        h, w = frame.shape[:2]

        if self.mode == "full_frame":
            return frame, (0, 0, w, h)

        if self.mode == "skin_contour":
            bbox = self._detect_skin_contour(frame)
            if bbox is not None:
                x, y, bw, bh = bbox
                cropped = frame[y:y+bh, x:x+bw]
                return cropped, bbox
            # Fallback to guide ROI if no skin contour found
            return self._get_guide_roi(frame)

        # Default: guide_roi
        return self._get_guide_roi(frame)

    def _get_guide_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Compute centered/right-biased guide ROI box."""
        h, w = frame.shape[:2]
        # Standard sign language capture box: square 56% of min dimension
        box_size = int(min(h, w) * 0.58)
        
        # Center horizontally and slightly upper-middle vertically
        x = (w - box_size) // 2
        y = max(10, int(h * 0.18))

        bbox = (x, y, box_size, box_size)
        cropped = frame[y:y+box_size, x:x+box_size]
        return cropped, bbox

    def _detect_skin_contour(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect hand candidate bounding box using YCrCb skin segmentation."""
        h, w = frame.shape[:2]
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        # Standard human skin color range in YCrCb color space
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)

        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        # Morphological noise filtering
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Find largest contour by area
        valid_contours = [c for c in contours if cv2.contourArea(c) > (w * h * 0.02)]
        if not valid_contours:
            return None

        largest = max(valid_contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)

        # Expand slightly and make square
        side = max(cw, ch)
        pad = int(side * 0.2)
        side += pad * 2

        cx, cy = x + cw // 2, y + ch // 2
        nx = max(0, cx - side // 2)
        ny = max(0, cy - side // 2)
        side = min(side, w - nx, h - ny)

        if side < 40:
            return None

        return (nx, ny, side, side)
