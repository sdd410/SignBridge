"""Hand Region-of-Interest (ROI) Detection Module for SignBridge.

Supports configurable detection strategies:
- Guide ROI: Fixed bounding box matching on-screen UI guide (default, highly reliable)
- Skin Contour: Color-based segmentation detecting hand contours
- Full Frame: Passthrough for pre-cropped or isolated inputs
"""

import logging
from typing import Tuple, Optional, List
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

    def get_roi(
        self, frame: np.ndarray, client_norm_roi: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[float, float, float, float]]:
        """Extract hand region from video frame based on configured strategy.
        
        Returns:
            cropped_roi: Cropped BGR image
            bbox: Tuple of (x, y, w, h) in pixel coordinates
            norm_bbox: Tuple of (nx, ny, nw, nh) in [0.0, 1.0] normalized coordinates
        """
        h, w = frame.shape[:2]

        # If client provided explicit normalized ROI box, use it directly (100% WYSIWYG)
        if client_norm_roi and len(client_norm_roi) == 4:
            nx, ny, nw, nh = client_norm_roi
            x = max(0, min(w - 10, int(nx * w)))
            y = max(0, min(h - 10, int(ny * h)))
            bw = max(20, min(w - x, int(nw * w)))
            bh = max(20, min(h - y, int(nh * h)))
            cropped = frame[y:y+bh, x:x+bw]
            return cropped, (x, y, bw, bh), (round(x / w, 4), round(y / h, 4), round(bw / w, 4), round(bh / h, 4))

        if self.mode == "full_frame":
            return frame, (0, 0, w, h), (0.0, 0.0, 1.0, 1.0)

        if self.mode == "skin_contour":
            bbox = self._detect_skin_contour(frame)
            if bbox is not None:
                x, y, bw, bh = bbox
                cropped = frame[y:y+bh, x:x+bw]
                norm_bbox = (round(x / w, 4), round(y / h, 4), round(bw / w, 4), round(bh / h, 4))
                return cropped, bbox, norm_bbox
            # Fallback to guide ROI
            return self._get_guide_roi(frame)

        # Default: guide_roi
        return self._get_guide_roi(frame)

    def _get_guide_roi(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[float, float, float, float]]:
        """Compute centered guide ROI box."""
        h, w = frame.shape[:2]
        # Standard square capture box: 60% of frame height (or min dimension)
        box_size = int(min(h, w) * 0.60)
        
        # Center horizontally and upper-middle vertically
        x = (w - box_size) // 2
        y = max(10, int(h * 0.16))

        bbox = (x, y, box_size, box_size)
        norm_bbox = (round(x / w, 4), round(y / h, 4), round(box_size / w, 4), round(box_size / h, 4))
        cropped = frame[y:y+box_size, x:x+box_size]
        return cropped, bbox, norm_bbox

    def _detect_skin_contour(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect hand candidate bounding box using YCrCb skin segmentation."""
        h, w = frame.shape[:2]
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        # Standard human skin color range in YCrCb color space
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)

        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        valid_contours = [c for c in contours if cv2.contourArea(c) > (w * h * 0.02)]
        if not valid_contours:
            return None

        largest = max(valid_contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)

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
