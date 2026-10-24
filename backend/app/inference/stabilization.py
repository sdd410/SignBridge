"""Temporal Prediction Stabilization Module for SignBridge.

Buffers recent predictions over a sliding temporal window to eliminate flickering,
enforces confidence thresholds, tracks stability duration, and manages repeat lockout
for steady transcript assembly.
"""

import collections
import logging
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TemporalStabilizer:
    """Stabilizes real-time predictions across consecutive frames."""

    def __init__(
        self,
        window_size: int = 6,
        confidence_threshold: float = 0.65,
        stability_duration_ms: int = 400,
        repeat_lockout_ms: int = 1200,
    ):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.stability_duration_ms = stability_duration_ms
        self.repeat_lockout_ms = repeat_lockout_ms

        # Buffer of (label, confidence, timestamp)
        self.history: collections.deque = collections.deque(maxlen=window_size)

        # State tracking for stability duration
        self.active_candidate: Optional[str] = None
        self.candidate_first_seen: float = 0.0

        # State tracking for transcript emission
        self.last_accepted_label: Optional[str] = None
        self.last_accepted_time: float = 0.0

    def update_settings(
        self,
        window_size: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        stability_duration_ms: Optional[int] = None,
        repeat_lockout_ms: Optional[int] = None,
    ):
        """Dynamically update stabilization parameters."""
        if window_size is not None and window_size != self.window_size:
            self.window_size = window_size
            self.history = collections.deque(self.history, maxlen=window_size)
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        if stability_duration_ms is not None:
            self.stability_duration_ms = stability_duration_ms
        if repeat_lockout_ms is not None:
            self.repeat_lockout_ms = repeat_lockout_ms

    def reset(self):
        """Clear stabilization history and state."""
        self.history.clear()
        self.active_candidate = None
        self.candidate_first_seen = 0.0
        self.last_accepted_label = None
        self.last_accepted_time = 0.0

    def process(self, raw_label: str, raw_confidence: float) -> Dict:
        """Process a raw prediction and return stabilized state."""
        now = time.time()
        self.history.append((raw_label, raw_confidence, now))

        # 1. Compute weighted majority vote over sliding window
        label_scores: Dict[str, float] = {}
        for lbl, conf, _ in self.history:
            label_scores[lbl] = label_scores.get(lbl, 0.0) + conf

        # Normalize smoothed confidence
        total_weight = sum(label_scores.values()) or 1.0
        stabilized_label, max_score = max(label_scores.items(), key=lambda item: item[1])
        smoothed_confidence = round(max_score / len(self.history), 4)

        # 2. Check if candidate passes confidence threshold
        passes_confidence = smoothed_confidence >= self.confidence_threshold and stabilized_label != "nothing"

        # 3. Track stability duration
        if passes_confidence:
            if self.active_candidate != stabilized_label:
                self.active_candidate = stabilized_label
                self.candidate_first_seen = now
        else:
            self.active_candidate = None
            self.candidate_first_seen = 0.0

        elapsed_candidate_ms = (now - self.candidate_first_seen) * 1000.0 if self.candidate_first_seen > 0 else 0.0
        stability_progress = min(1.0, elapsed_candidate_ms / max(1.0, float(self.stability_duration_ms))) if passes_confidence else 0.0
        is_stable = stability_progress >= 1.0

        # 4. Check acceptance and repeat lockout
        is_accepted = False
        accepted_char = None

        if is_stable and self.active_candidate:
            time_since_last_accepted = (now - self.last_accepted_time) * 1000.0
            
            # Acceptance condition:
            # - First time seeing this candidate, OR
            # - Different candidate than last accepted, OR
            # - Same candidate but repeat lockout cooldown has elapsed
            can_accept = (
                self.active_candidate != self.last_accepted_label or
                time_since_last_accepted >= self.repeat_lockout_ms
            )

            if can_accept:
                is_accepted = True
                self.last_accepted_label = self.active_candidate
                self.last_accepted_time = now

                # Map label to transcript character
                if self.active_candidate == "space":
                    accepted_char = " "
                elif self.active_candidate == "del":
                    accepted_char = "\b"  # Backspace control
                else:
                    accepted_char = self.active_candidate

        return {
            "stabilized_label": stabilized_label if passes_confidence else (raw_label if raw_label != "nothing" else "No Gesture"),
            "smoothed_confidence": smoothed_confidence,
            "is_stable": is_stable,
            "stability_progress": round(stability_progress, 3),
            "is_accepted": is_accepted,
            "accepted_char": accepted_char,
            "passes_confidence": passes_confidence
        }
