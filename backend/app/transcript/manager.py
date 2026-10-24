"""Transcript Manager for SignBridge.

Maintains live text assembly, history log, editing operations, and exports.
"""

import logging
import threading
import time
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class TranscriptManager:
    """Manages transcript text buffer and mutation events."""

    def __init__(self):
        self._lock = threading.RLock()
        self.text: str = ""
        self.history: List[Dict[str, Any]] = []
        self.last_updated: float = time.time()

    def append_char(self, char: str, confidence: float = 1.0, source: str = "model"):
        """Append a recognized or manual character to the transcript."""
        with self._lock:
            if char == "\b":
                self.backspace()
                return

            self.text += char
            self.last_updated = time.time()
            self.history.append({
                "char": char,
                "confidence": confidence,
                "timestamp": self.last_updated,
                "source": source,
                "current_length": len(self.text)
            })

            # Keep history buffer bounded
            if len(self.history) > 200:
                self.history = self.history[-100:]

    def space(self):
        """Append a space character."""
        with self._lock:
            if not self.text.endswith(" ") and len(self.text) > 0:
                self.text += " "
                self.last_updated = time.time()
                self.history.append({
                    "char": " ",
                    "confidence": 1.0,
                    "timestamp": self.last_updated,
                    "source": "manual_or_sign",
                    "current_length": len(self.text)
                })

    def backspace(self):
        """Remove last character."""
        with self._lock:
            if len(self.text) > 0:
                removed = self.text[-1]
                self.text = self.text[:-1]
                self.last_updated = time.time()
                self.history.append({
                    "char": "[DEL]",
                    "removed": removed,
                    "timestamp": self.last_updated,
                    "source": "backspace",
                    "current_length": len(self.text)
                })

    def clear(self):
        """Clear the entire transcript."""
        with self._lock:
            self.text = ""
            self.history.clear()
            self.last_updated = time.time()

    def set_text(self, new_text: str):
        """Explicitly set transcript text from manual user editing."""
        with self._lock:
            self.text = new_text
            self.last_updated = time.time()

    def get_state(self) -> Dict[str, Any]:
        """Return snapshot of transcript state."""
        with self._lock:
            words = [w for w in self.text.strip().split() if w]
            return {
                "text": self.text,
                "character_count": len(self.text),
                "word_count": len(words),
                "last_updated": self.last_updated,
                "history": list(self.history[-30:])  # Return recent 30 items
            }


# Global singleton instance
transcript_manager = TranscriptManager()
