"""Pydantic schemas and data models for SignBridge."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ClassPrediction(BaseModel):
    """Single class prediction score."""
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    index: int


class PredictionResponse(BaseModel):
    """Inference result for a single frame."""
    label: str
    confidence: float
    is_accepted: bool = False
    is_stable: bool = False
    stability_progress: float = 0.0  # 0.0 to 1.0 progress toward required stability duration
    top_k: List[ClassPrediction] = []
    latency_ms: float
    timestamp: float
    accepted_char: Optional[str] = None
    transcript: str = ""
    roi_bbox: Optional[List[int]] = None  # [x, y, w, h] pixel coords
    norm_roi_bbox: Optional[List[float]] = None  # [nx, ny, nw, nh] normalized coords [0.0 - 1.0]


class ModelStatusResponse(BaseModel):
    """Model availability and metadata status."""
    status: str  # "LOADED", "NOT_FOUND", "ERROR"
    model_type: Optional[str] = None
    num_classes: int = 0
    classes: List[str] = []
    parameter_count: int = 0
    device: str = "cpu"
    checkpoint_path: str = ""
    val_accuracy: Optional[float] = None
    created_at: Optional[str] = None
    message: Optional[str] = None


class ModelMetricsResponse(BaseModel):
    """Evaluation metrics of the trained model."""
    top1_accuracy: float
    top3_accuracy: float
    macro_avg_f1: float
    weighted_avg_f1: float
    total_test_samples: int
    class_count: int
    classes: List[str]
    per_class_metrics: Dict[str, Dict[str, Any]]
    confusion_matrix: List[List[int]]


class SettingsUpdate(BaseModel):
    """User-updatable settings schema."""
    confidence_threshold: Optional[float] = Field(None, ge=0.1, le=1.0)
    smoothing_window_size: Optional[int] = Field(None, ge=1, le=30)
    stability_duration_ms: Optional[int] = Field(None, ge=100, le=3000)
    repeat_lockout_ms: Optional[int] = Field(None, ge=200, le=5000)
    auto_insert_transcript: Optional[bool] = None
    roi_mode: Optional[str] = None  # "guide_roi", "skin_contour", "full_frame"
    target_fps: Optional[int] = Field(None, ge=1, le=60)


class SettingsResponse(BaseModel):
    """Current application configuration."""
    confidence_threshold: float
    smoothing_window_size: int
    stability_duration_ms: int
    repeat_lockout_ms: int
    auto_insert_transcript: bool
    roi_mode: str
    target_fps: int
    input_image_size: int


class TranscriptAction(BaseModel):
    """Transcript modification command."""
    action: str = Field(..., description="'append', 'space', 'backspace', 'clear', 'set'")
    text: Optional[str] = None


class TranscriptResponse(BaseModel):
    """Transcript state."""
    text: str
    character_count: int
    word_count: int
    last_updated: float
    history: List[Dict[str, Any]] = []
