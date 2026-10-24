"""Application configuration settings for SignBridge."""

from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    """Runtime application settings."""
    app_name: str = "SignBridge"
    app_version: str = "1.0.0"
    debug: bool = False

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "*"
    ]

    # Model & Weights paths
    weights_dir: Path = PROJECT_ROOT / "ml" / "weights"
    model_checkpoint_path: Path = PROJECT_ROOT / "ml" / "weights" / "sign_model.pt"
    class_indices_path: Path = PROJECT_ROOT / "ml" / "weights" / "class_indices.json"
    metrics_path: Path = PROJECT_ROOT / "ml" / "weights" / "metrics.json"
    model_config_path: Path = PROJECT_ROOT / "ml" / "weights" / "model_config.json"

    # Vision & Preprocessing
    input_image_size: int = 64
    in_channels: int = 3
    target_fps: int = 15
    max_frame_bytes: int = 4 * 1024 * 1024  # 4 MB max payload

    # Prediction Stabilization defaults
    confidence_threshold: float = 0.65
    smoothing_window_size: int = 6
    stability_duration_ms: int = 400
    repeat_lockout_ms: int = 1200
    auto_insert_transcript: bool = True

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }


settings = Settings()
