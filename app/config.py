"""Settings, read from environment variables with sensible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
    return [o.strip() for o in raw.split(",") if o.strip()]


@dataclass(frozen=True)
class Settings:
    app_name: str = "Deepfake Detector API"
    app_version: str = os.getenv("APP_VERSION", "0.2.0")
    model_checkpoint: Path = Path(os.getenv("MODEL_CHECKPOINT", BASE_DIR / "model" / "checkpoints" / "best.pth"))
    # Optional: where to download the checkpoint from if it isn't on disk
    # (e.g. a GitHub Release asset). Lets cloud deploys skip baking it in.
    model_url: str = os.getenv("MODEL_URL", "")
    face_detector_path: Path = Path(
        os.getenv("FACE_DETECTOR_PATH", BASE_DIR / "model" / "weights" / "face_detection_yunet_2023mar.onnx")
    )
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
    # Calibrated P(fake) at or above this -> "likely_fake"; at or below 1 - this -> "likely_real".
    decision_margin: float = float(os.getenv("DECISION_MARGIN", "0.7"))
    allowed_origins: list[str] = field(default_factory=_origins)


settings = Settings()
