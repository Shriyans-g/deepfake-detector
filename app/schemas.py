"""API response models."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfo(BaseModel):
    arch: str
    temperature: float
    epoch: Optional[int] = None
    val_metrics: Optional[dict] = None


class FaceBoxOut(BaseModel):
    x: int
    y: int
    w: int
    h: int
    score: float


class PredictionResponse(BaseModel):
    verdict: Literal["likely_fake", "uncertain", "likely_real"] = Field(
        ..., description="Three-way call. 'uncertain' when the calibrated score is near 0.5."
    )
    prob_fake: float = Field(..., ge=0, le=1, description="Calibrated probability the face is manipulated.")
    raw_prob_fake: float = Field(..., ge=0, le=1, description="Uncalibrated model output, for comparison.")
    face_box: FaceBoxOut = Field(..., description="Where the analyzed face is in the original image (pixels).")
    face_crop: str = Field(..., description="The face crop the model saw, as a PNG data URL.")
    heatmap: str = Field(..., description="Grad-CAM overlay on the face crop, as a PNG data URL.")
    heatmap_explains: Literal["fake", "real"] = Field(..., description="Which answer the heatmap shows evidence for.")
    model: ModelInfo
    latency_ms: float
