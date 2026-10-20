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


class RegionShare(BaseModel):
    region: Literal["eyes", "nose", "mouth", "rest_of_face", "outside_face"]
    share: float = Field(..., ge=0, le=1, description="Fraction of the Grad-CAM heatmap inside this region.")


class QualityReport(BaseModel):
    face_px: int = Field(..., description="Size of the detected face in the original image, in pixels.")
    sharpness: float = Field(..., description="Variance of the Laplacian on the face crop. Low = blurry.")
    detector_confidence: float
    faces_in_image: int
    notes: list[str] = Field(..., description="Plain-language reasons the result may be less reliable.")


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
    attention: list[RegionShare] = Field(..., description="Where the heatmap's weight falls on the face, largest first.")
    quality: QualityReport
    model: ModelInfo
    latency_ms: float
