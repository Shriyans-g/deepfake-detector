"""FastAPI app: /health, /model-info, /predict."""

from __future__ import annotations

from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import settings
from app.inference import DetectorService, ModelNotReady, NoFaceFound
from app.schemas import HealthResponse, ModelInfo, PredictionResponse

app = FastAPI(title=settings.app_name, version=settings.app_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

service = DetectorService(
    settings.model_checkpoint,
    settings.face_detector_path,
    settings.decision_margin,
    settings.model_url,
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


def read_image(file: UploadFile, payload: bytes) -> Image.Image:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Upload a JPEG, PNG, or WebP image.")
    if not payload:
        raise HTTPException(400, "Uploaded file is empty.")
    if len(payload) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds the {settings.max_upload_size_mb} MB limit.")
    try:
        image = Image.open(BytesIO(payload))
        image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(400, "Could not read the uploaded image.") from exc
    # Phone photos store rotation in EXIF; apply it so faces are upright.
    return ImageOps.exif_transpose(image).convert("RGB")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=service.ready)


@app.get("/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    service.try_load()
    if not service.ready:
        raise HTTPException(503, "Model not loaded.")
    return ModelInfo(**service.model_info())


# A plain `def` (not async) so FastAPI runs the CPU-heavy model call in its
# threadpool instead of blocking the event loop.
@app.post("/predict", response_model=PredictionResponse)
def predict(file: UploadFile = File(...)) -> PredictionResponse:
    image = read_image(file, file.file.read())
    try:
        result = service.predict(image)
    except ModelNotReady as exc:
        raise HTTPException(503, str(exc)) from exc
    except NoFaceFound as exc:
        raise HTTPException(422, str(exc)) from exc
    return PredictionResponse(**result)
