from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.inference import DetectorService
from app.config import settings
from model.faces import FaceBox

client = TestClient(main.app)


def png_bytes(color=(120, 120, 120), size=(320, 320)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def noisy_png() -> bytes:
    rng = np.random.default_rng(0)
    buffer = BytesIO()
    Image.fromarray(rng.integers(0, 255, (320, 320, 3), dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_reports_model_loaded():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info():
    body = client.get("/model-info").json()
    assert body["arch"] == "resnet18"
    assert body["temperature"] == 1.5


def test_rejects_non_image_type():
    response = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_rejects_empty_file():
    response = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code == 400


def test_rejects_corrupt_image():
    response = client.post("/predict", files={"file": ("bad.png", b"not really a png", "image/png")})
    assert response.status_code == 400


def test_no_face_returns_422():
    response = client.post("/predict", files={"file": ("blank.png", png_bytes(), "image/png")})
    assert response.status_code == 422
    assert "No face" in response.json()["detail"]


def test_predict_with_face(monkeypatch):
    # Pretend the detector found a face so the rest of the pipeline runs.
    monkeypatch.setattr(
        main.service.face_detector, "detect_largest", lambda _img: FaceBox(80, 80, 160, 160, 0.99)
    )
    response = client.post("/predict", files={"file": ("face.png", noisy_png(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["prob_fake"] <= 1
    assert body["verdict"] in {"likely_fake", "uncertain", "likely_real"}
    assert body["heatmap"].startswith("data:image/png;base64,")
    assert body["face_crop"].startswith("data:image/png;base64,")
    assert body["face_box"]["w"] == 160
    assert abs(sum(r["share"] for r in body["attention"]) - 1) < 1e-3
    assert {"face_px", "sharpness", "notes"} <= set(body["quality"])
    # Temperature 1.5 pulls the calibrated score toward 0.5 relative to raw.
    assert abs(body["prob_fake"] - 0.5) <= abs(body["raw_prob_fake"] - 0.5) + 1e-4


def test_missing_checkpoint_is_503(tmp_path):
    service = DetectorService(tmp_path / "missing.pth", settings.face_detector_path)
    assert not service.ready
    with pytest.raises(Exception, match="No trained checkpoint"):
        service.predict(Image.new("RGB", (64, 64)))


@pytest.mark.parametrize("prob,expected", [(0.9, "likely_fake"), (0.5, "uncertain"), (0.1, "likely_real")])
def test_verdict_thresholds(prob, expected):
    assert main.service.verdict(prob) == expected
