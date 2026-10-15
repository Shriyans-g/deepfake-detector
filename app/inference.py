"""The prediction pipeline behind /predict.

image -> find face -> crop (same code as training) -> model -> calibrated
probability -> Grad-CAM heatmap.
"""

from __future__ import annotations

import base64
import io
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from model.calibration import sigmoid
from model.faces import FaceDetector, crop_face
from model.gradcam import GradCAM, overlay_heatmap
from model.model_utils import get_device, gradcam_target_layer, load_checkpoint
from model.transforms import get_eval_transforms


class ModelNotReady(RuntimeError):
    pass


class NoFaceFound(ValueError):
    pass


def to_data_url(image_rgb: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class DetectorService:
    def __init__(
        self,
        checkpoint_path: Path,
        face_detector_path: Path,
        decision_margin: float = 0.7,
        model_url: str = "",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.model_url = model_url
        self.decision_margin = decision_margin
        self.face_detector = FaceDetector(face_detector_path)
        self.transform = get_eval_transforms()
        # Grad-CAM stores per-request state in hooks, so requests run one at a
        # time through the model. Fine for a demo; scale out with replicas.
        self._lock = threading.Lock()
        self.model = None
        self.meta: dict = {}
        self.gradcam: GradCAM | None = None
        self.device = get_device()
        self.try_load()

    @property
    def ready(self) -> bool:
        return self.model is not None

    def try_load(self) -> None:
        if self.ready:
            return
        if not self.checkpoint_path.exists() and self.model_url:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(self.model_url, self.checkpoint_path)
        if not self.checkpoint_path.exists():
            return
        model, meta = load_checkpoint(self.checkpoint_path, self.device)
        self.model, self.meta = model, meta
        self.gradcam = GradCAM(model, gradcam_target_layer(model, meta["arch"]))

    def model_info(self) -> dict:
        return {
            "arch": self.meta.get("arch", "unknown"),
            "temperature": float(self.meta.get("temperature", 1.0)),
            "epoch": self.meta.get("epoch"),
            "val_metrics": self.meta.get("val_metrics"),
        }

    def verdict(self, prob_fake: float) -> str:
        if prob_fake >= self.decision_margin:
            return "likely_fake"
        if prob_fake <= 1 - self.decision_margin:
            return "likely_real"
        return "uncertain"

    def predict(self, image: Image.Image) -> dict:
        self.try_load()
        if not self.ready:
            raise ModelNotReady(f"No trained checkpoint at {self.checkpoint_path}. Train the model first.")

        start = time.perf_counter()
        image_bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        box = self.face_detector.detect_largest(image_bgr)
        if box is None:
            raise NoFaceFound("No face detected. This model only analyzes images containing a face.")

        crop_rgb = cv2.cvtColor(crop_face(image_bgr, box), cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(crop_rgb)).unsqueeze(0).to(self.device)
        temperature = float(self.meta["temperature"])

        with self._lock:
            # First pass tells us the answer; then explain that answer.
            logit, heatmap = self.gradcam(tensor.clone(), explain_fake=True)
            explains_fake = logit >= 0
            if not explains_fake:
                _, heatmap = self.gradcam(tensor.clone(), explain_fake=False)

        prob_fake = float(sigmoid(np.array(logit / temperature)))
        display = cv2.resize(crop_rgb, (tensor.shape[3], tensor.shape[2]))
        return {
            "verdict": self.verdict(prob_fake),
            "prob_fake": round(prob_fake, 4),
            "raw_prob_fake": round(float(sigmoid(np.array(logit))), 4),
            "face_box": box.as_dict(),
            "face_crop": to_data_url(display),
            "heatmap": to_data_url(overlay_heatmap(display, heatmap)),
            "heatmap_explains": "fake" if explains_fake else "real",
            "model": self.model_info(),
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }
