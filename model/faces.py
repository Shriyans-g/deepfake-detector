"""Face detection and cropping.

Both the dataset builder and the API use this module, so training crops and
inference crops are produced the exact same way. If they drifted apart, the
model would see different inputs in production than it saw in training.

Detector: YuNet (OpenCV's built-in DNN face detector). The model file is
~230KB, ships in model/weights/, and only needs opencv, so it runs the same
on a Mac, in Colab, and inside the Docker image.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_DETECTOR_PATH = (
    Path(__file__).resolve().parent / "weights" / "face_detection_yunet_2023mar.onnx"
)

# Crops are saved at this size; training then random-crops down to 224.
CROP_SIZE = 256
# How much context to keep around the detected face box. Blending artifacts in
# face swaps often sit on the boundary between the face and the rest of the
# head, so a tight crop would throw away useful signal.
CROP_MARGIN = 1.3


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    score: float

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h, "score": round(self.score, 4)}


class FaceDetector:
    """Thin wrapper around cv2.FaceDetectorYN that returns the main face."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_DETECTOR_PATH,
        score_threshold: float = 0.7,
        max_side: int = 640,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Face detector weights not found: {model_path}")
        self.max_side = max_side
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), score_threshold, 0.3, 50
        )

    def detect_largest(self, image_bgr: np.ndarray) -> FaceBox | None:
        """Return the largest confident face in the image, or None."""

        h, w = image_bgr.shape[:2]
        # Detect on a downscaled copy for speed, then map the box back.
        scale = min(1.0, self.max_side / max(h, w))
        small = cv2.resize(image_bgr, (round(w * scale), round(h * scale))) if scale < 1 else image_bgr

        self._detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self._detector.detect(small)
        if faces is None or len(faces) == 0:
            return None

        # Each row: x, y, w, h, 10 landmark coords, score.
        best = max(faces, key=lambda f: f[2] * f[3])
        x, y, bw, bh = (best[:4] / scale).tolist()
        return FaceBox(int(x), int(y), int(bw), int(bh), float(best[-1]))


def crop_face(
    image_bgr: np.ndarray,
    box: FaceBox,
    margin: float = CROP_MARGIN,
    size: int = CROP_SIZE,
) -> np.ndarray:
    """Cut a square crop around the face box and resize it to size x size."""

    img_h, img_w = image_bgr.shape[:2]
    cx, cy = box.x + box.w / 2, box.y + box.h / 2
    side = max(box.w, box.h) * margin

    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = int(round(cx + side / 2)), int(round(cy + side / 2))

    # Pad with black if the square runs off the edge, instead of shifting it.
    # Shifting would move the face off-center differently near borders.
    pad = max(0, -x0, -y0, x1 - img_w, y1 - img_h)
    if pad:
        image_bgr = cv2.copyMakeBorder(image_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad

    crop = image_bgr[y0:y1, x0:x1]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
