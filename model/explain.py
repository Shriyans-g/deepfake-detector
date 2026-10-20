"""Turn a prediction into evidence a person can read.

Two things, both measured rather than made up:

1. attention_by_region: what share of the Grad-CAM heatmap falls on each part
   of the face (eyes, nose, mouth, rest of the face, outside the face). Regions
   come from the face detector's five landmarks.
2. quality_notes: properties of the upload that our own robustness tests showed
   lower accuracy (tiny faces, blur, a weak face detection, several faces).
"""

from __future__ import annotations

import cv2
import numpy as np

from model.faces import CROP_MARGIN, FaceBox, crop_square

REGIONS = ("eyes", "nose", "mouth", "rest_of_face", "outside_face")


def _to_crop_coords(points: list[tuple[float, float]], box: FaceBox, size: int) -> np.ndarray:
    """Map image-pixel points into the size x size face crop."""

    x0, y0, x1, y1 = crop_square(box, CROP_MARGIN)
    pts = np.array(points, dtype=np.float64)
    # Width and height can differ by 1px (see crop_square), so scale each axis.
    return (pts - [x0, y0]) * [size / (x1 - x0), size / (y1 - y0)]


def region_masks(box: FaceBox, size: int) -> dict[str, np.ndarray]:
    """Boolean masks (size x size) for each face region, built from landmarks.

    Mouth, eyes and nose are boxes around their landmarks, scaled by the
    distance between the eyes so they adapt to how big the face is. Anything
    else inside the face box is "rest of face" (cheeks, forehead, jaw); the
    remaining margin of the crop is "outside face" (hair, neck, background).
    """

    yy, xx = np.mgrid[0:size, 0:size]

    def rect(cx, cy, half_w, half_h):
        return (np.abs(xx - cx) <= half_w) & (np.abs(yy - cy) <= half_h)

    face = _to_crop_coords([(box.x, box.y), (box.x + box.w, box.y + box.h)], box, size)
    (fx0, fy0), (fx1, fy1) = face
    in_face = (xx >= fx0) & (xx <= fx1) & (yy >= fy0) & (yy <= fy1)

    if len(box.landmarks) != 5:
        empty = np.zeros((size, size), dtype=bool)
        return {"eyes": empty, "nose": empty, "mouth": empty, "rest_of_face": in_face, "outside_face": ~in_face}

    right_eye, left_eye, nose, mouth_r, mouth_l = _to_crop_coords(box.landmarks, box, size)
    eye_dist = max(float(np.linalg.norm(left_eye - right_eye)), 1.0)

    eyes = rect(*right_eye, 0.32 * eye_dist, 0.22 * eye_dist) | rect(*left_eye, 0.32 * eye_dist, 0.22 * eye_dist)
    nose_m = rect(*nose, 0.25 * eye_dist, 0.30 * eye_dist)
    mouth_c = (mouth_r + mouth_l) / 2
    mouth_half_w = abs(mouth_l[0] - mouth_r[0]) / 2 + 0.2 * eye_dist
    mouth = rect(*mouth_c, mouth_half_w, 0.28 * eye_dist)

    # Overlaps go to the more specific region (mouth > eyes > nose).
    nose_m &= ~(mouth | eyes)
    eyes &= ~mouth
    rest = in_face & ~(eyes | nose_m | mouth)
    outside = ~(in_face | eyes | nose_m | mouth)
    return {"eyes": eyes, "nose": nose_m, "mouth": mouth, "rest_of_face": rest, "outside_face": outside}


def attention_by_region(heatmap: np.ndarray, box: FaceBox) -> list[dict]:
    """Share of total Grad-CAM activation in each region, largest first."""

    size = heatmap.shape[0]
    masks = region_masks(box, size)
    total = float(heatmap.sum())
    if total <= 0:
        return [{"region": r, "share": 0.0} for r in REGIONS]
    shares = [{"region": r, "share": round(float(heatmap[masks[r]].sum()) / total, 4)} for r in REGIONS]
    return sorted(shares, key=lambda s: s["share"], reverse=True)


def sharpness(crop_bgr: np.ndarray) -> float:
    """Variance of the Laplacian: a standard, cheap measure of fine detail.

    Low values mean a blurry or heavily smoothed face.
    """

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# Thresholds are rough heuristics; the notes say "may", not "does".
SMALL_FACE_PX = 100
# Measured with sharpness() on every train crop (Oct 2026):
#   Celeb-DF (47,027 crops): median 17.1, 1st percentile 6.09, min 4.0
#   FF++     (35,999 crops): median 80.4, 1st percentile 16.3, min 7.4
# Set just below the lower 1st percentile, so only uploads blurrier than about
# 99% of what the model trained on are flagged (0.9% of Celeb-DF train crops,
# none of FF++). A typical crop needs a Gaussian blur of sigma ~1.5-2 to drop
# below it, so mild blur deliberately goes unflagged.
BLUR_THRESHOLD = 6.0
WEAK_DETECTION = 0.85


def quality_notes(box: FaceBox, crop_bgr: np.ndarray) -> dict:
    """Facts about the upload that affect how much to trust the result."""

    face_px = int(max(box.w, box.h))
    sharp = sharpness(crop_bgr)
    notes: list[str] = []
    if face_px < SMALL_FACE_PX:
        notes.append(
            f"The face is small ({face_px}px). In our tests, shrinking faces to a quarter "
            "of their size lowered AUC from 0.98 to 0.91."
        )
    if sharp < BLUR_THRESHOLD:
        notes.append("The face looks blurry or heavily smoothed, which hides the fine detail the model relies on.")
    if box.score < WEAK_DETECTION:
        notes.append(f"The face detector was only {box.score:.0%} sure this is a face, so the crop may be off.")
    if box.faces_in_image > 1:
        notes.append(f"{box.faces_in_image} faces were found. Only the largest one was analyzed.")
    return {
        "face_px": face_px,
        "sharpness": round(sharp, 1),
        "detector_confidence": round(box.score, 4),
        "faces_in_image": box.faces_in_image,
        "notes": notes,
    }
