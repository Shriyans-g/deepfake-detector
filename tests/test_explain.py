import numpy as np

from model.explain import REGIONS, attention_by_region, quality_notes, region_masks
from model.faces import FaceBox


def face_with_landmarks() -> FaceBox:
    # A 100x120 face at (100, 100) with plausible landmark positions.
    return FaceBox(
        100, 100, 100, 120, 0.98,
        landmarks=[(130, 145), (170, 145), (150, 170), (133, 195), (167, 195)],
    )


def test_regions_cover_every_pixel_exactly_once():
    masks = region_masks(face_with_landmarks(), 224)
    stacked = np.stack([masks[r] for r in REGIONS]).astype(int)
    assert (stacked.sum(axis=0) == 1).all()


def test_regions_are_where_they_should_be():
    masks = region_masks(face_with_landmarks(), 224)
    rows = {r: np.argwhere(masks[r])[:, 0].mean() for r in ("eyes", "nose", "mouth")}
    # Top to bottom: eyes, then nose, then mouth.
    assert rows["eyes"] < rows["nose"] < rows["mouth"]


def test_attention_shares_sum_to_one_and_follow_the_heatmap():
    box = face_with_landmarks()
    masks = region_masks(box, 224)
    heatmap = masks["mouth"].astype(float)  # all activation on the mouth
    shares = attention_by_region(heatmap, box)
    assert abs(sum(s["share"] for s in shares) - 1) < 1e-6
    assert shares[0] == {"region": "mouth", "share": 1.0}


def test_attention_without_landmarks_falls_back_to_face_vs_outside():
    box = FaceBox(100, 100, 100, 120, 0.98)
    shares = {s["region"]: s["share"] for s in attention_by_region(np.ones((224, 224)), box)}
    assert shares["eyes"] == shares["nose"] == shares["mouth"] == 0
    assert shares["rest_of_face"] > 0 and shares["outside_face"] > 0


def test_quality_notes_flag_small_blurry_multi_face_uploads():
    box = FaceBox(0, 0, 60, 60, 0.7, faces_in_image=3)
    flat_crop = np.full((256, 256, 3), 128, dtype=np.uint8)  # no detail at all
    report = quality_notes(box, flat_crop)
    assert report["face_px"] == 60
    assert len(report["notes"]) == 4


def test_quality_notes_empty_for_a_clean_upload():
    rng = np.random.default_rng(0)
    detailed_crop = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    report = quality_notes(FaceBox(0, 0, 300, 320, 0.99), detailed_crop)
    assert report["notes"] == []
