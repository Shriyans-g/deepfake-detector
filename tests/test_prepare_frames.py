from pathlib import Path

from model.prepare_frames import assign_splits, identities_in, sample_frame_indices


def make_fake_celebdf(root: Path) -> None:
    for folder in ("Celeb-real", "Celeb-synthesis", "YouTube-real"):
        (root / folder).mkdir(parents=True)
    for i in range(20):
        for clip in range(3):
            (root / "Celeb-real" / f"id{i}_{clip:04d}.mp4").touch()
            (root / "Celeb-synthesis" / f"id{i}_id{(i + 1) % 20}_{clip:04d}.mp4").touch()
    for n in range(30):
        (root / "YouTube-real" / f"{n:05d}.mp4").touch()
    (root / "List_of_testing_videos.txt").write_text(
        "1 Celeb-real/id0_0000.mp4\n0 Celeb-synthesis/id0_id1_0000.mp4\n1 YouTube-real/00000.mp4\n"
    )


def test_identities_parsing():
    assert identities_in("id3_0007.mp4") == {3}
    assert identities_in("id3_id17_0002.mp4") == {3, 17}
    assert identities_in("00042.mp4") == set()


def test_official_test_list_is_respected(tmp_path):
    make_fake_celebdf(tmp_path)
    jobs = assign_splits(tmp_path, val_identity_fraction=0.2)
    test = {f"{j.source}/{j.path.name}" for j in jobs if j.split == "test"}
    assert test == {"Celeb-real/id0_0000.mp4", "Celeb-synthesis/id0_id1_0000.mp4", "YouTube-real/00000.mp4"}


def test_no_val_identity_leaks_into_train(tmp_path):
    make_fake_celebdf(tmp_path)
    jobs = assign_splits(tmp_path, val_identity_fraction=0.2)
    train_ids = set().union(*(identities_in(j.path.name) for j in jobs if j.split == "train"))
    val_real_ids = set().union(*(identities_in(j.path.name) for j in jobs if j.split == "val" and j.source == "Celeb-real"))
    assert val_real_ids, "expected some held-out identities"
    assert not (train_ids & val_real_ids)


def test_splits_are_deterministic(tmp_path):
    make_fake_celebdf(tmp_path)
    first = [(j.path.name, j.split) for j in assign_splits(tmp_path, seed=7)]
    second = [(j.path.name, j.split) for j in assign_splits(tmp_path, seed=7)]
    assert first == second


def test_frame_sampling():
    assert sample_frame_indices(300, 10) == sorted(set(sample_frame_indices(300, 10)))
    assert len(sample_frame_indices(300, 10)) == 10
    assert all(0 < i < 299 for i in sample_frame_indices(300, 10))
    assert sample_frame_indices(0, 10) == []
    assert len(sample_frame_indices(3, 10)) >= 1


def make_fake_ffpp(root: Path) -> None:
    import json

    from model.prepare_frames import FFPP_SPLITS_DIR

    for split in ("train", "val", "test"):
        for a, b in json.loads((FFPP_SPLITS_DIR / f"ffpp_{split}.json").read_text())[:3]:
            real_dir = root / "original_sequences" / "youtube" / "c23" / "videos"
            real_dir.mkdir(parents=True, exist_ok=True)
            (real_dir / f"{a}.mp4").touch()
            (real_dir / f"{b}.mp4").touch()
            for method in ("Deepfakes", "NeuralTextures"):
                fake_dir = root / "manipulated_sequences" / method / "c23" / "videos"
                fake_dir.mkdir(parents=True, exist_ok=True)
                (fake_dir / f"{a}_{b}.mp4").touch()
                (fake_dir / f"{b}_{a}.mp4").touch()


def test_ffpp_uses_official_splits_without_identity_overlap(tmp_path):
    from model.prepare_frames import assign_splits_ffpp

    make_fake_ffpp(tmp_path)
    jobs = assign_splits_ffpp(tmp_path, methods=("Deepfakes", "NeuralTextures"))
    assert {j.split for j in jobs} == {"train", "val", "test"}

    ids_by_split: dict[str, set[str]] = {}
    for job in jobs:
        ids_by_split.setdefault(job.split, set()).update(job.path.stem.split("_"))
    assert not (ids_by_split["train"] & ids_by_split["test"])
    assert not (ids_by_split["train"] & ids_by_split["val"])

    # Each pair gives 2 originals + 2 fakes per method.
    test_jobs = [j for j in jobs if j.split == "test"]
    assert sum(j.label == 0 for j in test_jobs) == 6
    assert sum(j.label == 1 for j in test_jobs) == 12
