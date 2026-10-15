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
