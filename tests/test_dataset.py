import csv

from PIL import Image

from model.dataset import FaceCropDataset


def make_dataset_dir(root, source, n=2):
    (root / "train" / "real").mkdir(parents=True)
    rows = []
    for i in range(n):
        rel = f"train/real/{source}__v{i}__f00001.jpg"
        Image.new("RGB", (8, 8)).save(root / rel)
        rows.append({"path": rel, "label": 0, "split": "train", "video_id": f"{source}__v{i}", "source": source})
    with (root / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_combines_multiple_dataset_folders(tmp_path):
    make_dataset_dir(tmp_path / "a", "Celeb-real")
    make_dataset_dir(tmp_path / "b", "FFpp-original", n=3)
    dataset = FaceCropDataset([tmp_path / "a", tmp_path / "b"], "train")
    assert len(dataset) == 5
    assert set(dataset.sources) == {"Celeb-real", "FFpp-original"}
    image, label, _ = dataset[4]  # loads from the second folder
    assert image.size == (8, 8) and label == 0
