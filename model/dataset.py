"""Dataset of face crops, driven by data/faces/manifest.csv."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from model.transforms import get_eval_transforms, get_train_transforms


DataDirs = str | Path | list[str | Path] | tuple[str | Path, ...]


def read_manifest(data_dirs: DataDirs, split: str) -> list[dict]:
    """Rows for one split, from one or more dataset folders.

    Passing several folders (e.g. data/faces and data/faces_ffpp) trains on
    the datasets combined. Each row gets an absolute "file" path.
    """

    if isinstance(data_dirs, (str, Path)):
        data_dirs = [data_dirs]
    rows: list[dict] = []
    for data_dir in map(Path, data_dirs):
        manifest = data_dir / "manifest.csv"
        if not manifest.exists():
            raise FileNotFoundError(f"{manifest} not found. Run `python -m model.prepare_frames` first.")
        with manifest.open() as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    row["file"] = str(data_dir / row["path"])
                    rows.append(row)
    if not rows:
        raise ValueError(f"No rows for split '{split}' in {list(map(str, data_dirs))}")
    return rows


class FaceCropDataset(Dataset):
    """Returns (image_tensor, label_float, index). label: 1 = fake, 0 = real."""

    def __init__(self, data_dirs: DataDirs, split: str, transform: Callable | None = None) -> None:
        self.rows = read_manifest(data_dirs, split)
        self.labels = [int(r["label"]) for r in self.rows]
        self.video_ids = [r["video_id"] for r in self.rows]
        self.sources = [r["source"] for r in self.rows]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        image = Image.open(self.rows[index]["file"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(self.labels[index], dtype=torch.float32), index


def balanced_sampler(labels: list[int]) -> WeightedRandomSampler:
    """Sample real and fake crops equally often.

    Celeb-DF has ~6x more fake videos than real ones. Without this, a model
    that always answers "fake" already scores ~86% accuracy and learns little.
    """

    counts = Counter(labels)
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


def build_loader(
    data_dir: DataDirs,
    split: str,
    batch_size: int = 32,
    num_workers: int = 2,
    transform: Callable | None = None,
) -> DataLoader:
    is_train = split == "train"
    if transform is None:
        transform = get_train_transforms() if is_train else get_eval_transforms()
    dataset = FaceCropDataset(data_dir, split, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=balanced_sampler(dataset.labels) if is_train else None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
