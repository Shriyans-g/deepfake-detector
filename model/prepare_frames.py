"""Turn deepfake video datasets into datasets of face crops.

Supports Celeb-DF v2 and FaceForensics++ (FF++).

What this does:
  1. Splits videos into train / val / test.
     Celeb-DF:
     - test = the official List_of_testing_videos.txt (518 videos), so
       results are comparable to published numbers.
     - val  = every remaining video that involves a held-out set of celebrity
       IDs. No val identity ever appears in train, so val measures
       generalization to new people instead of memorized faces.
     - train = everything else.
     FF++:
     - the official split files (model/splits/ffpp_*.json, 720/140/140 source
       videos). A fake "A_B" goes in the same split as originals A and B.
  2. Samples N evenly spaced frames per video.
  3. Detects the face in each frame and saves a 256x256 crop.
  4. Writes <output-dir>/manifest.csv (path, label, split, video_id, source)
     which the training and eval code read.

Usage:
  python -m model.prepare_frames --celeb-df-root ~/Documents/Celeb-DF-v2
  python -m model.prepare_frames --ffpp-root ~/Documents/FaceForensics   # -> data/faces_ffpp

Quick test on a handful of videos:
  python -m model.prepare_frames --celeb-df-root ... --limit-per-source 5

Safe to re-run: videos whose crops already exist are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from model.faces import FaceDetector, crop_face

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "faces"
DEFAULT_FFPP_OUTPUT_DIR = PROJECT_ROOT / "data" / "faces_ffpp"
FFPP_SPLITS_DIR = Path(__file__).resolve().parent / "splits"
FFPP_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")

# Celeb-DF folder name -> label. 1 = fake, 0 = real (matches the model output).
SOURCES = {"Celeb-real": 0, "YouTube-real": 0, "Celeb-synthesis": 1}
ID_PATTERN = re.compile(r"id(\d+)")


@dataclass(frozen=True)
class VideoJob:
    path: Path
    source: str
    label: int
    split: str

    @property
    def video_id(self) -> str:
        return f"{self.source}__{self.path.stem}"


def identities_in(filename: str) -> set[int]:
    """Celebrity IDs that appear in a video name.

    Celeb-real/id3_0007.mp4         -> {3}
    Celeb-synthesis/id3_id17_0002   -> {3, 17}  (target and source faces)
    YouTube-real/00042.mp4          -> set()    (no identity info)
    """

    return {int(m) for m in ID_PATTERN.findall(filename)}


def load_test_list(root: Path) -> set[str]:
    """Return the official test videos as 'Folder/name.mp4' strings."""

    test_file = root / "List_of_testing_videos.txt"
    if not test_file.exists():
        raise FileNotFoundError(f"Missing {test_file}")
    entries = set()
    for line in test_file.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            entries.add(parts[1])
    return entries


def assign_splits(
    root: Path,
    val_identity_fraction: float = 0.1,
    youtube_val_fraction: float = 0.1,
    seed: int = 42,
    limit_per_source: int | None = None,
) -> list[VideoJob]:
    """Decide which split every video goes into. Deterministic given the seed."""

    test_set = load_test_list(root)
    rng = random.Random(seed)

    all_ids = sorted(
        {i for p in (root / "Celeb-real").glob("*.mp4") for i in identities_in(p.name)}
    )
    n_val = max(1, round(len(all_ids) * val_identity_fraction))
    val_ids = set(rng.sample(all_ids, n_val))

    jobs: list[VideoJob] = []
    for source, label in SOURCES.items():
        videos = sorted((root / source).glob("*.mp4"))
        if limit_per_source:
            # Keep a mix of test and non-test videos in quick runs.
            test_videos = [v for v in videos if f"{source}/{v.name}" in test_set]
            other_videos = [v for v in videos if f"{source}/{v.name}" not in test_set]
            half = max(1, limit_per_source // 2)
            videos = test_videos[:half] + other_videos[: limit_per_source - half]

        for video in videos:
            if f"{source}/{video.name}" in test_set:
                split = "test"
            elif source == "YouTube-real":
                split = "val" if rng.random() < youtube_val_fraction else "train"
            elif identities_in(video.name) & val_ids:
                # Any held-out identity in the video sends it to val, so train
                # never contains a val person, not even as the swap source.
                split = "val"
            else:
                split = "train"
            jobs.append(VideoJob(video, source, label, split))

    return jobs


def assign_splits_ffpp(
    root: Path,
    compression: str = "c23",
    methods: tuple[str, ...] = FFPP_METHODS,
    limit_per_source: int | None = None,
) -> list[VideoJob]:
    """Assign FF++ videos to splits using the official split files.

    Layout (from the official download script):
      original_sequences/youtube/<c>/videos/000.mp4             (real)
      manipulated_sequences/<Method>/<c>/videos/000_003.mp4     (fake: target 000, source 003)

    Each split file lists pairs of source video IDs. Originals in a pair, and
    both fakes made from that pair (A_B and B_A), go into that split, so no
    person appears in more than one split.
    """

    split_of: dict[str, str] = {}
    for split in ("train", "val", "test"):
        pairs = json.loads((FFPP_SPLITS_DIR / f"ffpp_{split}.json").read_text())
        for a, b in pairs:
            split_of[a] = split_of[b] = split
            split_of[f"{a}_{b}"] = split_of[f"{b}_{a}"] = split

    folders = {"FFpp-original": (root / "original_sequences" / "youtube" / compression / "videos", 0)}
    for method in methods:
        folders[f"FFpp-{method}"] = (root / "manipulated_sequences" / method / compression / "videos", 1)

    jobs: list[VideoJob] = []
    for source, (folder, label) in folders.items():
        if not folder.exists():
            print(f"  warning: {folder} not found, skipping {source}")
            continue
        videos = sorted(folder.glob("*.mp4"))
        if limit_per_source:
            videos = videos[:limit_per_source]
        for video in videos:
            split = split_of.get(video.stem)
            if split is None:
                print(f"  warning: {video.name} is not in the official splits, skipping")
                continue
            jobs.append(VideoJob(video, source, label, split))
    if not jobs:
        raise FileNotFoundError(f"No FF++ videos found under {root} for compression '{compression}'.")
    return jobs


def sample_frame_indices(total_frames: int, n: int) -> list[int]:
    """n evenly spaced frame indices, skipping the very start and end."""

    if total_frames <= 0:
        return []
    n = min(n, total_frames)
    return sorted({int(i) for i in np.linspace(0, total_frames - 1, n + 2)[1:-1]}) or [0]


# Each worker process builds its own detector once.
_detector: FaceDetector | None = None


def _init_worker(detector_path: str | None) -> None:
    global _detector
    cv2.setNumThreads(1)
    _detector = FaceDetector(detector_path) if detector_path else FaceDetector()


def process_video(job: VideoJob, output_dir: Path, frames_per_video: int, jpeg_quality: int) -> list[dict]:
    """Extract face crops from one video and return manifest rows."""

    target_dir = output_dir / job.split / ("fake" if job.label else "real")
    target_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(job.path))
    if not capture.isOpened():
        return []

    wanted = set(sample_frame_indices(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), frames_per_video))
    last_wanted = max(wanted) if wanted else -1
    crops: list[tuple[int, np.ndarray]] = []
    index = 0
    try:
        while index <= last_wanted:
            # grab() skips the costly color conversion for frames we don't keep.
            if not capture.grab():
                break
            if index in wanted:
                ok, frame = capture.retrieve()
                if ok:
                    box = _detector.detect_largest(frame)
                    if box is not None:
                        crops.append((index, crop_face(frame, box)))
            index += 1
    finally:
        capture.release()

    # Write all crops at the end, so an interrupted run never leaves a
    # half-finished video that the resume check would mistake for done.
    rows: list[dict] = []
    for frame_index, crop in crops:
        out_path = target_dir / f"{job.video_id}__f{frame_index:05d}.jpg"
        cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        rows.append(_row(out_path, job, output_dir))
    return rows


def _row(path: Path, job: VideoJob, output_dir: Path) -> dict:
    return {
        "path": str(path.relative_to(output_dir)),
        "label": job.label,
        "split": job.split,
        "video_id": job.video_id,
        "source": job.source,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a face-crop dataset from Celeb-DF v2 or FaceForensics++.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--celeb-df-root", type=Path)
    source.add_argument("--ffpp-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: data/faces (Celeb-DF) or data/faces_ffpp (FF++).")
    parser.add_argument("--ffpp-compression", choices=["raw", "c23", "c40"], default="c23")
    parser.add_argument("--ffpp-methods", nargs="+", default=list(FFPP_METHODS))
    parser.add_argument("--frames-per-video", type=int, default=10)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--val-identity-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--limit-per-source", type=int, default=None, help="For quick test runs.")
    parser.add_argument("--detector-path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ffpp_root:
        output_dir = (args.output_dir or DEFAULT_FFPP_OUTPUT_DIR).expanduser()
        jobs = assign_splits_ffpp(
            args.ffpp_root.expanduser(),
            compression=args.ffpp_compression,
            methods=tuple(args.ffpp_methods),
            limit_per_source=args.limit_per_source,
        )
    else:
        output_dir = (args.output_dir or DEFAULT_OUTPUT_DIR).expanduser()
        jobs = assign_splits(
            args.celeb_df_root.expanduser(),
            val_identity_fraction=args.val_identity_fraction,
            seed=args.seed,
            limit_per_source=args.limit_per_source,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[tuple[str, int], int] = {}
    for job in jobs:
        counts[(job.split, job.label)] = counts.get((job.split, job.label), 0) + 1
    print("Videos per split (real / fake):")
    for split in ("train", "val", "test"):
        print(f"  {split:5s}  {counts.get((split, 0), 0):5d} / {counts.get((split, 1), 0):5d}")

    # Resume support: find videos that already have crops with one directory
    # scan, instead of globbing a folder of tens of thousands of files per video.
    existing: dict[str, list[Path]] = {}
    for path in output_dir.glob("*/*/*.jpg"):
        existing.setdefault(path.name.rsplit("__f", 1)[0], []).append(path)
    rows = [_row(p, job, output_dir) for job in jobs for p in sorted(existing.get(job.video_id, []))]
    jobs = [job for job in jobs if job.video_id not in existing]
    if existing:
        print(f"Resuming: {len(existing)} videos already done, {len(jobs)} to go")

    failed: list[str] = []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(args.detector_path,)
    ) as pool:
        futures = {
            pool.submit(process_video, job, output_dir, args.frames_per_video, args.jpeg_quality): job
            for job in jobs
        }
        for done, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            video_rows = future.result()
            if not video_rows:
                failed.append(job.video_id)
            rows.extend(video_rows)
            if done % 100 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} videos, {len(rows)} face crops")

    rows.sort(key=lambda r: r["path"])
    manifest = output_dir / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "split", "video_id", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} crops and {manifest}")
    if failed:
        print(f"{len(failed)} videos produced no faces (unreadable or no detection), e.g. {failed[:5]}")


if __name__ == "__main__":
    main()
