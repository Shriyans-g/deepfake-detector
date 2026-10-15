"""Evaluate a trained checkpoint on the test split of one or more datasets.

  python -m model.eval --data-dir data/faces --checkpoint-path model/checkpoints/best.pth
  python -m model.eval --data-dir data/faces data/faces_ffpp   # both combined

Reports (all written to reports/):
  - Frame-level and video-level AUC, accuracy, precision, recall, F1
  - Confusion matrix
  - Calibration: ECE before and after temperature scaling + reliability diagram
  - ROC curve
  - Robustness: how frame AUC changes under JPEG compression, downscaling, blur
  - metrics.json with everything, and a markdown table printed for the README
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve
from torchvision import transforms

from model.calibration import expected_calibration_error, reliability_diagram, sigmoid
from model.dataset import build_loader
from model.metrics import collect_logits, safe_auc, video_level
from model.model_utils import get_device, load_checkpoint
from model.transforms import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class JPEG:
    def __init__(self, quality: int) -> None:
        self.quality = quality

    def __call__(self, image: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class Downscale:
    def __init__(self, factor: float) -> None:
        self.factor = factor

    def __call__(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        small = image.resize((max(1, int(w * self.factor)), max(1, int(h * self.factor))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


class Blur:
    def __init__(self, radius: float) -> None:
        self.radius = radius

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(self.radius))


DEGRADATIONS = {
    "clean": None,
    "jpeg_q70": JPEG(70),
    "jpeg_q50": JPEG(50),
    "jpeg_q30": JPEG(30),
    "downscale_0.5x": Downscale(0.5),
    "downscale_0.25x": Downscale(0.25),
    "blur_r1.5": Blur(1.5),
}


def degraded_transform(degradation) -> transforms.Compose:
    steps = [degradation] if degradation else []
    return transforms.Compose(
        steps
        + [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def classification_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds = (probs >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "auc": safe_auc(labels, probs),
        "accuracy": float((preds == labels).mean()),
        "precision_fake": float(precision),
        "recall_fake": float(recall),
        "f1_fake": float(f1),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
        "n": int(len(labels)),
    }


def dataset_of(source: str) -> str:
    return "FF++" if source.startswith("FFpp-") else "Celeb-DF"


def breakdowns(probs: np.ndarray, labels: np.ndarray, video_ids: list[str], sources: list[str]) -> dict:
    """AUC per dataset, and per manipulation method (each fake source vs. that dataset's reals).

    The per-method numbers show which kinds of fakes the model finds easy or
    hard, e.g. FF++ NeuralTextures is usually much harder than FaceSwap.
    """

    sources_arr = np.array(sources)
    datasets = np.array([dataset_of(s) for s in sources])
    out: dict = {"by_dataset": {}, "by_method": {}}

    def scores(mask: np.ndarray) -> dict:
        vids = [v for v, m in zip(video_ids, mask) if m]
        vp, vl = video_level(probs[mask], labels[mask], vids)
        return {"frame_auc": safe_auc(labels[mask], probs[mask]), "video_auc": safe_auc(vl, vp), "n_frames": int(mask.sum())}

    for ds in sorted(set(datasets)):
        out["by_dataset"][ds] = scores(datasets == ds)
    for src in sorted({s for s, y in zip(sources, labels) if y == 1}):
        ds = dataset_of(src)
        mask = (datasets == ds) & ((labels == 0) | (sources_arr == src))
        out["by_method"][src] = scores(mask)
    return out


def plot_roc(frame: tuple, video: tuple, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (labels, probs) in {"frame": frame, "video": video}.items():
        if len(np.unique(labels)) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC {safe_auc(labels, probs):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC (test split)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the test split of one or more datasets.")
    parser.add_argument("--data-dir", type=Path, nargs="+", default=[PROJECT_ROOT / "data" / "faces"], help="One or more dataset folders; several = evaluate on them combined.")
    parser.add_argument("--checkpoint-path", type=Path, default=PROJECT_ROOT / "model" / "checkpoints" / "best.pth")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--split", default="test")
    parser.add_argument("--skip-robustness", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model, meta = load_checkpoint(args.checkpoint_path, device)
    temperature = float(meta["temperature"])
    print(f"Loaded {meta['arch']} (epoch {meta.get('epoch')}), T={temperature:.3f}, device {device}")

    loader = build_loader(args.data_dir, args.split, args.batch_size, args.num_workers, degraded_transform(None))
    logits, labels, video_ids = collect_logits(model, loader, device)
    raw_probs = sigmoid(logits)
    probs = sigmoid(logits / temperature)
    video_probs, video_labels = video_level(probs, labels, video_ids)
    sources = loader.dataset.sources  # loader is unshuffled, so order matches

    results = {
        "checkpoint": str(args.checkpoint_path),
        "arch": meta["arch"],
        "temperature": temperature,
        "frame": classification_metrics(probs, labels),
        "video": classification_metrics(video_probs, video_labels),
        "breakdown": breakdowns(probs, labels, video_ids, sources),
        "calibration": {
            "ece_raw": expected_calibration_error(raw_probs, labels),
            "ece_temperature_scaled": expected_calibration_error(probs, labels),
        },
    }
    reliability_diagram(raw_probs, probs, labels, args.reports_dir / "reliability_diagram.png")
    plot_roc((labels, probs), (video_labels, video_probs), args.reports_dir / "roc_curve.png")

    if not args.skip_robustness:
        robustness = {}
        for name, degradation in DEGRADATIONS.items():
            if degradation is None:
                robustness[name] = results["frame"]["auc"]
                continue
            deg_loader = build_loader(
                args.data_dir, args.split, args.batch_size, args.num_workers, degraded_transform(degradation)
            )
            deg_logits, deg_labels, _ = collect_logits(model, deg_loader, device)
            robustness[name] = safe_auc(deg_labels, deg_logits)
            print(f"  robustness {name:16s} frame AUC {robustness[name]:.4f}")
        results["robustness_frame_auc"] = robustness

    (args.reports_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    f, v, c = results["frame"], results["video"], results["calibration"]
    print("\n| Level | AUC | Accuracy | Precision (fake) | Recall (fake) | F1 (fake) | N |")
    print("|---|---|---|---|---|---|---|")
    for name, m in (("Frame", f), ("Video", v)):
        print(
            f"| {name} | {m['auc']:.3f} | {m['accuracy']:.3f} | {m['precision_fake']:.3f} | "
            f"{m['recall_fake']:.3f} | {m['f1_fake']:.3f} | {m['n']} |"
        )
    print(f"\nECE: {c['ece_raw']:.4f} raw -> {c['ece_temperature_scaled']:.4f} after temperature scaling")
    if "robustness_frame_auc" in results:
        print("\n| Condition | Frame AUC |\n|---|---|")
        for name, auc in results["robustness_frame_auc"].items():
            print(f"| {name} | {auc:.3f} |")
    b = results["breakdown"]
    if len(b["by_dataset"]) > 1 or len(b["by_method"]) > 1:
        print("\n| Subset | Frame AUC | Video AUC |\n|---|---|---|")
        for name, m in list(b["by_dataset"].items()) + list(b["by_method"].items()):
            print(f"| {name} | {m['frame_auc']:.3f} | {m['video_auc']:.3f} |")
    print(f"\nSaved metrics and plots to {args.reports_dir}/")


if __name__ == "__main__":
    main()
