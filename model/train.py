"""Train the deepfake classifier on face crops.

  python -m model.train --data-dir data/faces --arch resnet50 --epochs 8

What happens:
  - Fine-tunes an ImageNet backbone with a single "fake" logit (BCE loss).
  - Balanced sampling so real and fake crops are seen equally often.
  - AdamW + cosine learning-rate decay; mixed precision on CUDA.
  - Keeps the checkpoint with the best validation AUC, with early stopping.
  - Fits a temperature on the validation set at the end and stores it in the
    checkpoint, so the API's confidence scores are calibrated.
  - Writes reports/train_history.json and reports/training_curves.png.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from model.calibration import expected_calibration_error, fit_temperature, sigmoid
from model.dataset import build_loader
from model.metrics import collect_logits, safe_auc, video_level
from model.model_utils import SUPPORTED_ARCHS, build_model, get_device, load_checkpoint, save_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the deepfake face classifier.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "faces")
    parser.add_argument("--arch", choices=SUPPORTED_ARCHS, default="resnet50")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3, help="Epochs without val AUC improvement before stopping.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None, help="Cap steps per epoch (quick runs).")
    parser.add_argument("--checkpoint-path", type=Path, default=PROJECT_ROOT / "model" / "checkpoints" / "best.pth")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true", help="Random init instead of ImageNet weights (offline smoke tests only).")
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, max_steps=None) -> float:
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total, count = 0.0, 0
    for step, (images, labels, _) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler is not None):
            loss = loss_fn(model(images).squeeze(1), labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()
        total += loss.item() * images.size(0)
        count += images.size(0)
    return total / max(count, 1)


def evaluate(model, loader, device) -> dict:
    logits, labels, video_ids = collect_logits(model, loader, device)
    probs = sigmoid(logits)
    loss = float(nn.functional.binary_cross_entropy_with_logits(torch.tensor(logits), torch.tensor(labels, dtype=torch.float32)))
    video_probs, video_labels = video_level(probs, labels, video_ids)
    return {
        "loss": loss,
        "frame_auc": safe_auc(labels, probs),
        "video_auc": safe_auc(video_labels, video_probs),
        "accuracy": float(((probs >= 0.5) == labels).mean()),
    }


def plot_history(history: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], "o-", label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], "o-", label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("epoch")
    ax1.legend()
    ax2.plot(epochs, [h["val_frame_auc"] for h in history], "o-", label="val frame AUC")
    ax2.plot(epochs, [h["val_video_auc"] for h in history], "o-", label="val video AUC")
    ax2.set_title("Validation AUC")
    ax2.set_xlabel("epoch")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    print(f"Device: {device} | arch: {args.arch}")

    train_loader = build_loader(args.data_dir, "train", args.batch_size, args.num_workers)
    val_loader = build_loader(args.data_dir, "val", args.batch_size, args.num_workers)
    print(f"Train crops: {len(train_loader.dataset)} | Val crops: {len(val_loader.dataset)}")

    model = build_model(args.arch, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = min(len(train_loader), args.max_steps_per_epoch or math.inf)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=int(steps_per_epoch * args.epochs), pct_start=0.1
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_auc, epochs_without_improvement = -1.0, 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, args.max_steps_per_epoch)
        val = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}}
        history.append(record)
        print(
            f"Epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} | val loss {val['loss']:.4f} | "
            f"val frame AUC {val['frame_auc']:.4f} | val video AUC {val['video_auc']:.4f} | {time.time() - start:.0f}s"
        )

        score = val["frame_auc"] if not math.isnan(val["frame_auc"]) else -val["loss"]
        if score > best_auc:
            best_auc, epochs_without_improvement = score, 0
            save_checkpoint(args.checkpoint_path, model, args.arch, epoch=epoch, val_metrics=val)
            print(f"  saved best checkpoint -> {args.checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"  no improvement for {args.patience} epochs, stopping early")
                break

    (args.reports_dir / "train_history.json").write_text(json.dumps(history, indent=2))
    plot_history(history, args.reports_dir / "training_curves.png")

    # Calibrate the best model on the validation set.
    best_model, meta = load_checkpoint(args.checkpoint_path, device)
    logits, labels, _ = collect_logits(best_model, val_loader, device)
    temperature = fit_temperature(logits, labels)
    ece_before = expected_calibration_error(sigmoid(logits), labels)
    ece_after = expected_calibration_error(sigmoid(logits / temperature), labels)
    save_checkpoint(
        args.checkpoint_path,
        best_model,
        args.arch,
        epoch=meta["epoch"],
        val_metrics=meta["val_metrics"],
        temperature=temperature,
        val_ece_before=ece_before,
        val_ece_after=ece_after,
    )
    print(f"Temperature T={temperature:.3f} | val ECE {ece_before:.4f} -> {ece_after:.4f}")
    print(f"Done. Best checkpoint (epoch {meta['epoch']}): {args.checkpoint_path}")


if __name__ == "__main__":
    main()
