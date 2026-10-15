"""Shared helpers for running a model over a loader and scoring it."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader


@torch.inference_mode()
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run the model over a loader. Returns (logits, labels, video_ids)."""

    model.eval()
    logits, labels, video_ids = [], [], []
    dataset = loader.dataset
    for images, batch_labels, indices in loader:
        out = model(images.to(device, non_blocking=True)).squeeze(1)
        logits.append(out.float().cpu().numpy())
        labels.append(batch_labels.numpy())
        video_ids.extend(dataset.video_ids[i] for i in indices.tolist())
    return np.concatenate(logits), np.concatenate(labels).astype(int), video_ids


def video_level(probs: np.ndarray, labels: np.ndarray, video_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Average frame probabilities per video. Returns (video_probs, video_labels)."""

    grouped: dict[str, list[float]] = defaultdict(list)
    video_label: dict[str, int] = {}
    for p, y, vid in zip(probs, labels, video_ids):
        grouped[vid].append(float(p))
        video_label[vid] = int(y)
    vids = sorted(grouped)
    return np.array([np.mean(grouped[v]) for v in vids]), np.array([video_label[v] for v in vids])


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC, or NaN if only one class is present (e.g. tiny test runs)."""

    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))
