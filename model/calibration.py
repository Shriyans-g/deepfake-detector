"""Confidence calibration: temperature scaling and expected calibration error.

Neural nets tend to be overconfident: a raw 0.99 might be right only 85% of
the time. Temperature scaling divides the logit by one number T (fit on the
validation set) so that predicted probabilities match observed accuracy.
It never changes which class wins, only how confident the model claims to be.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    """Find T that minimizes BCE of sigmoid(logits / T) on held-out data."""

    logits_t = torch.as_tensor(logits, dtype=torch.float64)
    labels_t = torch.as_tensor(labels, dtype=torch.float64)
    # Optimize log(T) so T stays positive.
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)
    loss_fn = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(logits_t / log_t.exp(), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    # Clamp to a sane range. On tiny or broken validation sets the optimizer
    # can run off toward 0 or infinity, which would make every score 0/1.
    return float(min(max(log_t.exp().item(), 0.05), 20.0))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Average gap between confidence and accuracy, weighted by bin size.

    Uses the confidence of the predicted class (max(p, 1-p)), the standard
    definition for classifiers. 0 = perfectly calibrated.
    """

    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    predictions = (probs >= 0.5).astype(int)
    confidence = np.where(predictions == 1, probs, 1 - probs)
    correct = (predictions == labels).astype(float)

    # Confidence lives in [0.5, 1]; split that range into n_bins equal bins.
    bins = np.minimum(((confidence - 0.5) / 0.5 * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def reliability_diagram(
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    labels: np.ndarray,
    path: str | Path,
    n_bins: int = 10,
) -> None:
    """Plot predicted P(fake) vs actual fraction fake, before and after scaling."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    def curve(probs):
        idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
        return np.array([labels[idx == b].mean() if (idx == b).any() else np.nan for b in range(n_bins)])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(centers, curve(probs_before), "o-", label=f"Raw (ECE {expected_calibration_error(probs_before, labels):.3f})")
    ax.plot(centers, curve(probs_after), "s-", label=f"Temperature-scaled (ECE {expected_calibration_error(probs_after, labels):.3f})")
    ax.set_xlabel("Predicted P(fake)")
    ax.set_ylabel("Actual fraction fake")
    ax.set_title("Reliability diagram (test set)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
