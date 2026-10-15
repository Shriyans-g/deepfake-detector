"""Grad-CAM: which parts of the face drove the prediction.

How it works:
  1. Run the image forward and keep the activations of the last conv block
     (a stack of feature maps, e.g. 2048 maps of 7x7 for ResNet50).
  2. Backpropagate the output score to get the gradient of that score with
     respect to each feature map.
  3. Average each map's gradient into one weight: "how much does this map
     matter for the score?"
  4. Weighted sum of the maps, ReLU (keep only positive evidence), upsample
     to image size, normalize to [0, 1].

Bright regions are where the model found evidence for its answer.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._on_forward)

    def _on_forward(self, _module, _inputs, output: torch.Tensor) -> None:
        self._activations = output.detach()
        # A tensor hook (rather than a module backward hook) plays nicely with
        # the in-place ReLUs inside ResNet blocks.
        if output.requires_grad:
            output.register_hook(self._save_gradients)

    def _save_gradients(self, grad: torch.Tensor) -> None:
        self._gradients = grad.detach()

    def __call__(self, image_tensor: torch.Tensor, explain_fake: bool = True) -> tuple[float, np.ndarray]:
        """Returns (raw logit, heatmap HxW in [0, 1]) for a 1xCxHxW input.

        explain_fake=True highlights evidence for "fake"; False highlights
        evidence for "real" (we backprop the negated logit).
        """

        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logit = self.model(image_tensor.requires_grad_(True)).squeeze()
            (logit if explain_fake else -logit).backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # 1 x C x 1 x 1
        cam = torch.relu((weights * self._activations).sum(dim=1)).squeeze(0)
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        heatmap = cv2.resize(cam, (image_tensor.shape[3], image_tensor.shape[2]))
        return float(logit.detach().cpu()), heatmap


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0, 1] heatmap onto an RGB uint8 image."""

    heatmap = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
    colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return np.uint8(image_rgb * (1 - alpha) + colored * alpha)
