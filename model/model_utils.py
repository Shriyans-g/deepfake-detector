"""Model construction, device selection, and checkpoint I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models

# The model outputs ONE logit: higher = more likely fake.
# sigmoid(logit / temperature) is the calibrated probability of "fake".
SUPPORTED_ARCHS = ("resnet18", "resnet50", "efficientnet_b0", "efficientnet_b4")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def build_model(arch: str = "resnet50", pretrained: bool = True) -> nn.Module:
    """ImageNet backbone with its classifier replaced by a single-logit head."""

    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    elif arch == "efficientnet_b4":
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {SUPPORTED_ARCHS}.")
    return model


def gradcam_target_layer(model: nn.Module, arch: str) -> nn.Module:
    """The last convolutional block, which Grad-CAM hooks into."""

    if arch.startswith("resnet"):
        return model.layer4[-1]
    if arch.startswith("efficientnet"):
        return model.features[-1]
    raise ValueError(f"No Grad-CAM layer defined for '{arch}'.")


def save_checkpoint(path: str | Path, model: nn.Module, arch: str, **metadata: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": arch, "model_state_dict": model.state_dict(), **metadata}, path)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild the model from a checkpoint. Returns (model in eval mode, metadata)."""

    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    model = build_model(checkpoint["arch"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    metadata.setdefault("temperature", 1.0)
    return model, metadata
