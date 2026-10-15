"""Test setup: a small randomly initialized checkpoint so the API can load.

The real checkpoint isn't in git (too big, and trained on Celeb-DF), so tests
build a resnet18 with random weights. Predictions are meaningless, but every
step of the pipeline runs for real.
"""

import os
import tempfile
from pathlib import Path

import torch

from model.model_utils import build_model, save_checkpoint

_tmp = Path(tempfile.mkdtemp())
CHECKPOINT = _tmp / "test.pth"
torch.manual_seed(0)
save_checkpoint(CHECKPOINT, build_model("resnet18", pretrained=False), "resnet18", epoch=1, temperature=1.5)
os.environ["MODEL_CHECKPOINT"] = str(CHECKPOINT)
