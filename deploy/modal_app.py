"""Deploy the API on Modal (serverless, scales to zero when idle).

One-time setup:
  pip install modal
  modal setup                      # log in

Deploy (from the repo root, after training so model/checkpoints/best.pth exists):
  modal deploy deploy/modal_app.py

Modal prints a public URL like https://<you>--deepfake-detector-api.modal.run
Put that URL in the frontend's NEXT_PUBLIC_API_URL.

Set ALLOWED_ORIGINS to your Vercel URL so the browser is allowed to call it:
  ALLOWED_ORIGINS=https://your-app.vercel.app modal deploy deploy/modal_app.py

Inference runs on CPU (~0.3-1s per image with Grad-CAM), which is plenty for
a demo and costs nothing while idle. Add gpu="T4" to the function to use a GPU.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "model" / "checkpoints" / "best.pth"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .env({"ALLOWED_ORIGINS": os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")})
    .add_local_dir(str(ROOT / "app"), "/root/app")
    .add_local_dir(str(ROOT / "model"), "/root/model", ignore=["checkpoints/*", "__pycache__"])
)
if CHECKPOINT.exists():
    image = image.add_local_file(str(CHECKPOINT), "/root/model/checkpoints/best.pth")

app = modal.App("deepfake-detector", image=image)


@app.function(cpu=2.0, memory=2048, scaledown_window=300)
@modal.concurrent(max_inputs=4)
@modal.asgi_app(label="deepfake-detector-api")
def api():
    from app.main import app as fastapi_app

    return fastapi_app
