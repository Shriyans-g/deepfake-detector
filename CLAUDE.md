# CLAUDE.md

Context for Claude Code working in this repo.

## What this is

A resume project by Shriyans (Rutgers CS + Data Science). Detects face-swap deepfakes in images:
PyTorch classifier trained on Celeb-DF v2 face crops, served by FastAPI with calibrated
confidence and Grad-CAM heatmaps, plus a Next.js frontend. See README.md for the full design.

Shriyans needs to be able to explain every line in an interview. Prefer simple, readable code
over clever code, and explain non-obvious changes when you make them.

## Decisions already made (don't undo without a reason)

- **Face crops, not full frames.** `model/faces.py` (YuNet detector, 1.3x margin, 256px) is shared
  by the dataset builder and the API so training and inference inputs match exactly.
- **Identity-based split.** Val = videos with any held-out celebrity ID; test = Celeb-DF's official
  `List_of_testing_videos.txt`. Never split randomly by frame.
- **Single-logit output**, label 1 = fake. `sigmoid(logit / temperature)` = calibrated P(fake).
- **Balanced sampler + AUC** for model selection (dataset is ~6x more fake than real).
- **Temperature scaling** fit on val after training; stored in the checkpoint as `temperature`.
- **Grad-CAM implemented by hand** in `model/gradcam.py` (no pytorch-grad-cam dependency: it pulls
  in non-headless opencv, which breaks the slim Docker image).
- **RAG/ChromaDB/Ollama were removed.** The old version retrieved canned text based on only the
  label and confidence, so the LLM never saw the image and added no real information.
- Runtime deps (`requirements.txt`) stay minimal for the Docker image. Training-only deps go in
  `requirements-train.txt`, test deps in `requirements-dev.txt`.

## Commands

```bash
pip install -r requirements-dev.txt
python -m model.prepare_frames --celeb-df-root ~/Documents/Celeb-DF-v2   # build data/faces
python -m model.train --data-dir data/faces --arch resnet50              # GPU recommended (Colab notebook)
python -m model.eval --data-dir data/faces                               # writes reports/
uvicorn app.main:app --reload
pytest -q
cd frontend && npm run dev
```

Checkpoints go in `model/checkpoints/best.pth` (gitignored). Celeb-DF frames must never be committed
(dataset license forbids redistribution); `data/` is gitignored.

## Status / next steps

- [x] Code for data pipeline, training, eval, calibration, Grad-CAM, API, tests, Docker, CI, frontend, Modal deploy
- [x] Run `prepare_frames` on the full Celeb-DF dataset (65,277 crops from 6,529 videos; zipped at data/faces.zip)
- [ ] Train ResNet50 on Colab, then evaluate; fill the README results tables with real numbers
- [ ] Optional: train EfficientNet-B0 and compare
- [ ] Deploy backend (Modal) and frontend (Vercel); add demo GIF to README
- [ ] Optional: request FaceForensics++ access and add a cross-dataset test (train FF++, test Celeb-DF)
