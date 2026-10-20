# CLAUDE.md

Context for Claude Code working in this repo.

## What this is

A resume project by Shriyans (Rutgers CS + Data Science). Detects deepfake face manipulations in images:
PyTorch classifier trained on Celeb-DF v2 and FaceForensics++ (c23) face crops, served by FastAPI with calibrated
confidence and Grad-CAM heatmaps, plus a Next.js frontend. See README.md for the full design.

Shriyans needs to be able to explain every line in an interview. Prefer simple, readable code
over clever code, and explain non-obvious changes when you make them.

## Decisions already made (don't undo without a reason)

- **Face crops, not full frames.** `model/faces.py` (YuNet detector, 1.3x margin, 256px) is shared
  by the dataset builder and the API so training and inference inputs match exactly.
- **Identity-based split.** Celeb-DF: val = videos with any held-out celebrity ID; test = the official
  `List_of_testing_videos.txt`. FF++: official split files in `model/splits/`. Never split randomly by frame.
- **Three experiments** (see notebook): A = Celeb-DF only, B = FF++ only tested on FF++ and on unseen
  Celeb-DF (the cross-dataset generalization result), C = both combined (the deployed model).
- `--data-dir` in train/eval accepts several dataset folders (combined); eval prints per-dataset and
  per-method AUC (sources are named `Celeb-*` / `FFpp-<Method>`).
- **Single-logit output**, label 1 = fake. `sigmoid(logit / temperature)` = calibrated P(fake).
- **Balanced sampler + AUC** for model selection (dataset is ~6x more fake than real).
- **Temperature scaling** fit on val after training; stored in the checkpoint as `temperature`.
- **Grad-CAM implemented by hand** in `model/gradcam.py` (no pytorch-grad-cam dependency: it pulls
  in non-headless opencv, which breaks the slim Docker image).
- **RAG/ChromaDB/Ollama were removed.** The old version retrieved canned text based on only the
  label and confidence, so the LLM never saw the image and added no real information.
- **Explanations are measured, not generated.** `model/explain.py` turns the Grad-CAM heatmap into per-region shares
  (landmark-based masks) and adds quality notes tied to our robustness results. The frontend's benchmark numbers
  live in `frontend/app/benchmarks.ts` (copied from reports/*/metrics.json; update them if you retrain).
- Runtime deps (`requirements.txt`) stay minimal for the Docker image. Training-only deps go in
  `requirements-train.txt`, test deps in `requirements-dev.txt`.

## Commands

```bash
pip install -r requirements-dev.txt
python -m model.prepare_frames --celeb-df-root ~/Documents/Celeb-DF-v2   # build data/faces
python -m model.prepare_frames --ffpp-root ~/Documents/FaceForensics     # build data/faces_ffpp
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
- [x] Repo pushed to GitHub (Shriyans-g/deepfake-detector)
- [x] FF++ c23 downloaded to ~/Documents/FaceForensics; prepare_frames --ffpp-root done (49,999 crops from 5,000 videos; zipped at data/faces_ffpp.zip)
- [x] Ran experiments A, B, C on Colab; results in reports/*/metrics.json; README results filled in
- [x] Checkpoint C is at model/checkpoints/best.pth (gitignored; ~94MB)
- [ ] Deploy backend (Docker on Google Cloud Run, using MODEL_URL from a GitHub Release) and frontend (Vercel); add demo GIF to README
