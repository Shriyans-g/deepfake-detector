FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only PyTorch: the default wheel bundles CUDA and is ~2GB bigger.
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY model ./model

# Run as a non-root user.
# (It owns the checkpoints folder so MODEL_URL downloads can be written.)
RUN useradd --create-home appuser \
    && mkdir -p /app/model/checkpoints \
    && chown -R appuser /app/model/checkpoints
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Render/Railway/Fly pass the port in $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
