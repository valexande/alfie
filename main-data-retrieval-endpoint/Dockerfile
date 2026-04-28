# XAI Explainability API - Production Dockerfile
FROM python:3.11-slim

WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

# ── Python dependencies (grouped for clear layer caching) ─────────────────────

# 1. FastAPI web framework and async server
RUN uv pip install --system --no-cache \
    "fastapi>=0.109.0" \
    "uvicorn[standard]>=0.27.0" \
    "python-multipart>=0.0.6" \
    "pydantic>=2.0" \
    "pydantic-settings>=2.0"

# 2. AutoGluon — pinned to match the uploaded model version
RUN uv pip install --system --no-cache \
    "autogluon.tabular==1.4.0"

# 3. ML backends used by AutoGluon ensemble models
#    torchvision + pillow power the PyTorch vision-classifier explainer
#    (see xai_core/model_loader.load_vision_model_from_bytes and the
#    /explain-vision-model endpoint in api/routers/uc2.py).
RUN uv pip install --system --no-cache \
    "catboost>=1.2" \
    "xgboost>=2.0" \
    "lightgbm>=4.0" \
    "torch>=2.0" \
    "torchvision>=0.15" \
    "pillow>=9.0"

# 4. FastAI + fasttransform
#    fasttransform is a hard dependency of AutoGluon's NeuralNetFastAI model.
#    Without it, predictor.predict() raises "No module named 'fasttransform'"
#    for any ensemble that includes NeuralNetFastAI.  The API falls back to
#    the best non-FastAI model, but installing it here lets the full ensemble
#    run without any fallback.
RUN uv pip install --system --no-cache \
    "fastai>=2.3.1,<2.8" \
    "fasttransform>=0.0.2"

# 5. Explainability
RUN uv pip install --system --no-cache \
    "shap>=0.42"

# 6. Visualization
RUN uv pip install --system --no-cache \
    "matplotlib>=3.7" \
    "seaborn>=0.12"

# 7. Testing utilities (kept in image for CI convenience)
RUN uv pip install --system --no-cache \
    "pytest>=7.0" \
    "pytest-asyncio>=0.21.0" \
    "httpx>=0.24.0"

# ── Application code ───────────────────────────────────────────────────────────
COPY requirements.txt requirements.txt
COPY xai_core/ ./xai_core/
COPY api/ ./api/
COPY run.py .

# ── Security: run as non-root ──────────────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# ── Runtime configuration ──────────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV XAI_HOST=0.0.0.0
ENV XAI_PORT=5000

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "5000"]
