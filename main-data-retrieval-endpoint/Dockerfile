# Metadata-compatible tabular runtime. CUDA profile is not a GPU validation claim.
ARG PYTHON_VERSION=3.12.12
FROM docker.io/library/python:${PYTHON_VERSION}-slim

# Preserve a CUDA-capable deployment default; local validation uses --build-arg TORCH_PROFILE=cpu.
ARG TORCH_PROFILE=cu126
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.8.22

# One resolver transaction, including backend constraints: later layers cannot override them.
COPY requirements.txt runtime-constraints.txt ./
RUN case "$TORCH_PROFILE" in cpu|cu126) ;; *) echo "Unsupported TORCH_PROFILE"; exit 2;; esac \
    && printf 'torch==2.7.1+%s\ntorchvision==0.22.1+%s\n' "$TORCH_PROFILE" "$TORCH_PROFILE" > /tmp/torch-constraints.txt \
    && uv pip install --system --no-cache \
       --extra-index-url "https://download.pytorch.org/whl/${TORCH_PROFILE}" \
       --index-strategy unsafe-best-match \
       -r requirements.txt -c runtime-constraints.txt -c /tmp/torch-constraints.txt \
    && python -m pip check \
    && python -m pip freeze > /app/runtime-freeze.txt \
    && rm /tmp/torch-constraints.txt

COPY xai_core/ ./xai_core/
COPY api/ ./api/
COPY bias_reporting/ ./bias_reporting/
COPY run.py .
COPY scripts/validate_runtime.py ./scripts/validate_runtime.py

RUN useradd --uid 1000 --create-home --shell /bin/bash appuser
USER appuser
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    XAI_HOST=0.0.0.0 \
    XAI_PORT=5000
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "5000"]
