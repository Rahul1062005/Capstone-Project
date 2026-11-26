# CPU-only Python AI Microservice for Render / Hugging Face Spaces / Docker
FROM python:3.11-slim

WORKDIR /app

# libsndfile1 is required by soundfile on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Install CPU-only PyTorch wheels first to avoid multi-GB CUDA downloads and OOM on 512MB builds
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    grep -vE '^(torch|torchaudio)' requirements.txt | pip install --no-cache-dir -r /dev/stdin

COPY ai_service ./ai_service
RUN mkdir -p /app/models /tmp/numba_cache

ENV PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn ai_service.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
