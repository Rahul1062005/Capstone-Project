# CPU-only Python AI Microservice for Render / Hugging Face Spaces / Docker
FROM python:3.11-slim

WORKDIR /app

# libsndfile1 is required by soundfile on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Install CPU-only PyTorch wheel first (skip unused torchaudio to fit 512MB RAM limits)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    grep -vE '^(torch|torchaudio)' requirements.txt | pip install --no-cache-dir -r /dev/stdin

COPY ai_service ./ai_service
RUN mkdir -p /app/models /app/.numba_cache

ENV PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMBA_CACHE_DIR=/app/.numba_cache \
    PORT=8000

# Pre-compile Numba JIT kernels into /app/.numba_cache at build time
RUN python -c "import numpy as np, librosa; from ai_service.audio_processor import AudioProcessor; from ai_service.vad import VoiceActivityDetector; from ai_service.bafv_features import BiomechanicalFeatureExtractor; from ai_service.baseline_detector import BaselineAntiSpoofDetector; w = np.sin(2*np.pi*220*np.linspace(0, 1.0, 16000)).astype(np.float32); AudioProcessor().calculate_channel_metrics(w, 16000, 16000); librosa.resample(w[:8000], orig_sr=8000, target_sr=16000); VoiceActivityDetector().get_speech_ratio(w); BiomechanicalFeatureExtractor().compute_bafv_profile(w); BaselineAntiSpoofDetector().predict(w)"

EXPOSE 8000

CMD ["sh", "-c", "uvicorn ai_service.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
