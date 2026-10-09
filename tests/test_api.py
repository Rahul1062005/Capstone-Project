"""Integration and Unit Tests for BAFV-PCTA FastAPI AI Service (Phase 3 Exit Check)."""

import io
import pytest
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from ai_service.api import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_wav_bytes():
    bio = io.BytesIO()
    # 2.5 seconds of 16 kHz audio
    samples = 0.5 * np.sin(2 * np.pi * 350 * np.linspace(0, 2.5, int(16000 * 2.5))).astype(np.float32)
    sf.write(bio, samples, 16000, format="WAV")
    return bio.getvalue()


def test_health_check(client):
    """Verify /health endpoint returns active model metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "bafv-pcta-ai-service"
    assert "AASIST" in data["model_name"]
    assert data["target_sample_rate"] == 16000


def test_detect_file_success(client, sample_wav_bytes):
    """FR-01, FR-02, FR-03, FR-04, FR-05: Upload valid WAV, verify timeline & BAFV features."""
    files = {"file": ("test_call.wav", sample_wav_bytes, "audio/wav")}
    response = client.post("/api/v1/detect/file", files=files)
    assert response.status_code == 200
    data = response.json()

    assert data["file_name"] == "test_call.wav"
    assert "processing_latency_ms" in data
    assert "channel_metadata" in data
    assert "baseline_detection" in data
    assert "biomechanical_features" in data
    assert "windowed_timeline" in data

    # Verify Timeline Structure (FR-05: 2.0s windows, 0.5s hop)
    timeline_data = data["windowed_timeline"]
    assert timeline_data["total_windows"] > 0
    first_window = timeline_data["timeline"][0]
    assert first_window["start_time_seconds"] == 0.0
    assert first_window["end_time_seconds"] == 2.0
    assert "recommendation" in first_window
    assert "smoothed_risk" in first_window

    # Verify Biomechanical Acoustic Features (BAFV)
    bafv = data["biomechanical_features"]
    assert "pitch_f0" in bafv
    assert "formants" in bafv
    assert "voice_quality" in bafv


def test_detect_file_corrupt_rejection(client):
    """FR-01, FR-12: Reject corrupted/malformed audio safely with HTTP 422."""
    corrupted_data = b"CORRUPTED_NON_AUDIO_HEADER_12345678"
    files = {"file": ("corrupted.wav", corrupted_data, "audio/wav")}
    response = client.post("/api/v1/detect/file", files=files)
    assert response.status_code == 422
    data = response.json()
    assert "Failed to decode" in data["detail"] or "Format not recognised" in data["detail"]


def test_detect_window_payload(client):
    """Evaluate raw 2.0-second float32 window endpoint."""
    samples = (0.3 * np.sin(2 * np.pi * 220 * np.linspace(0, 2.0, 32000))).tolist()
    response = client.post("/api/v1/detect/window", json={"samples": samples})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "biomechanical_features" in data
