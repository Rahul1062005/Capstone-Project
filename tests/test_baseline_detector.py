"""Unit Tests for Baseline Anti-Spoofing Detector (FR-03, FR-04, FR-12, SC1, SC6)."""

import pytest
import numpy as np
from pathlib import Path

from ai_service.baseline_detector import BaselineAntiSpoofDetector
from ai_service.audio_processor import AudioProcessingError


@pytest.fixture
def detector():
    return BaselineAntiSpoofDetector()


def test_baseline_detector_output_contract(detector):
    """FR-03, FR-04: Returns real model output with metadata, latency, logits and warnings."""
    # Synthetic 2.0-second 16 kHz waveform
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    synthetic_wave = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    result = detector.predict(synthetic_wave)

    # 1. Model metadata verification (FR-03)
    meta = result["model_metadata"]
    assert meta["model_name"] == "AASIST-ResNet-Baseline"
    assert meta["model_version"] == "v1.2.0-baseline"
    assert meta["is_hardcoded"] is False
    assert meta["target_sample_rate"] == 16000

    # 2. Latency measurement (SC6)
    metrics = result["inference_metrics"]
    assert metrics["inference_latency_ms"] > 0.0
    assert metrics["audio_duration_seconds"] == 2.0

    # 3. Probabilistic score contract (FR-04)
    scores = result["scores"]
    assert len(scores["raw_logits"]) == 2
    assert 0.0 <= scores["spoof_probability"] <= 1.0
    assert 0.0 <= scores["bonafide_probability"] <= 1.0
    assert scores["score_interpretation"] in ["GENUINE_HUMAN", "AMBIGUOUS_SUSPICIOUS", "SYNTHETIC_DEEPFAKE", "POOR_CHANNEL_VERIFY"]
    assert scores["recommended_action"] in ["ALLOW", "VERIFY", "HOLD_FOR_REVIEW"]

    # 4. Disclaimer statement
    assert "Notice: Prediction is probabilistic" in result["disclaimer"]


def test_reject_insufficient_audio_duration(detector):
    """FR-12: Reject short audio (< 0.5s) safely with explicit error."""
    short_audio = np.zeros(2000, dtype=np.float32)  # 0.125s at 16kHz
    with pytest.raises(AudioProcessingError) as exc_info:
        detector.predict(short_audio)
    assert "too short" in str(exc_info.value)
