"""Unit Tests for Audio Processor (FR-01, FR-02, FR-12)."""

import io
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path

from ai_service.audio_processor import AudioProcessor, AudioProcessingError


@pytest.fixture
def processor():
    return AudioProcessor(target_sample_rate=16000)


@pytest.fixture
def sample_wav_bytes():
    bio = io.BytesIO()
    samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 16000)).astype(np.float32)
    sf.write(bio, samples, 16000, format="WAV")
    return bio.getvalue()


def test_reject_corrupted_audio(processor):
    """FR-01, FR-12: Reject invalid files safely, never fabricate predictions."""
    corrupted = b"NOT_A_VALID_AUDIO_HEADER_123456789"
    with pytest.raises(AudioProcessingError) as exc_info:
        processor.load_and_preprocess(corrupted)
    assert "Failed to decode" in str(exc_info.value) or "Format not recognised" in str(exc_info.value)


def test_reject_zero_byte_payload(processor):
    """FR-01: Reject empty payload."""
    with pytest.raises(AudioProcessingError) as exc_info:
        processor.load_and_preprocess(b"")
    assert "0-byte audio payload" in str(exc_info.value)


def test_valid_audio_preprocessing(processor, sample_wav_bytes):
    """FR-02: Convert valid audio to 16 kHz float32 mono."""
    waveform, meta = processor.load_and_preprocess(sample_wav_bytes)
    assert isinstance(waveform, np.ndarray)
    assert waveform.dtype == np.float32
    assert waveform.ndim == 1
    assert meta["target_sr"] == 16000
    assert meta["original_sr"] == 16000
    assert meta["duration_seconds"] == 1.0
    assert 0.0 <= meta["channel_confidence"] <= 1.0


def test_detect_8khz_band_limiting(processor):
    """SC3: Detect 8 kHz telephone band-limiting."""
    # Generate 8 kHz sampled audio
    bio = io.BytesIO()
    samples = np.sin(2 * np.pi * 300 * np.linspace(0, 1.0, 8000)).astype(np.float32)
    sf.write(bio, samples, 8000, format="WAV")
    
    waveform, meta = processor.load_and_preprocess(bio.getvalue())
    assert meta["original_sr"] == 8000
    assert meta["is_band_limited_8khz"] is True
    assert any("Telephone 8 kHz" in w for w in meta["quality_warnings"])
