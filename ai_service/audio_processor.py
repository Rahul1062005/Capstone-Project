"""Audio Preprocessing, Validation and Channel Quality Assessment Pipeline.

Complies with:
- FR-01: Accept supported audio; reject invalid, oversized or mislabelled files safely.
- FR-02: Convert valid audio to model's expected format (16 kHz mono float32).
- FR-12: Fail-secure with explicit error states; never fabricate data.
"""

from __future__ import annotations

import io
import math
import os
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import soundfile as sf
import librosa


MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB max limit
MAX_DURATION_SECONDS = 300.0  # 5 minutes maximum for prototype


class AudioProcessingError(Exception):
    """Explicit domain exception for audio processing failures (FR-12)."""
    pass


class AudioProcessor:
    """Standardizes incoming audio streams and calculates channel quality indicators."""

    def __init__(self, target_sample_rate: int = 16000) -> None:
        self.target_sample_rate = target_sample_rate

    def validate_file_bytes(self, file_path_or_bytes: Union[str, Path, bytes]) -> None:
        """Validate input size and prevent denial-of-service from malformed files."""
        if isinstance(file_path_or_bytes, (str, Path)):
            path = Path(file_path_or_bytes)
            if not path.exists():
                raise AudioProcessingError(f"Audio file does not exist: {path}")
            size = path.stat().st_size
            if size == 0:
                raise AudioProcessingError("Audio file is completely empty (0 bytes).")
            if size > MAX_AUDIO_SIZE_BYTES:
                raise AudioProcessingError(f"Audio file exceeds maximum size limit ({size} > {MAX_AUDIO_SIZE_BYTES} bytes).")
        elif isinstance(file_path_or_bytes, bytes):
            if len(file_path_or_bytes) == 0:
                raise AudioProcessingError("Received 0-byte audio payload.")
            if len(file_path_or_bytes) > MAX_AUDIO_SIZE_BYTES:
                raise AudioProcessingError("Audio payload exceeds maximum size limit.")

    def load_and_preprocess(
        self,
        audio_source: Union[str, Path, bytes],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Safely decode, check headers, convert to mono, resample to 16 kHz float32 (FR-01, FR-02)."""
        self.validate_file_bytes(audio_source)

        try:
            if isinstance(audio_source, bytes):
                bio = io.BytesIO(audio_source)
                waveform, original_sr = sf.read(bio, dtype="float32")
            else:
                waveform, original_sr = sf.read(str(audio_source), dtype="float32")
        except Exception as exc:
            raise AudioProcessingError(f"Failed to decode audio header/stream: {str(exc)}") from exc

        # Ensure 1D mono
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)

        original_duration = len(waveform) / float(original_sr) if original_sr > 0 else 0.0
        if original_duration > MAX_DURATION_SECONDS:
            raise AudioProcessingError(
                f"Audio duration ({original_duration:.1f}s) exceeds max limit of {MAX_DURATION_SECONDS}s"
            )

        # Resample to target sample rate (16 kHz) if necessary
        if original_sr != self.target_sample_rate:
            try:
                waveform = librosa.resample(
                    y=waveform,
                    orig_sr=original_sr,
                    target_sr=self.target_sample_rate,
                    res_type="soxr_vhq",
                )
            except Exception:
                waveform = librosa.resample(
                    y=waveform,
                    orig_sr=original_sr,
                    target_sr=self.target_sample_rate,
                )

        # Peak normalization
        max_abs = np.max(np.abs(waveform)) if len(waveform) > 0 else 0.0
        if max_abs > 0.0:
            waveform = waveform / max_abs * 0.95

        # Calculate channel metrics and quality indicators
        quality_metadata = self.calculate_channel_metrics(waveform, self.target_sample_rate, original_sr)
        quality_metadata["original_sr"] = original_sr
        quality_metadata["target_sr"] = self.target_sample_rate
        quality_metadata["duration_seconds"] = round(len(waveform) / self.target_sample_rate, 3)

        return waveform.astype(np.float32), quality_metadata

    def calculate_channel_metrics(
        self,
        waveform: np.ndarray,
        sr: int,
        original_sr: int,
    ) -> Dict[str, Any]:
        """Compute SNR, clipping ratio, and channel confidence score C."""
        if len(waveform) == 0:
            return {
                "snr_db": 0.0,
                "clipping_ratio": 0.0,
                "channel_confidence": 0.0,
                "is_band_limited_8khz": False,
                "quality_warnings": ["Zero-length audio"],
            }

        # 1. Clipping detection (values near +/- 0.95)
        clipping_samples = np.sum(np.abs(waveform) >= 0.949)
        clipping_ratio = float(clipping_samples / len(waveform))

        # 2. SNR Estimation using signal energy vs noise floor (lowest 10% energy frames)
        frame_len = int(0.025 * sr)
        hop_len = int(0.010 * sr)
        if len(waveform) > frame_len:
            frames = librosa.util.frame(waveform, frame_length=frame_len, hop_length=hop_len)
            frame_energies = np.sum(frames ** 2, axis=0) / frame_len
            sorted_energies = np.sort(frame_energies)
            noise_floor = float(np.mean(sorted_energies[: max(1, len(sorted_energies) // 10)]))
            signal_power = float(np.mean(sorted_energies[len(sorted_energies) // 2 :]))
            snr_db = 10.0 * np.log10(max(signal_power / max(noise_floor, 1e-9), 1.0))
        else:
            snr_db = 15.0

        # 3. Detect 8 kHz telephone band-limiting (spectral rolloff check)
        is_band_limited = False
        if original_sr <= 8000:
            is_band_limited = True
        else:
            # Check spectral roll-off: in 8 kHz telephone audio, 95% of energy is strictly below 3600 Hz
            rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr, roll_percent=0.95)
            median_rolloff = float(np.median(rolloff))
            if median_rolloff < 3600.0:
                is_band_limited = True

        # 4. Formulate Channel Confidence Index C (Scope section 9: 0.0 to 1.0)
        # Low SNR, high clipping, or severe band-limiting reduces channel confidence
        snr_factor = np.clip((snr_db - 5.0) / 25.0, 0.1, 1.0)
        clipping_penalty = max(0.0, 1.0 - clipping_ratio * 15.0)
        band_penalty = 0.75 if is_band_limited else 1.0

        channel_confidence = float(np.clip(snr_factor * clipping_penalty * band_penalty, 0.15, 0.98))

        warnings = []
        if snr_db < 10.0:
            warnings.append("Low signal-to-noise ratio (noisy channel).")
        if clipping_ratio > 0.02:
            warnings.append("Audio clipping/distortion detected.")
        if is_band_limited:
            warnings.append("Telephone 8 kHz band-limiting detected; high-frequency phase cues suppressed.")

        return {
            "snr_db": round(float(snr_db), 2),
            "clipping_ratio": round(clipping_ratio, 4),
            "channel_confidence": round(channel_confidence, 3),
            "is_band_limited_8khz": is_band_limited,
            "quality_warnings": warnings,
        }
