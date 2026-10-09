"""Voice Activity Detection (VAD) Module for BAFV-PCTA.

Extracts speech activity segments from continuous audio streams.
Ensures sliding window inference only computes predictions on active speech frames,
ignoring prolonged silences and background noise.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import librosa


class VoiceActivityDetector:
    """Robust Energy and Spectral Flux Voice Activity Detector."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        energy_threshold_db: float = -38.0,
        min_speech_duration_ms: int = 100,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_length = int(sample_rate * (frame_ms / 1000.0))
        self.hop_length = int(self.frame_length // 2)
        self.energy_threshold_db = energy_threshold_db
        self.min_speech_frames = int(min_speech_duration_ms / (frame_ms / 2))

    def detect_speech_frames(self, waveform: np.ndarray) -> np.ndarray:
        """Return boolean mask indicating speech presence per hop frame."""
        if len(waveform) < self.frame_length:
            return np.zeros(1, dtype=bool)

        # 1. Compute Short-Time RMS energy in dB
        rms = librosa.feature.rms(
            y=waveform,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max(rms) if np.max(rms) > 0 else 1.0)

        # 2. Compute Spectral Flatness to reject pure stationary noise
        flatness = librosa.feature.spectral_flatness(
            y=waveform,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
        )[0]

        # Speech frames typically have higher energy and moderate-to-low spectral flatness (harmonics)
        is_speech = (rms_db > self.energy_threshold_db) & (flatness < 0.65)

        # Morphological smoothing (close small gaps of silence < 100ms)
        smoothed = is_speech.copy()
        for i in range(1, len(smoothed) - 1):
            if is_speech[i - 1] and is_speech[i + 1]:
                smoothed[i] = True

        return smoothed

    def get_speech_ratio(self, waveform: np.ndarray) -> float:
        """Calculate percentage of active speech in the given waveform segment."""
        speech_mask = self.detect_speech_frames(waveform)
        if len(speech_mask) == 0:
            return 0.0
        return float(np.sum(speech_mask) / len(speech_mask))

    def get_speech_intervals(self, waveform: np.ndarray) -> List[Tuple[float, float]]:
        """Return list of (start_time_sec, end_time_sec) active speech timestamps."""
        speech_mask = self.detect_speech_frames(waveform)
        intervals: List[Tuple[float, float]] = []

        in_speech = False
        start_idx = 0

        frame_duration = self.hop_length / float(self.sample_rate)

        for i, val in enumerate(speech_mask):
            if val and not in_speech:
                in_speech = True
                start_idx = i
            elif not val and in_speech:
                in_speech = False
                intervals.append((round(start_idx * frame_duration, 3), round(i * frame_duration, 3)))

        if in_speech:
            intervals.append((round(start_idx * frame_duration, 3), round(len(speech_mask) * frame_duration, 3)))

        return intervals
