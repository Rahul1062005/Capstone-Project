"""Synthetic and Biomechanical Benchmark Audio Generator for Testing & Validation.

Generates realistic audio samples for Phase 1 indexing and Phase 2 baseline model verification:
1. Genuine biological speech acoustics: Dynamic F0 glide, vocal tract formant resonances (F1, F2, F3),
   micro-jitter (pitch perturbation), micro-shimmer (amplitude perturbation), and realistic aspiration noise.
2. Synthetic deepfake voice characteristics: Constant pitch lock, phase discontinuity boundaries,
   spectrogram inversion artifacts, and unnatural harmonic flatness typical of neural vocoders.
3. Telephone channel degradation: 8 kHz band-limiting, high-frequency cutoff, background line hum.
4. Corrupted file fixture: Non-RIFF payload to verify FR-01 & FR-12 rejection mechanisms.
"""

from __future__ import annotations

import math
import os
import wave
from pathlib import Path
import numpy as np


def generate_genuine_speech_waveform(
    duration: float = 3.0,
    sr: int = 16000,
    base_f0: float = 140.0,
) -> np.ndarray:
    """Simulate biological speech with glottal flow pulses, formant filtering, jitter and shimmer."""
    total_samples = int(duration * sr)
    t = np.linspace(0, duration, total_samples, endpoint=False)

    # 1. Biomechanical Pitch contour (F0 with natural intonation curve and micro-jitter)
    f0_curve = base_f0 + 25.0 * np.sin(2 * np.pi * 0.8 * t) + 12.0 * np.sin(2 * np.pi * 1.7 * t)
    jitter = np.random.normal(0, 1.2, total_samples)  # 0.8-1.5% micro-jitter
    instantaneous_f0 = np.clip(f0_curve + jitter, 80.0, 300.0)

    # Glottal pulse phase accumulation
    phase = np.cumsum(2 * np.pi * instantaneous_f0 / sr)
    glottal_source = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)

    # 2. Shimmer (Cycle-to-cycle amplitude perturbation)
    shimmer = 1.0 + 0.04 * np.sin(2 * np.pi * 4.2 * t) + np.random.normal(0, 0.015, total_samples)
    glottal_source = glottal_source * shimmer

    # 3. Coupled Formants (Vocal tract transfer function simulation: F1 ~ 700Hz, F2 ~ 1220Hz, F3 ~ 2600Hz)
    f1, f2, f3 = 700.0, 1220.0, 2600.0
    formant_res = (
        0.6 * np.sin(2 * np.pi * f1 * t + phase * 0.1)
        + 0.35 * np.sin(2 * np.pi * f2 * t + phase * 0.05)
        + 0.15 * np.sin(2 * np.pi * f3 * t)
    )

    # Natural unvoiced frication / aspiration noise in vocal tract
    aspiration = np.random.normal(0, 0.02, total_samples)

    # Combined human acoustic waveform
    waveform = (glottal_source * 0.7 + formant_res * 0.25 + aspiration)
    
    # Syllabic envelope (simulating human speaking pauses and breath bursts)
    envelope = 0.5 * (1.0 - np.cos(2 * np.pi * 1.5 * t)) * (0.8 + 0.2 * np.sin(2 * np.pi * 3.0 * t))
    waveform = waveform * np.clip(envelope, 0.0, 1.0)

    # Normalization
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val * 0.85
    return waveform.astype(np.float32)


def generate_synthetic_clone_waveform(
    duration: float = 3.0,
    sr: int = 16000,
    carrier_freq: float = 145.0,
) -> np.ndarray:
    """Simulate neural vocoder artifacts: phase discontinuities, lack of micro-jitter, spectral flatness."""
    total_samples = int(duration * sr)
    t = np.linspace(0, duration, total_samples, endpoint=False)

    # 1. Unnatural constant pitch (robotic F0, virtually zero biological jitter)
    f0 = carrier_freq  # perfectly flat pitch
    phase = 2 * np.pi * f0 * t

    # Invertible spectrogram synthesis artifacts: unnatural harmonic sharpness
    waveform = np.zeros(total_samples, dtype=np.float32)
    for harmonic in range(1, 16):
        amp = 1.0 / (harmonic ** 0.8)  # unnatural harmonic decay
        waveform += amp * np.sin(harmonic * phase)

    # 2. Phase discontinuity hops every 256 samples (frame-boundary phase error typical of early neural vocoders)
    hop_size = 256
    for i in range(0, total_samples, hop_size):
        end_idx = min(i + hop_size, total_samples)
        phase_offset = (i // hop_size) * 0.785  # abrupt phase shift
        waveform[i:end_idx] *= np.cos(phase_offset)

    # 3. Add high-frequency neural vocoder buzz / quantization noise
    buzz = 0.06 * np.sin(2 * np.pi * 4200.0 * t)
    waveform = waveform + buzz

    # Syllabic envelope
    envelope = 0.5 * (1.0 - np.cos(2 * np.pi * 1.5 * t))
    waveform = waveform * envelope

    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val * 0.85
    return waveform.astype(np.float32)


def apply_telephone_degradation(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Simulate 8 kHz G.711 / AMR band-limiting and telephone line noise."""
    import scipy.signal as signal

    # 8 kHz downsampling simulation (Nyquist 4000 Hz, telephone bandpass 300Hz - 3400Hz)
    sos = signal.butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos")
    filtered = signal.sosfilt(sos, waveform)

    # Line hum (50 Hz / 60 Hz hum + noise)
    t = np.linspace(0, len(waveform) / sr, len(waveform), endpoint=False)
    hum = 0.015 * np.sin(2 * np.pi * 50.0 * t)
    noise = np.random.normal(0, 0.01, len(waveform))

    degraded = filtered + hum + noise
    max_val = np.max(np.abs(degraded))
    if max_val > 0:
        degraded = degraded / max_val * 0.8
    return degraded.astype(np.float32)


def save_wav(waveform: np.ndarray, file_path: Path, sr: int = 16000) -> None:
    """Save float32 array as 16-bit PCM WAV."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    int16_samples = (waveform * 32767.0).astype(np.int16)
    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16_samples.tobytes())


def main() -> None:
    out_dir = Path("data/test_samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating benchmark audio dataset for BAFV-PCTA...")

    # 1. Genuine human biological speech samples
    genuine_1 = generate_genuine_speech_waveform(duration=3.5, base_f0=135.0)
    save_wav(genuine_1, out_dir / "genuine_human_speaker_a.wav")

    genuine_2 = generate_genuine_speech_waveform(duration=2.5, base_f0=195.0)
    save_wav(genuine_2, out_dir / "genuine_human_speaker_b.wav")

    # 2. Synthetic deepfake / cloned voice samples
    clone_1 = generate_synthetic_clone_waveform(duration=3.5, carrier_freq=140.0)
    save_wav(clone_1, out_dir / "synthetic_voice_clone_v1.wav")

    clone_2 = generate_synthetic_clone_waveform(duration=3.0, carrier_freq=180.0)
    save_wav(clone_2, out_dir / "synthetic_tts_neural_v2.wav")

    # 3. Degraded telephone sample (human speech passed through 8kHz codec)
    tel_sample = apply_telephone_degradation(genuine_1)
    save_wav(tel_sample, out_dir / "genuine_telephone_8khz_amr.wav")

    # 4. Corrupted non-audio file to test FR-01 rejection
    corrupted_path = out_dir / "corrupted_payload.wav"
    with open(corrupted_path, "wb") as f:
        f.write(b"NOT_A_REAL_WAV_HEADER_RANDOM_DATA_1234567890")

    print(f"Benchmark audio files successfully generated in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
