"""Unit and Smoke Tests for the Model Training & Manifold Calibration Pipeline."""

from pathlib import Path

import numpy as np
import soundfile as sf

from ai_service.bafv_features import BiomechanicalFeatureExtractor
from ai_service.baseline_detector import BaselineAntiSpoofDetector
from ai_service.open_set_detector import OpenSetGeneratorDetector
from scripts.calibrate_manifold import calibrate_from_directories
from scripts.train_detector import (
    collect_asvspoof_samples,
    collect_folder_samples,
    compute_eer,
    train_pipeline,
)


def _write_tone_wav(path: Path, freq_hz: float, duration_sec: float = 1.5, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, duration_sec, int(sr * duration_sec), endpoint=False)
    wave = (0.4 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)
    sf.write(path, wave, sr)


def test_folder_and_asvspoof_collection(tmp_path: Path) -> None:
    """Verify folder and ASVspoof protocol dataset discovery."""
    bonafide_wav = tmp_path / "dataset" / "bonafide" / "h1.wav"
    spoof_wav = tmp_path / "dataset" / "spoof" / "s1.wav"
    _write_tone_wav(bonafide_wav, 220.0)
    _write_tone_wav(spoof_wav, 440.0)

    folder_samples = collect_folder_samples(tmp_path / "dataset")
    assert len(folder_samples) == 2
    assert {lbl for _, lbl in folder_samples} == {0, 1}

    audio_root = tmp_path / "asv_audio"
    _write_tone_wav(audio_root / "LA_0001.wav", 200.0)
    _write_tone_wav(audio_root / "LA_0002.wav", 500.0)
    proto = tmp_path / "protocol.txt"
    proto.write_text("SPK1 LA_0001 - - bonafide\nSPK2 LA_0002 - A01 spoof\n", encoding="utf-8")

    asv_samples = collect_asvspoof_samples(proto, audio_root)
    assert len(asv_samples) == 2
    assert asv_samples[0][1] == 0
    assert asv_samples[1][1] == 1


def test_compute_eer_separable() -> None:
    """EER should be 0.0 and AUC 1.0 when spoof scores cleanly separate classes."""
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    eer, auc = compute_eer(y_true, scores)
    assert eer == 0.0
    assert auc == 1.0


def test_end_to_end_training_and_checkpoint_loading(tmp_path: Path) -> None:
    """Smoke-test 2-epoch PyTorch training, checkpoint export, and detector auto-loading."""
    data_dir = tmp_path / "train_ds"
    for idx in range(4):
        _write_tone_wav(data_dir / "bonafide" / f"real_{idx}.wav", 180.0 + idx * 10.0, duration_sec=1.5)
        _write_tone_wav(data_dir / "spoof" / f"fake_{idx}.wav", 620.0 + idx * 15.0, duration_sec=1.5)

    samples = collect_folder_samples(data_dir)
    ckpt_path = tmp_path / "models" / "test_baseline.pt"
    metrics_path = tmp_path / "models" / "test_metrics.json"

    summary = train_pipeline(
        train_samples=samples,
        output_checkpoint=ckpt_path,
        metrics_out=metrics_path,
        epochs=2,
        batch_size=2,
        clip_seconds=1.5,
        telephony_aug_prob=0.5,
    )

    assert ckpt_path.exists()
    assert metrics_path.exists()
    assert len(summary["history"]) == 2

    loaded_detector = BaselineAntiSpoofDetector(checkpoint_path=ckpt_path)
    assert loaded_detector.weights_source == "trained_checkpoint"
    assert loaded_detector.checkpoint_path == str(ckpt_path)

    pred = loaded_detector.predict(data_dir / "bonafide" / "real_0.wav")
    assert pred["model_metadata"]["weights_source"] == "trained_checkpoint"
    assert pred["model_metadata"]["checkpoint_loaded"] is True


def test_manifold_calibration_and_autoloading(tmp_path: Path) -> None:
    """Verify calibration_stats.json generation and loading into BAFV and OpenSet modules."""
    bonafide_dir = tmp_path / "calib_bonafide"
    spoof_root = tmp_path / "calib_spoof"
    for idx in range(3):
        _write_tone_wav(bonafide_dir / f"human_{idx}.wav", 190.0 + idx * 12.0, duration_sec=1.5)
        _write_tone_wav(spoof_root / "MockVocoder" / f"gen_{idx}.wav", 520.0 + idx * 20.0, duration_sec=1.5)

    cal_path = tmp_path / "calibration_stats.json"
    stats = calibrate_from_directories(
        bonafide_dir=bonafide_dir,
        spoof_root=spoof_root,
        output_path=cal_path,
    )

    assert cal_path.exists()
    assert stats["bonafide_sample_count"] == 3
    assert len(stats["human_coupling_mean"]) == 6
    assert len(stats["human_manifold_centroid"]) == 5
    assert "MockVocoder" in stats["known_generators"]

    extractor = BiomechanicalFeatureExtractor(calibration_path=cal_path)
    assert extractor.calibration_loaded is True
    assert np.allclose(extractor.human_coupling_mean, np.array(stats["human_coupling_mean"]))

    ogd = OpenSetGeneratorDetector(calibration_path=cal_path)
    assert ogd.calibration_loaded is True
    assert "MockVocoder" in ogd.known_generators
