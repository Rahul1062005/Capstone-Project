"""Tests for Phase 7 Cross-Dataset, Telephony Guard Ablation & LOGO Evaluation."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.evaluate_phase7 import compute_min_tdcf, run_phase7_evaluation
from scripts.train_detector import collect_folder_samples


def _write_wav(path: Path, freq_hz: float, duration_sec: float = 1.2, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, duration_sec, int(sr * duration_sec), endpoint=False)
    wave = (0.4 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)
    sf.write(path, wave, sr)


def test_min_tdcf_bounds() -> None:
    """min-tDCF should be near 0 for perfectly separable scores and bounded in [0, 1]."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    perfect_scores = np.array([0.05, 0.1, 0.15, 0.85, 0.9, 0.95])
    tdcf = compute_min_tdcf(y_true, perfect_scores)
    assert 0.0 <= tdcf <= 0.1


def test_phase7_evaluation_pipeline_and_reports(tmp_path: Path) -> None:
    """Run full Phase 7 clean + 8 kHz + LOGO evaluation on synthetic temp fixtures."""
    ds_dir = tmp_path / "eval_ds"
    spoof_root = tmp_path / "spoof_by_gen"
    for i in range(3):
        _write_wav(ds_dir / "bonafide" / f"h_{i}.wav", 170.0 + i * 15.0)
        _write_wav(ds_dir / "spoof" / f"s_{i}.wav", 520.0 + i * 25.0)
        _write_wav(spoof_root / "GenA" / f"ga_{i}.wav", 510.0 + i * 10.0)
        _write_wav(spoof_root / "GenB" / f"gb_{i}.wav", 660.0 + i * 15.0)

    samples = collect_folder_samples(ds_dir)
    json_out = tmp_path / "phase7_eval.json"
    report_out = tmp_path / "PHASE_7_EVALUATION_REPORT.md"

    results = run_phase7_evaluation(
        samples=samples,
        checkpoint_path=Path("__none__"),
        calibration_path=Path("__none__"),
        spoof_root=spoof_root,
        json_out=json_out,
        report_out=report_out,
    )

    assert json_out.exists()
    assert report_out.exists()
    assert results["clean_wideband"]["samples_evaluated"] == 6
    assert results["telephony_8khz"]["samples_evaluated"] == 6
    assert results["logo_open_set"]["evaluated"] is True
    assert len(results["logo_open_set"]["folds"]) == 2

    parsed_json = json.loads(json_out.read_text(encoding="utf-8"))
    assert "clean_wideband" in parsed_json
    md_text = report_out.read_text(encoding="utf-8")
    assert "Clean 16 kHz Wideband" in md_text
    assert "Leave-One-Generator-Out" in md_text
