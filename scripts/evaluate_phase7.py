"""Phase 7 Cross-Dataset, Telephony Ablation & Open-Set LOGO Evaluation Suite.

Evaluates:
1. Clean Wideband Performance (RQ1): EER, min-tDCF, ROC-AUC, Accuracy, FRR, FAR, Avg Latency.
2. Telephony 8 kHz Degradation & Low-Confidence Guard Ablation (RQ3):
   Compares raw neural detector false-HOLD rate on degraded human calls vs. DRFE Guard.
3. Leave-One-Generator-Out (LOGO) Open-Set Detection (RQ2):
   When --spoof-root has generator subfolders, holds out each generator one at a time
   and measures zero-day synthetic catch rate vs. bonafide rejection rate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_service.audio_processor import AudioProcessingError, AudioProcessor
from ai_service.bafv_features import BiomechanicalFeatureExtractor
from ai_service.baseline_detector import BaselineAntiSpoofDetector
from ai_service.open_set_detector import OpenSetGeneratorDetector, TelephonyChannelConfidenceSystem
from ai_service.risk_engine import AudioEvidenceContext, RiskEngine, TransactionContext
from scripts.train_detector import (
    apply_telephony_augmentation,
    collect_asvspoof_samples,
    collect_folder_samples,
    compute_eer,
)


def compute_min_tdcf(
    y_true: np.ndarray,
    spoof_scores: np.ndarray,
    p_tar: float = 0.9405,
    p_non: float = 0.0095,
    p_spoof: float = 0.05,
    c_miss_cm: float = 1.0,
    c_fa_cm: float = 10.0,
    c_miss_asv: float = 1.0,
    c_fa_asv: float = 10.0,
    p_miss_asv: float = 0.05,
    p_fa_asv: float = 0.01,
    p_fa_spoof_asv: float = 0.20,
) -> float:
    """Compute normalized ASVspoof-style minimum Tandem Detection Cost Function (min-tDCF)."""
    if len(np.unique(y_true)) < 2:
        return 1.0

    # Convention: y_true=1 is spoof, y_true=0 is bonafide; spoof_scores higher => more spoof.
    # For CM ROC: miss = bonafide rejected as spoof (FPR on bonafide=0), fa = spoof accepted (1 - TPR).
    fpr_bonafide, tpr_spoof, _ = roc_curve(y_true, spoof_scores, pos_label=1)
    p_miss_cm = fpr_bonafide  # bonafide falsely flagged as spoof
    p_fa_cm = 1.0 - tpr_spoof  # spoof falsely accepted as bonafide

    c1 = p_tar * (c_miss_cm - c_miss_asv * p_miss_asv) - p_non * c_fa_asv * p_fa_asv
    c2 = c_fa_cm * p_spoof * (1.0 - p_miss_asv)
    c0 = (
        p_tar * c_miss_asv * p_miss_asv
        + p_non * c_fa_asv * p_fa_asv
        + p_spoof * c_fa_asv * p_fa_spoof_asv
    )

    denom = max(c0 + min(c1, c2), 1e-9)
    tdcf_norm = (c1 * p_miss_cm + c2 * p_fa_cm) / denom
    min_val = float(np.min(tdcf_norm))
    return float(np.clip(min_val, 0.0, 1.0))


def evaluate_condition(
    samples: Sequence[Tuple[Path, int]],
    detector: BaselineAntiSpoofDetector,
    bafv: BiomechanicalFeatureExtractor,
    risk_engine: RiskEngine,
    telephony_degrade: bool = False,
    seed: int = 2026,
    max_samples: int = 0,
) -> Dict[str, Any]:
    """Evaluate detector + TCS + DRFE across clean or 8 kHz telephony-degraded audio."""
    processor = AudioProcessor(target_sample_rate=16000)
    rng = np.random.default_rng(seed)

    items = list(samples)[:max_samples] if max_samples > 0 else list(samples)
    y_true: List[int] = []
    spoof_scores: List[float] = []
    latencies_ms: List[float] = []

    bonafide_raw_holds = 0
    bonafide_drfe_holds = 0
    bonafide_guard_triggered = 0
    bonafide_total = 0

    for path, label in items:
        try:
            waveform, meta = processor.load_and_preprocess(path)
            if len(waveform) < int(0.5 * 16000):
                continue

            if telephony_degrade:
                waveform = apply_telephony_augmentation(waveform, 16000, rng)
                meta = processor.calculate_channel_metrics(waveform, 16000, 8000)
                meta["is_band_limited_8khz"] = True

            t0 = time.perf_counter()
            pred = detector.predict(waveform)
            profile = bafv.compute_bafv_profile(waveform)
            tcs = TelephonyChannelConfidenceSystem.compute_tcs(
                snr_db=meta["snr_db"],
                is_band_limited_8khz=bool(meta["is_band_limited_8khz"]),
            )
            decision = risk_engine.evaluate_risk(
                audio=AudioEvidenceContext(
                    voice_spoof_prob=pred["scores"]["spoof_probability"],
                    bafv_pcta_score=profile["bafv_anomaly_score"],
                    sustained_stability_risk=0.2,
                    channel_confidence=tcs["channel_confidence"],
                ),
                txn=TransactionContext(
                    transaction_id="EVAL-TXN",
                    amount_inr=45000.0,
                    is_new_beneficiary=False,
                    transfer_velocity_per_hour=1,
                ),
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            spoof_prob = float(pred["scores"]["spoof_probability"])
            y_true.append(label)
            spoof_scores.append(spoof_prob)

            if label == 0:
                bonafide_total += 1
                if pred["scores"]["recommended_action"] == "HOLD_FOR_REVIEW":
                    bonafide_raw_holds += 1
                if decision.recommended_action == "HOLD_FOR_REVIEW":
                    bonafide_drfe_holds += 1
                if decision.low_confidence_guard_active:
                    bonafide_guard_triggered += 1
        except AudioProcessingError:
            continue

    y_arr = np.array(y_true, dtype=np.int64)
    s_arr = np.array(spoof_scores, dtype=np.float64)
    eer, auc = compute_eer(y_arr, s_arr)
    min_tdcf = compute_min_tdcf(y_arr, s_arr)

    preds = (s_arr >= 0.5).astype(np.int64)
    acc = float(np.mean(preds == y_arr)) if len(y_arr) > 0 else 0.0
    bonafide_mask = y_arr == 0
    spoof_mask = y_arr == 1
    frr = float(np.mean(preds[bonafide_mask] == 1)) if np.any(bonafide_mask) else 0.0
    far = float(np.mean(preds[spoof_mask] == 0)) if np.any(spoof_mask) else 0.0

    return {
        "samples_evaluated": len(y_arr),
        "bonafide_count": int(np.sum(bonafide_mask)),
        "spoof_count": int(np.sum(spoof_mask)),
        "eer": round(eer, 4),
        "min_tdcf": round(min_tdcf, 4),
        "roc_auc": round(auc, 4),
        "accuracy_at_0_5": round(acc, 4),
        "frr_at_0_5": round(frr, 4),
        "far_at_0_5": round(far, 4),
        "avg_latency_ms": round(float(np.mean(latencies_ms)) if latencies_ms else 0.0, 2),
        "bonafide_raw_hold_rate": round(bonafide_raw_holds / max(1, bonafide_total), 4),
        "bonafide_drfe_hold_rate": round(bonafide_drfe_holds / max(1, bonafide_total), 4),
        "low_confidence_guard_activations": bonafide_guard_triggered,
    }


def evaluate_logo_open_set(
    bonafide_samples: Sequence[Tuple[Path, int]],
    spoof_root: Path,
    detector: BaselineAntiSpoofDetector,
    bafv: BiomechanicalFeatureExtractor,
    max_per_generator: int = 100,
) -> Dict[str, Any]:
    """Leave-One-Generator-Out (LOGO) Open-Set evaluation across generator subdirectories."""
    if not spoof_root.exists():
        return {"evaluated": False, "reason": "No spoof_root directory provided."}

    processor = AudioProcessor(target_sample_rate=16000)
    gen_dirs = sorted(p for p in spoof_root.iterdir() if p.is_dir())
    if len(gen_dirs) < 2:
        return {"evaluated": False, "reason": "Need at least 2 generator subfolders for LOGO."}

    def _extract_obs(audio_paths: Sequence[Path]) -> List[np.ndarray]:
        vectors: List[np.ndarray] = []
        for p in audio_paths[:max_per_generator]:
            try:
                wave, _ = processor.load_and_preprocess(p)
                if len(wave) < int(0.5 * 16000):
                    continue
                pred = detector.predict(wave)
                prof = bafv.compute_bafv_profile(wave)
                obs = OpenSetGeneratorDetector.build_observation_vector(
                    spoof_prob=pred["scores"]["spoof_probability"],
                    pcvs_score=prof["bafv_anomaly_score"],
                    mahalanobis_dist=prof["pcta_engine"]["mahalanobis_distance"],
                    jitter_percent=prof["pitch_f0"]["jitter_percent"],
                    shimmer_percent=prof["voice_quality"]["shimmer_percent"],
                )
                vectors.append(obs)
            except AudioProcessingError:
                continue
        return vectors

    # Fit human centroid from bonafide samples
    human_obs = _extract_obs([p for p, _ in bonafide_samples])
    if not human_obs:
        return {"evaluated": False, "reason": "No valid bonafide audio for human centroid."}
    human_mat = np.vstack(human_obs)
    human_centroid = np.mean(human_mat, axis=0)

    # Extract per-generator observation vectors
    gen_vectors: Dict[str, List[np.ndarray]] = {}
    for g_dir in gen_dirs:
        files = [f for f in sorted(g_dir.rglob("*")) if f.is_file() and f.suffix.lower() in {".wav", ".flac", ".ogg", ".mp3"}]
        vecs = _extract_obs(files)
        if vecs:
            gen_vectors[g_dir.name] = vecs

    if len(gen_vectors) < 2:
        return {"evaluated": False, "reason": "Fewer than 2 generator folders contained valid audio."}

    per_fold: List[Dict[str, Any]] = []
    for held_out_gen, held_vecs in gen_vectors.items():
        ogd = OpenSetGeneratorDetector(calibration_path="__none__")
        ogd.human_manifold_centroid = human_centroid
        ogd.known_generators = {}
        for train_gen, train_vecs in gen_vectors.items():
            if train_gen == held_out_gen:
                continue
            mat = np.vstack(train_vecs)
            cent = np.mean(mat, axis=0)
            dists = np.linalg.norm(mat - cent, axis=1)
            ogd.known_generators[train_gen] = {
                "family": train_gen,
                "centroid": cent,
                "max_cluster_radius": float(max(0.25, np.percentile(dists, 95))),
                "description": f"LOGO calibrated {train_gen}",
            }

        caught_synthetic = 0
        flagged_zero_day = 0
        for obs in held_vecs:
            # Reconstruct inputs into evaluate_open_set cleanly
            res = ogd.evaluate_open_set(
                spoof_prob=float(obs[0]),
                pcvs_score=float(obs[2]),
                mahalanobis_dist=float((1.0 - obs[4]) * 6.0),
                jitter_percent=float(obs[1] * 2.0),
                shimmer_percent=float(obs[3] * 5.0),
            )
            if res["detected_class"] != "BONAFIDE_HUMAN_SPEECH":
                caught_synthetic += 1
            if res["is_zero_day_unknown"]:
                flagged_zero_day += 1

        total = len(held_vecs)
        per_fold.append({
            "held_out_generator": held_out_gen,
            "samples": total,
            "synthetic_catch_rate": round(caught_synthetic / max(1, total), 4),
            "unknown_zero_day_rate": round(flagged_zero_day / max(1, total), 4),
        })

    avg_catch = float(np.mean([f["synthetic_catch_rate"] for f in per_fold]))
    return {
        "evaluated": True,
        "folds": per_fold,
        "mean_logo_catch_rate": round(avg_catch, 4),
    }


def render_markdown_report(results: Dict[str, Any]) -> str:
    """Format Phase 7 evaluation results as a viva-ready Markdown report."""
    clean = results["clean_wideband"]
    tel = results["telephony_8khz"]
    logo = results["logo_open_set"]

    lines = [
        "# BAFV-PCTA Phase 7: Cross-Dataset & Ablation Evaluation Report",
        "",
        f"**Weights Source:** `{results['weights_source']}`  ",
        f"**Calibration Loaded:** `{results['calibration_loaded']}`  ",
        f"**Total Samples Evaluated:** `{clean['samples_evaluated']}`",
        "",
        "---",
        "",
        "## 1. Clean Wideband vs. 8 kHz Telephony Degradation (RQ1 & RQ3)",
        "",
        "| Metric | Clean 16 kHz Wideband | 8 kHz Narrowband Telephony |",
        "|:---|:---:|:---:|",
        f"| **EER (Lower is better)** | `{clean['eer']:.4f}` | `{tel['eer']:.4f}` |",
        f"| **min-tDCF (Lower is better)** | `{clean['min_tdcf']:.4f}` | `{tel['min_tdcf']:.4f}` |",
        f"| **ROC-AUC (Higher is better)** | `{clean['roc_auc']:.4f}` | `{tel['roc_auc']:.4f}` |",
        f"| **Accuracy (@ 0.5)** | `{clean['accuracy_at_0_5']:.2%}` | `{tel['accuracy_at_0_5']:.2%}` |",
        f"| **FRR / FAR (@ 0.5)** | `{clean['frr_at_0_5']:.2%}` / `{clean['far_at_0_5']:.2%}` | `{tel['frr_at_0_5']:.2%}` / `{tel['far_at_0_5']:.2%}` |",
        f"| **Avg Inference Latency** | `{clean['avg_latency_ms']:.2f} ms` | `{tel['avg_latency_ms']:.2f} ms` |",
        "",
        "## 2. Low-Confidence Channel Guard Ablation (RQ3)",
        "",
        "| Condition | Raw Detector False-Hold Rate | DRFE With Channel Guard False-Hold Rate | Guard Activations |",
        "|:---|:---:|:---:|:---:|",
        f"| **Clean Wideband** | `{clean['bonafide_raw_hold_rate']:.2%}` | `{clean['bonafide_drfe_hold_rate']:.2%}` | `{clean['low_confidence_guard_activations']}` |",
        f"| **8 kHz Telephony** | `{tel['bonafide_raw_hold_rate']:.2%}` | `{tel['bonafide_drfe_hold_rate']:.2%}` | `{tel['low_confidence_guard_activations']}` |",
        "",
        "## 3. Leave-One-Generator-Out (LOGO) Open-Set Evaluation (RQ2)",
        "",
    ]

    if logo.get("evaluated"):
        lines.extend([
            f"**Mean Unseen-Generator Catch Rate:** `{logo['mean_logo_catch_rate']:.2%}`",
            "",
            "| Held-Out Generator | Samples | Caught as Synthetic | Labeled Zero-Day Unknown |",
            "|:---|:---:|:---:|:---:|",
        ])
        for fold in logo["folds"]:
            lines.append(
                f"| `{fold['held_out_generator']}` | {fold['samples']} | `{fold['synthetic_catch_rate']:.2%}` | `{fold['unknown_zero_day_rate']:.2%}` |"
            )
    else:
        lines.append(f"*Not run:* {logo.get('reason', 'N/A')}")

    lines.append("")
    return "\n".join(lines)


def run_phase7_evaluation(
    samples: Sequence[Tuple[Path, int]],
    checkpoint_path: Optional[Path] = None,
    calibration_path: Optional[Path] = None,
    spoof_root: Optional[Path] = None,
    json_out: Path = PROJECT_ROOT / "data" / "predictions" / "phase7_eval.json",
    report_out: Path = PROJECT_ROOT / "docs" / "PHASE_7_EVALUATION_REPORT.md",
    max_samples: int = 0,
) -> Dict[str, Any]:
    """Run full Phase 7 evaluation suite and export JSON + Markdown report."""
    detector = BaselineAntiSpoofDetector(checkpoint_path=checkpoint_path)
    bafv = BiomechanicalFeatureExtractor(sample_rate=16000, calibration_path=calibration_path)
    risk_engine = RiskEngine()

    clean_metrics = evaluate_condition(
        samples, detector, bafv, risk_engine, telephony_degrade=False, max_samples=max_samples
    )
    tel_metrics = evaluate_condition(
        samples, detector, bafv, risk_engine, telephony_degrade=True, max_samples=max_samples
    )

    bonafide_only = [(p, lbl) for p, lbl in samples if lbl == 0]
    logo_metrics = (
        evaluate_logo_open_set(bonafide_only, spoof_root, detector, bafv)
        if spoof_root
        else {"evaluated": False, "reason": "No --spoof-root directory provided."}
    )

    results: Dict[str, Any] = {
        "weights_source": detector.weights_source,
        "calibration_loaded": bafv.calibration_loaded,
        "clean_wideband": clean_metrics,
        "telephony_8khz": tel_metrics,
        "logo_open_set": logo_metrics,
    }

    json_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report_out.write_text(render_markdown_report(results), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BAFV-PCTA Phase 7 Evaluation Suite")
    parser.add_argument("--data-dir", type=Path, help="Evaluation folder with bonafide/ and spoof/ subfolders")
    parser.add_argument("--asvspoof-protocol", type=Path, help="ASVspoof 2019 LA eval/dev protocol txt")
    parser.add_argument("--audio-root", type=Path, help="Audio directory for ASVspoof protocol")
    parser.add_argument("--spoof-root", type=Path, help="Optional generator-grouped spoof folder for LOGO evaluation")
    parser.add_argument("--checkpoint", type=Path, help="Trained checkpoint path (.pt)")
    parser.add_argument("--calibration", type=Path, help="Empirical calibration JSON path")
    parser.add_argument("--max-samples", type=int, default=0, help="Cap evaluation samples (0 = all)")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=PROJECT_ROOT / "data" / "predictions" / "phase7_eval.json",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=PROJECT_ROOT / "docs" / "PHASE_7_EVALUATION_REPORT.md",
    )
    args = parser.parse_args()

    if args.data_dir:
        samples = collect_folder_samples(args.data_dir)
    elif args.asvspoof_protocol and args.audio_root:
        samples = collect_asvspoof_samples(args.asvspoof_protocol, args.audio_root)
    else:
        parser.error("Provide either --data-dir or (--asvspoof-protocol and --audio-root).")
        return

    res = run_phase7_evaluation(
        samples=samples,
        checkpoint_path=args.checkpoint,
        calibration_path=args.calibration,
        spoof_root=args.spoof_root,
        json_out=args.json_out,
        report_out=args.report_out,
        max_samples=args.max_samples,
    )
    print(
        f"Phase 7 Eval Complete | Clean EER={res['clean_wideband']['eer']:.4f} "
        f"| 8kHz EER={res['telephony_8khz']['eer']:.4f} | Report: {args.report_out}"
    )


if __name__ == "__main__":
    main()
