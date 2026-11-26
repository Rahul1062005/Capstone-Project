"""Calibrate Empirical BAFV Coupling & Open-Set Generator Centroids.

Computes:
1. human_coupling_mean & human_coupling_std (6D PCTA descriptors) from genuine human speech.
2. human_manifold_centroid & human_manifold_radius_95th (5D Open-Set descriptors).
3. Optional known_generators clusters from --spoof-root/<generator_name>/*.wav.

Writes models/calibration_stats.json, which BiomechanicalFeatureExtractor and
OpenSetGeneratorDetector load automatically at startup.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_service.audio_processor import AudioProcessingError, AudioProcessor
from ai_service.bafv_features import BiomechanicalFeatureExtractor
from ai_service.baseline_detector import BaselineAntiSpoofDetector
from ai_service.open_set_detector import OpenSetGeneratorDetector

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}


def _iter_audio_files(folder: Path, limit: int) -> List[Path]:
    files = [
        p for p in sorted(folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return files[:limit] if limit > 0 else files


def calibrate_from_directories(
    bonafide_dir: Path,
    spoof_root: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    output_path: Path = PROJECT_ROOT / "models" / "calibration_stats.json",
    max_files_per_class: int = 500,
) -> Dict[str, Any]:
    """Extract empirical BAFV & Open-Set statistics from genuine and synthetic speech folders."""
    processor = AudioProcessor(target_sample_rate=16000)
    detector = BaselineAntiSpoofDetector(checkpoint_path=checkpoint_path)
    bafv = BiomechanicalFeatureExtractor(sample_rate=16000, calibration_path="__none__")

    bonafide_files = _iter_audio_files(bonafide_dir, max_files_per_class)
    if len(bonafide_files) < 2:
        raise ValueError(f"Need at least 2 audio files in {bonafide_dir}; found {len(bonafide_files)}.")

    coupling_vectors: List[np.ndarray] = []
    human_obs_vectors: List[np.ndarray] = []

    for audio_path in bonafide_files:
        try:
            waveform, _ = processor.load_and_preprocess(audio_path)
            if len(waveform) < int(0.5 * 16000):
                continue
            trajectories = bafv.extract_frame_trajectories(waveform)
            coupling_vec, _ = bafv.compute_coupling_vector(trajectories)
            coupling_vectors.append(coupling_vec)

            profile = bafv.compute_bafv_profile(waveform)
            pred = detector.predict(waveform)
            obs = OpenSetGeneratorDetector.build_observation_vector(
                spoof_prob=pred["scores"]["spoof_probability"],
                pcvs_score=profile["bafv_anomaly_score"],
                mahalanobis_dist=profile["pcta_engine"]["mahalanobis_distance"],
                jitter_percent=profile["pitch_f0"]["jitter_percent"],
                shimmer_percent=profile["voice_quality"]["shimmer_percent"],
            )
            human_obs_vectors.append(obs)
        except AudioProcessingError:
            continue

    if not coupling_vectors or not human_obs_vectors:
        raise ValueError("No valid audio samples could be processed for calibration.")

    coupling_mat = np.vstack(coupling_vectors)
    coupling_mean = np.mean(coupling_mat, axis=0)
    coupling_std = np.maximum(np.std(coupling_mat, axis=0), 1e-3)

    human_mat = np.vstack(human_obs_vectors)
    human_centroid = np.mean(human_mat, axis=0)
    human_dists = np.linalg.norm(human_mat - human_centroid, axis=1)
    human_radius_95th = float(np.percentile(human_dists, 95))

    known_generators: Dict[str, Dict[str, Any]] = {}
    if spoof_root and spoof_root.exists():
        for gen_dir in sorted(p for p in spoof_root.iterdir() if p.is_dir()):
            gen_files = _iter_audio_files(gen_dir, max_files_per_class)
            gen_obs: List[np.ndarray] = []
            for audio_path in gen_files:
                try:
                    waveform, _ = processor.load_and_preprocess(audio_path)
                    if len(waveform) < int(0.5 * 16000):
                        continue
                    profile = bafv.compute_bafv_profile(waveform)
                    pred = detector.predict(waveform)
                    obs = OpenSetGeneratorDetector.build_observation_vector(
                        spoof_prob=pred["scores"]["spoof_probability"],
                        pcvs_score=profile["bafv_anomaly_score"],
                        mahalanobis_dist=profile["pcta_engine"]["mahalanobis_distance"],
                        jitter_percent=profile["pitch_f0"]["jitter_percent"],
                        shimmer_percent=profile["voice_quality"]["shimmer_percent"],
                    )
                    gen_obs.append(obs)
                except AudioProcessingError:
                    continue

            if gen_obs:
                gen_mat = np.vstack(gen_obs)
                gen_centroid = np.mean(gen_mat, axis=0)
                gen_dists = np.linalg.norm(gen_mat - gen_centroid, axis=1)
                radius = float(max(0.25, np.percentile(gen_dists, 95)))
                known_generators[gen_dir.name] = {
                    "family": gen_dir.name,
                    "centroid": [round(float(x), 4) for x in gen_centroid],
                    "max_cluster_radius": round(radius, 4),
                    "sample_count": len(gen_obs),
                    "description": f"Empirically calibrated cluster from {len(gen_obs)} samples",
                }

    calibration_payload: Dict[str, Any] = {
        "bonafide_sample_count": len(coupling_vectors),
        "human_coupling_mean": [round(float(x), 4) for x in coupling_mean],
        "human_coupling_std": [round(float(x), 4) for x in coupling_std],
        "human_manifold_centroid": [round(float(x), 4) for x in human_centroid],
        "human_manifold_radius_95th": round(human_radius_95th, 4),
        "known_generators": known_generators,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(calibration_payload, indent=2), encoding="utf-8")
    return calibration_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate BAFV-PCTA and Open-Set centroids")
    parser.add_argument("--bonafide-dir", type=Path, required=True, help="Directory of genuine human audio")
    parser.add_argument(
        "--spoof-root",
        type=Path,
        help="Optional directory with one subfolder per synthetic generator (e.g., ElevenLabs/, RVC/)",
    )
    parser.add_argument("--checkpoint", type=Path, help="Optional trained detector checkpoint (.pt)")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models" / "calibration_stats.json",
        help="Output JSON path",
    )
    parser.add_argument("--max-files", type=int, default=500, help="Max files per class/generator")
    args = parser.parse_args()

    stats = calibrate_from_directories(
        bonafide_dir=args.bonafide_dir,
        spoof_root=args.spoof_root,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        max_files_per_class=args.max_files,
    )
    print(f"Saved empirical calibration to {args.output} ({stats['bonafide_sample_count']} bonafide samples)")


if __name__ == "__main__":
    main()
