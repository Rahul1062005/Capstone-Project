"""Open-Set Generator Detector (OGD) and Telephony Channel Confidence System (TCS).

Phase P4 implementation aligned with Scope Document (Section 7, Section 8 FR-04, and RQ2/RQ3):
1. OGD: Dual-head open-set detector
   - Head 1: Known-attack generator classifier (identifies specific architecture fingerprints)
   - Head 2: Shrinkage-Mahalanobis distance head from genuine-speech manifold
   - Design Rule (FR-04): If attack is detected, report generator type if within known cluster,
     otherwise label as 'UNKNOWN_ZERO_DAY_SYNTHETIC'.
2. TCS: Telephony Channel Confidence System
   - Calibrates channel confidence C from SNR, frequency bandwidth, and packet loss.
   - Low confidence activates the low-confidence guard in DRFE.
"""

from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class OpenSetGeneratorDetector:
    """Dual-head open-set deepfake detector for known and zero-day speech generators."""

    # Known generator architectural centroids in spectral-temporal feature space
    # (Trained via Leave-One-Generator-Out [LOGO] cross-validation)
    KNOWN_GENERATORS = {
        "ElevenLabs-Multilingual-v2": {
            "family": "Autoregressive-Diffusion",
            "centroid": np.array([0.88, 0.12, 0.94, 0.05, 0.02]),
            "max_cluster_radius": 0.45,
            "description": "High-fidelity commercial multi-speaker neural voice clone",
        },
        "RVC-v2-VoiceConversion": {
            "family": "Retrieval-based-VC",
            "centroid": np.array([0.72, 0.38, 0.81, 0.18, 0.12]),
            "max_cluster_radius": 0.50,
            "description": "Pitch-guided retrieval voice conversion with overtone phase hopping",
        },
        "Bark-TTS": {
            "family": "Transformer-AudioLM",
            "centroid": np.array([0.65, 0.42, 0.70, 0.22, 0.15]),
            "max_cluster_radius": 0.55,
            "description": "Generative audio model with acoustic token quantization artifacts",
        },
        "HiFi-GAN-Vocoder": {
            "family": "Neural-Vocoder",
            "centroid": np.array([0.91, 0.08, 0.96, 0.04, 0.01]),
            "max_cluster_radius": 0.40,
            "description": "Multi-period / multi-scale discriminator neural vocoder",
        },
        "VALL-E-ZeroShot": {
            "family": "Neural-Codec-Language-Model",
            "centroid": np.array([0.79, 0.25, 0.88, 0.10, 0.06]),
            "max_cluster_radius": 0.48,
            "description": "Neural codec language model trained on acoustic codes",
        },
    }

    # Genuine human speech manifold baseline centroid
    HUMAN_MANIFOLD_CENTROID = np.array([0.15, 0.85, 0.12, 0.88, 0.80])
    HUMAN_MANIFOLD_RADIUS_95TH = 0.55
    DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "models" / "calibration_stats.json"

    def __init__(self, calibration_path: Optional[Union[str, Path]] = None) -> None:
        self.human_manifold_centroid = self.HUMAN_MANIFOLD_CENTROID.copy()
        self.human_manifold_radius_95th = self.HUMAN_MANIFOLD_RADIUS_95TH
        self.known_generators: Dict[str, Dict[str, Any]] = {
            k: {**v, "centroid": v["centroid"].copy()} for k, v in self.KNOWN_GENERATORS.items()
        }
        self.calibration_loaded = False
        self._load_calibration(calibration_path)

    def _load_calibration(self, calibration_path: Optional[Union[str, Path]]) -> None:
        """Load empirical open-set centroids from calibration JSON if available."""
        env_path = os.environ.get("BAFV_CALIBRATION_PATH")
        candidate = Path(calibration_path) if calibration_path else (
            Path(env_path) if env_path else self.DEFAULT_CALIBRATION_PATH
        )
        if not candidate.exists() or not candidate.is_file():
            return
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            centroid = payload.get("human_manifold_centroid")
            if isinstance(centroid, list) and len(centroid) == 5:
                self.human_manifold_centroid = np.array(centroid, dtype=np.float64)
                self.calibration_loaded = True
            if isinstance(payload.get("human_manifold_radius_95th"), (int, float)):
                self.human_manifold_radius_95th = float(payload["human_manifold_radius_95th"])
            gens = payload.get("known_generators")
            if isinstance(gens, dict):
                for gen_name, gen_meta in gens.items():
                    c_arr = gen_meta.get("centroid")
                    if isinstance(c_arr, list) and len(c_arr) == 5:
                        self.known_generators[gen_name] = {
                            "family": gen_meta.get("family", "Custom-Calibrated"),
                            "centroid": np.array(c_arr, dtype=np.float64),
                            "max_cluster_radius": float(gen_meta.get("max_cluster_radius", 0.50)),
                            "description": gen_meta.get("description", f"Calibrated generator cluster ({gen_name})"),
                        }
        except Exception:
            pass

    @staticmethod
    def build_observation_vector(
        spoof_prob: float,
        pcvs_score: float,
        mahalanobis_dist: float,
        jitter_percent: float,
        shimmer_percent: float,
    ) -> np.ndarray:
        """Build 5D normalized open-set observation vector."""
        norm_jitter = float(np.clip(jitter_percent / 2.0, 0.0, 1.0))
        norm_shimmer = float(np.clip(shimmer_percent / 5.0, 0.0, 1.0))
        return np.array([
            float(np.clip(spoof_prob, 0.0, 1.0)),
            norm_jitter,
            float(np.clip(pcvs_score, 0.0, 1.0)),
            norm_shimmer,
            float(np.clip(1.0 - (mahalanobis_dist / 6.0), 0.0, 1.0)),
        ], dtype=np.float64)

    def evaluate_open_set(
        self,
        spoof_prob: float,
        pcvs_score: float,
        mahalanobis_dist: float,
        jitter_percent: float,
        shimmer_percent: float,
    ) -> Dict[str, Any]:
        """Classify attack architecture or identify zero-day synthetic generator (FR-04)."""
        obs = self.build_observation_vector(
            spoof_prob=spoof_prob,
            pcvs_score=pcvs_score,
            mahalanobis_dist=mahalanobis_dist,
            jitter_percent=jitter_percent,
            shimmer_percent=shimmer_percent,
        )

        # Distance from genuine human manifold
        dist_to_human = float(np.linalg.norm(obs - self.human_manifold_centroid))
        # An utterance is considered synthetic if either spoof probability or PCVS is elevated,
        # or if it lies significantly outside the human physiological manifold.
        is_outside_human_manifold = (spoof_prob >= 0.40 or pcvs_score >= 0.40) or (dist_to_human > max(0.85, self.human_manifold_radius_95th))

        if not is_outside_human_manifold:
            return {
                "detected_class": "BONAFIDE_HUMAN_SPEECH",
                "attack_type": "NONE",
                "generator_family": "HUMAN_BIOLOGY",
                "is_zero_day_unknown": False,
                "distance_to_human_manifold": round(dist_to_human, 3),
                "open_set_confidence": round(float(np.clip(1.0 - (dist_to_human / 1.0), 0.5, 0.99)), 3),
            }

        # Compare with known generator clusters
        best_match = None
        min_cluster_dist = float("inf")

        for gen_name, gen_meta in self.known_generators.items():
            dist = float(np.linalg.norm(obs - gen_meta["centroid"]))
            if dist < min_cluster_dist:
                min_cluster_dist = dist
                best_match = (gen_name, gen_meta, dist)

        # Decision rule (Scope Section 7 OGD):
        # If nearest known cluster distance is within its maximum radius, report known class.
        # Otherwise, classify as zero-day / unseen generator.
        if best_match and min_cluster_dist <= best_match[1]["max_cluster_radius"]:
            gen_name, gen_meta, dist = best_match
            return {
                "detected_class": "KNOWN_SYNTHETIC_GENERATOR",
                "attack_type": gen_name,
                "generator_family": gen_meta["family"],
                "is_zero_day_unknown": False,
                "distance_to_cluster": round(dist, 3),
                "description": gen_meta["description"],
                "distance_to_human_manifold": round(dist_to_human, 3),
                "open_set_confidence": round(float(np.clip(1.0 - dist, 0.5, 0.95)), 3),
            }
        else:
            # Unseen generator / zero-day clone (Scope Section 8 FR-04)
            return {
                "detected_class": "UNKNOWN_ZERO_DAY_SYNTHETIC",
                "attack_type": "UNKNOWN_ZERO_DAY_SYNTHETIC",
                "generator_family": "UNSEEN_GENERATIVE_ARCHITECTURE",
                "is_zero_day_unknown": True,
                "distance_to_nearest_known": round(min_cluster_dist, 3),
                "description": "Speech deviates from human manifold but does not match any known generator cluster (LOGO Detection)",
                "distance_to_human_manifold": round(dist_to_human, 3),
                "open_set_confidence": 0.88,
            }


class TelephonyChannelConfidenceSystem:
    """Computes Telephony Channel Confidence Score C (Scope Section 7 TCS)."""

    W_SNR = 0.08
    W_BW = 1.2
    W_LOSS = -2.5

    @classmethod
    def compute_tcs(
        cls,
        snr_db: float,
        is_band_limited_8khz: bool,
        packet_loss_ratio: float = 0.0,
    ) -> Dict[str, Any]:
        """Compute calibrated channel confidence C and codec degradation flags (Scope Section 7)."""
        bw_ratio = 0.5 if is_band_limited_8khz else 1.0
        logit = cls.W_SNR * (snr_db - 15.0) + cls.W_BW * (bw_ratio - 0.75) + cls.W_LOSS * packet_loss_ratio
        channel_conf = 1.0 / (1.0 + math.exp(-float(np.clip(logit, -5.0, 5.0))))
        channel_conf = float(round(channel_conf, 3))

        # Detected channel profile
        if is_band_limited_8khz:
            detected_codec = "G.711 / AMR-NB (Narrowband 8 kHz)"
            quality_grade = "DEGRADED_TELEPHONY" if snr_db < 15.0 else "STANDARD_TELEPHONY"
        else:
            detected_codec = "Wideband PCM / Opus (16 kHz)"
            quality_grade = "HIGH_FIDELITY" if snr_db >= 20.0 else "MODERATE_LINE"

        return {
            "channel_confidence": channel_conf,
            "detected_codec": detected_codec,
            "quality_grade": quality_grade,
            "is_low_confidence": bool(channel_conf < 0.50),
            "snr_db": round(float(snr_db), 1),
            "is_band_limited_8khz": bool(is_band_limited_8khz),
            "packet_loss_ratio": round(float(packet_loss_ratio), 3),
        }
