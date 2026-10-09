"""Windowed Audio Analyzer for BAFV-PCTA.

Implements FR-05:
- Analyses continuous audio in 2.0 s windows with 0.5 s hop.
- Generates a timestamped timeline of risk scores and recommendations.
- Integrates VAD (Voice Activity Detection), baseline inference, and BAFV features.
- Applies exponential risk smoothing: S_t = 0.6 * S_(t-1) + 0.4 * s_t.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from ai_service.audio_processor import AudioProcessor
from ai_service.baseline_detector import BaselineAntiSpoofDetector
from ai_service.bafv_features import BiomechanicalFeatureExtractor
from ai_service.vad import VoiceActivityDetector


class WindowedStreamAnalyzer:
    """Sliding-window near-real-time timeline analyzer."""

    def __init__(
        self,
        window_duration_sec: float = 2.0,
        hop_duration_sec: float = 0.5,
        target_sample_rate: int = 16000,
        detector: Optional[BaselineAntiSpoofDetector] = None,
    ) -> None:
        self.target_sample_rate = target_sample_rate
        self.window_samples = int(window_duration_sec * target_sample_rate)
        self.hop_samples = int(hop_duration_sec * target_sample_rate)

        self.audio_processor = AudioProcessor(target_sample_rate=target_sample_rate)
        self.vad = VoiceActivityDetector(sample_rate=target_sample_rate)
        self.bafv_extractor = BiomechanicalFeatureExtractor(sample_rate=target_sample_rate)
        self.detector = detector or BaselineAntiSpoofDetector()

    def analyze_full_audio_timeline(self, waveform: np.ndarray) -> Dict[str, Any]:
        """Slide across waveform and produce complete windowed risk timeline."""
        total_samples = len(waveform)
        total_duration = total_samples / float(self.target_sample_rate)

        # Minimum required length: at least 0.5 seconds
        if total_samples < int(0.5 * self.target_sample_rate):
            return {
                "total_duration_seconds": total_duration,
                "total_windows": 0,
                "timeline": [],
                "aggregate_recommendation": "ALLOW",
                "overall_risk_score": 0.0,
            }

        # If audio is shorter than 2.0s, pad with zeros to form at least 1 full window
        if total_samples < self.window_samples:
            pad_len = self.window_samples - total_samples
            padded_waveform = np.pad(waveform, (0, pad_len), mode="constant")
            windows = [(0, padded_waveform)]
        else:
            windows = []
            for start in range(0, total_samples - self.window_samples + 1, self.hop_samples):
                chunk = waveform[start : start + self.window_samples]
                windows.append((start, chunk))

        timeline: List[Dict[str, Any]] = []
        smoothed_score = 0.0
        hold_count = 0

        for idx, (start_idx, chunk) in enumerate(windows):
            start_sec = round(start_idx / float(self.target_sample_rate), 2)
            end_sec = round((start_idx + len(chunk)) / float(self.target_sample_rate), 2)

            # 1. Voice Activity Detection
            speech_ratio = self.vad.get_speech_ratio(chunk)
            is_speech_active = speech_ratio >= 0.25

            # 2. Channel Metrics
            chan_metrics = self.audio_processor.calculate_channel_metrics(
                chunk, self.target_sample_rate, self.target_sample_rate
            )
            chan_conf = chan_metrics.get("channel_confidence", 1.0)

            # 3. Baseline & BAFV Feature Inference
            if is_speech_active:
                pred = self.detector.predict(chunk)
                base_spoof_prob = pred["scores"]["spoof_probability"]
                bafv_profile = self.bafv_extractor.compute_bafv_profile(chunk)
                bafv_score = bafv_profile["bafv_anomaly_score"]

                # Window-level raw risk fused
                window_raw_risk = float(np.clip(base_spoof_prob * 0.7 + bafv_score * 0.3, 0.01, 0.99))
            else:
                # Silence/background noise carries zero synthetic threat
                base_spoof_prob = 0.0
                bafv_score = 0.0
                window_raw_risk = 0.0

            # 4. Exponential Smoothing: S_t = 0.6 * S_(t-1) + 0.4 * s_t (Scope Section 9)
            if idx == 0:
                smoothed_score = window_raw_risk
            else:
                smoothed_score = 0.6 * smoothed_score + 0.4 * window_raw_risk

            # 5. Recommendation Boundaries (ALLOW < 0.35, VERIFY [0.35, 0.75), HOLD >= 0.75)
            if smoothed_score < 0.35:
                window_action = "ALLOW"
            elif smoothed_score < 0.75:
                window_action = "VERIFY"
            else:
                window_action = "HOLD_FOR_REVIEW"
                hold_count += 1

            timeline.append({
                "window_index": idx,
                "start_time_seconds": start_sec,
                "end_time_seconds": end_sec,
                "is_speech_active": is_speech_active,
                "speech_ratio": round(speech_ratio, 3),
                "baseline_spoof_probability": round(base_spoof_prob, 4),
                "bafv_anomaly_score": round(bafv_score, 4),
                "channel_confidence": chan_conf,
                "instantaneous_risk": round(window_raw_risk, 4),
                "smoothed_risk": round(smoothed_score, 4),
                "recommendation": window_action,
            })

        # Final aggregate recommendation with persistence rule (Scope section 9):
        # HOLD FOR REVIEW requires hold band in >= 3 of the last 5 windows
        recent_windows = timeline[-5:] if len(timeline) >= 5 else timeline
        recent_holds = sum(1 for w in recent_windows if w["recommendation"] == "HOLD_FOR_REVIEW")

        if recent_holds >= 3 or (len(timeline) < 3 and smoothed_score >= 0.75):
            overall_recommendation = "HOLD_FOR_REVIEW"
        elif smoothed_score >= 0.35:
            overall_recommendation = "VERIFY"
        else:
            overall_recommendation = "ALLOW"

        return {
            "total_duration_seconds": round(total_duration, 2),
            "total_windows": len(timeline),
            "timeline": timeline,
            "overall_risk_score": round(smoothed_score, 4),
            "aggregate_recommendation": overall_recommendation,
            "persistence_rule_satisfied": recent_holds >= 3,
        }
