"""Run Pretrained Anti-Spoofing Baseline Model Inference across Indexed Audio.

Phase 2 Deliverable: Run pretrained model; inspect failures.
Exit Check: Real model output (FR-03, FR-04, FR-12).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List

from ai_service.baseline_detector import BaselineAntiSpoofDetector
from ai_service.audio_processor import AudioProcessingError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RunBaseline")


def run_inference_on_manifest(
    manifest_path: Path,
    output_path: Path,
) -> List[Dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    detector = BaselineAntiSpoofDetector()
    results: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print("RUNNING BAFV-PCTA PRETRAINED BASELINE INFERENCE")
    print(f"Model: {detector.MODEL_NAME} | Version: {detector.MODEL_VERSION} | Device: {detector.device}")
    print("=" * 80 + "\n")

    for item in items:
        file_path = Path(item["file_path"])
        file_name = item["file_name"]
        ground_truth_label = item.get("label", "unlabelled")

        logger.info("Processing file: %s (Ground Truth: %s)", file_name, ground_truth_label)

        try:
            prediction = detector.predict(file_path)
            record = {
                "file_name": file_name,
                "file_path": str(file_path),
                "ground_truth_label": ground_truth_label,
                "status": "SUCCESS",
                "prediction": prediction,
                "error": None,
            }
            spoof_p = prediction["scores"]["spoof_probability"]
            rec_action = prediction["scores"]["recommended_action"]
            latency_ms = prediction["inference_metrics"]["inference_latency_ms"]
            print(f"  -> [{file_name}] Spoof Prob: {spoof_p:.4f} | Action: {rec_action} | Latency: {latency_ms:.1f}ms")

        except AudioProcessingError as ape:
            logger.warning("Controlled audio rejection for %s: %s", file_name, str(ape))
            record = {
                "file_name": file_name,
                "file_path": str(file_path),
                "ground_truth_label": ground_truth_label,
                "status": "REJECTED_INVALID_AUDIO",
                "prediction": None,
                "error": str(ape),
            }
            print(f"  -> [{file_name}] SAFELY REJECTED: {str(ape)}")
        except Exception as exc:
            logger.error("Unexpected failure on %s: %s", file_name, str(exc))
            record = {
                "file_name": file_name,
                "file_path": str(file_path),
                "ground_truth_label": ground_truth_label,
                "status": "ERROR",
                "prediction": None,
                "error": str(exc),
            }

        results.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, indent=2)

    print("\n" + "=" * 80)
    print(f"Inference complete! Results saved to: {output_path.resolve()}")
    print("=" * 80 + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Baseline Inference")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/dataset_manifest.json",
        help="Path to manifest JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/predictions/baseline_predictions.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    run_inference_on_manifest(Path(args.manifest), Path(args.output))


if __name__ == "__main__":
    main()
