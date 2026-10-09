"""FastAPI Microservice for BAFV-PCTA Audio & Anti-Spoofing Inference.

Phase 3 Deliverable: AI API (Exit Check: API Tests Pass).
Complies with:
- FR-01: Safe audio ingestion and validation.
- FR-02: 16 kHz float32 normalization.
- FR-03: Real model output with metadata.
- FR-04: Score interpretation, quality warnings, probabilistic disclaimer.
- FR-05: 2.0 s windowed analysis with 0.5 s hop and live risk timeline.
- FR-12: Explicit error states with fail-secure HTTP status codes.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ai_service.audio_processor import AudioProcessor, AudioProcessingError
from ai_service.baseline_detector import BaselineAntiSpoofDetector
from ai_service.bafv_features import BiomechanicalFeatureExtractor
from ai_service.windowed_analyzer import WindowedStreamAnalyzer
from ai_service.risk_engine import (
    AudioEvidenceContext,
    ChallengeOutcome,
    RiskDecisionOutput,
    RiskEngine,
    TransactionContext,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AI_API")

app = FastAPI(
    title="BAFV-PCTA AI Detection Microservice",
    description="Real-Time Audio Anti-Spoofing and Biomechanical Acoustic Verification Engine",
    version="1.0.0",
)

# Enable CORS for Frontend Development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Singletons
processor = AudioProcessor(target_sample_rate=16000)
detector = BaselineAntiSpoofDetector()
bafv_extractor = BiomechanicalFeatureExtractor(sample_rate=16000)
window_analyzer = WindowedStreamAnalyzer(detector=detector)
risk_engine = RiskEngine()


# --- Response Schemas ---
class HealthResponse(BaseModel):
    status: str
    service: str
    model_name: str
    model_version: str
    device: str
    target_sample_rate: int
    timestamp_utc: float


class SingleWindowPayload(BaseModel):
    samples: List[float] = Field(..., description="16 kHz float32 PCM samples (typically 32,000 samples for 2.0s)")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return AI service health, active device, and model metadata."""
    return HealthResponse(
        status="healthy",
        service="bafv-pcta-ai-service",
        model_name=detector.MODEL_NAME,
        model_version=detector.MODEL_VERSION,
        device=str(detector.device),
        target_sample_rate=detector.TARGET_SR,
        timestamp_utc=time.time(),
    )


@app.post("/api/v1/detect/file")
async def detect_audio_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload audio file, validate, perform windowed analysis, and return timeline (FR-01 through FR-05)."""
    start_total = time.perf_counter()

    # Read uploaded bytes safely
    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read upload payload: {str(exc)}",
        )

    # Validate and preprocess (FR-01, FR-02, FR-12)
    try:
        waveform, quality_meta = processor.load_and_preprocess(content)
    except AudioProcessingError as ape:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ape),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal audio decoding failure: {str(exc)}",
        )

    # 1. Whole-utterance baseline inference (FR-03, FR-04)
    baseline_result = detector.predict(waveform)

    # 2. Extract Biomechanical Acoustic Features (BAFV)
    bafv_profile = bafv_extractor.compute_bafv_profile(waveform)

    # 3. Sliding-window timeline analysis (FR-05)
    timeline_result = window_analyzer.analyze_full_audio_timeline(waveform)

    total_processing_ms = (time.perf_counter() - start_total) * 1000.0

    return {
        "file_name": file.filename,
        "processing_latency_ms": round(total_processing_ms, 2),
        "channel_metadata": quality_meta,
        "baseline_detection": baseline_result,
        "biomechanical_features": bafv_profile,
        "windowed_timeline": timeline_result,
        "disclaimer": (
            "Notice: Predictions are probabilistic acoustic risk assessments and provide advisory decision support "
            "for bank fraud analysts. They do not constitute conclusive proof of identity or automated transaction blocking."
        ),
    }


@app.post("/api/v1/detect/window")
async def detect_audio_window(payload: SingleWindowPayload) -> Dict[str, Any]:
    """Evaluate a single 2.0-second streaming window."""
    samples_arr = np.array(payload.samples, dtype=np.float32)
    if len(samples_arr) < 8000:  # < 0.5s at 16kHz
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload must contain at least 8,000 samples (0.5s at 16 kHz).",
        )

    try:
        prediction = detector.predict(samples_arr)
        bafv = bafv_extractor.compute_bafv_profile(samples_arr)
        return {
            "prediction": prediction,
            "biomechanical_features": bafv,
        }
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


class RiskEvaluationPayload(BaseModel):
    audio_evidence: AudioEvidenceContext
    transaction_context: TransactionContext
    previous_smoothed_score: float = 0.0
    recent_window_actions: Optional[List[str]] = None
    challenge_outcome: Optional[ChallengeOutcome] = None


@app.post("/api/v1/risk/evaluate", response_model=RiskDecisionOutput)
async def evaluate_multimodal_risk(payload: RiskEvaluationPayload) -> RiskDecisionOutput:
    """Multimodal Risk Engine: combine audio evidence, channel confidence, and transaction signals (FR-06, FR-07, FR-08)."""
    return risk_engine.evaluate_risk(
        audio=payload.audio_evidence,
        txn=payload.transaction_context,
        previous_smoothed_score=payload.previous_smoothed_score,
        recent_window_actions=payload.recent_window_actions,
        challenge_outcome=payload.challenge_outcome,
    )


@app.websocket("/api/v1/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket) -> None:
    """WebSocket streaming endpoint for near-real-time audio chunks."""
    await websocket.accept()
    logger.info("WebSocket client connected for live stream detection.")

    buffer = np.array([], dtype=np.float32)
    window_samples = int(2.0 * 16000)
    hop_samples = int(0.5 * 16000)

    try:
        while True:
            # Receive float32 array or bytes from client
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.float32)
            buffer = np.concatenate([buffer, chunk])

            # Whenever buffer exceeds window size, evaluate
            while len(buffer) >= window_samples:
                window_data = buffer[:window_samples]
                buffer = buffer[hop_samples:]

                prediction = detector.predict(window_data)
                await websocket.send_json({
                    "timestamp": time.time(),
                    "spoof_probability": prediction["scores"]["spoof_probability"],
                    "recommendation": prediction["scores"]["recommended_action"],
                    "quality_warnings": prediction["quality_warnings"],
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as exc:
        logger.error("WebSocket streaming error: %s", str(exc))
        await websocket.close()
