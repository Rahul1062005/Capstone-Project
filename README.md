# BAFV-PCTA: AI Deepfake & Synthetic Audio Detection System for Bank Cyber-Fraud
### Voice-Clone Fraud Shield: Biomechanical Acoustic Feature Verification with Physiological Coupled-Trajectory Analysis
> **SIH Domain / Theme:** Cybersecurity & FinTech  
> **Status:** Phase 1 (Define) & Phase 2 (Baseline) Completed  
> **Intended Use:** Student research prototype and demonstration; advisory support for bank fraud analysts (not for production banking use).

---

## 1. Problem Formulation & Novelty

### The Real-World Threat: Vishing & Executive Impersonation
Scammers use AI voice cloners and neural text-to-speech (TTS) engines to sound like family members or corporate executives, convincing banks and victims to authorize high-value wire transfers over the phone (Vishing attacks).

### The BAFV-PCTA Solution
Human vocal production creates physical acoustics—subtle breathing pauses, glottal micro-perturbations (jitter/shimmer), and continuous vocal-tract resonance transitions ($F_1, F_2, F_3$ formants)—that generative AI models miss or distort. 

BAFV-PCTA introduces:
1. **Windowed Passive Anti-Spoofing Baseline:** Real-time 2.0 s windows with 0.5 s hop analyzing spectro-temporal anomalies and channel quality.
2. **Transaction-Bound Spoken Challenge (TB-PC):** When caller risk is ambiguous (`VERIFY`), the system issues an ephemeral, keyed challenge derived from transaction details and a single-use nonce to defeat replay attacks and expose synthesis delay.
3. **Multimodal Risk Fusion Engine:** Combines audio evidence, channel confidence index, and simulated transaction context into an analyst recommendation: `ALLOW`, `VERIFY`, or `HOLD FOR REVIEW`.

---

## 2. Completed Milestones (Phase 1 & Phase 2)

As specified in the Project Scope Document, work has strictly progressed through **Phase 1** and **Phase 2**:

### Phase 1: Define (Weeks 1–2) — Exit Check: Requirements Doc & Data Indexed ✅
- [x] **Project Requirements & Specs:** Documented in [`docs/PROJECT_SPECIFICATION.md`](docs/PROJECT_SPECIFICATION.md), tracking functional requirements (FR-01 through FR-13), Non-functional requirements, and the mathematical risk engine.
- [x] **SIH Verification & Alignment:** Documented in [`docs/SIH_PROBLEM_ALIGNMENT.md`](docs/SIH_PROBLEM_ALIGNMENT.md) mapping threat models, patent core, and prototype boundaries.
- [x] **Licensing & Academic Attribution:** Documented in [`docs/LICENSES_AND_ATTRIBUTION.md`](docs/LICENSES_AND_ATTRIBUTION.md) attributing ASVspoof 2019, In-the-Wild, AASIST, PyTorch, Librosa, and Whisper.
- [x] **Dataset Indexer & Validator:** Automated in [`scripts/index_data.py`](scripts/index_data.py) with RIFF header inspection, SHA-256 fingerprinting, and generation of `data/manifests/dataset_manifest.json` and `data/manifests/dataset_manifest.csv`.
- [x] **Data Fixtures & Protocols:** Generated benchmark dataset in `data/test_samples/` featuring genuine human speech, synthetic clones, and 8 kHz telephone degraded audio.

### Phase 2: Baseline (Weeks 3–4) — Exit Check: Real Model Output ✅
- [x] **Audio Preprocessor & Validator (`ai_service/audio_processor.py`):**
  - Safe decoding and validation (FR-01: content verification, rejects corrupted files).
  - Normalization and resampling to 16 kHz mono float32 (FR-02).
  - Channel metrics calculation: SNR estimation, clipping ratio, and 8 kHz telephone band-limiting detection.
  - Fail-secure error handling (FR-12).
- [x] **Pretrained Baseline Detector (`ai_service/baseline_detector.py`):**
  - Spectro-temporal attention network (`AASIST-ResNet-Baseline` v1.2.0-baseline).
  - Real tensor forward inference with raw logits and calibrated spoof probability (FR-03).
  - Score interpretation (`ALLOW`, `VERIFY`, `HOLD_FOR_REVIEW`), quality warnings, and probabilistic disclaimer (FR-04).
  - Sub-25ms inference latency tracking on standard CPU (SC6).
- [x] **Failure Inspection & Degradation Analysis (`scripts/inspect_baseline_failures.py`):**
  - Evaluates performance across clean vs 8 kHz telephone channel degradation (SC3, SC4).
  - Full report generated at [`docs/PHASE_2_BASELINE_REPORT.md`](docs/PHASE_2_BASELINE_REPORT.md).

### Phase 3: AI API (Weeks 5–6) — Exit Check: API Tests Pass ✅
- [x] **Voice Activity Detection (`ai_service/vad.py`):** Energy and spectral flux VAD filtering silence and background noise.
- [x] **Biomechanical Feature Extractor (`ai_service/bafv_features.py`):** F0 jitter, LPC formants (F1, F2, F3), shimmer, HNR, and anomaly scoring.
- [x] **Windowed Stream Analyzer (`ai_service/windowed_analyzer.py`):** 2.0 s sliding windows with 0.5 s hop (FR-05) and risk timeline smoothing ($S_t$).
- [x] **FastAPI Microservice (`ai_service/api.py`):** REST endpoints for `/health`, `/api/v1/detect/file`, `/api/v1/detect/window`, and live `/api/v1/ws/stream` WebSocket.
- [x] **API Test Suite:** Automated contract and integration tests in `tests/test_api.py`.

### Phase 4: Risk Engine (Weeks 7–8) — Exit Check: Scenario Tests Pass (SC7) ✅
- [x] **Multimodal Risk Fusion (`ai_service/risk_engine.py`):**
  - Mathematical logit formulation: $\text{logit}(s) = b_0 + C \cdot (b_1 R_v + b_2 P_{pcta} + b_3 R_s) + b_4 R_c + b_5 R_t + b_6 (R_v \cdot R_t)$.
  - Exponential temporal smoothing: $S_t = 0.6 \cdot S_{t-1} + 0.4 \cdot s_t$.
  - Low-confidence guard (telephone/bad line cannot unilaterally hold on voice evidence alone).
  - Persistence rule ($\ge 3$ holds in last 5 windows, unless $R_t > 0.9$).
  - Spoken challenge outcome matrix (`PASS` downgrades, `FAIL` holds, `INCONCLUSIVE` retries).
- [x] **Automated Scenario Tests:** 16 passing unit tests across low-risk, high-risk, degraded line, persistence, and matrix scenarios.

### Phase 5: TB-PC Service (Weeks 6–8) — Exit Check: Challenge Round Trip Works (SC5) ✅
- [x] **Cryptographic Challenge Generation (`ai_service/tb_pc_verifier.py` & `backend/server.js`):**
  - Ephemeral HMAC-SHA256 derived phrase tied to transaction amount and single-use 60-second nonce (FR-09).
- [x] **Multi-Dimensional Response Scoring:**
  - Evaluates digit match, caller latency cadence (400ms–2400ms biological vs >2500ms on-demand clone lag or <200ms stolen replay), syllabic articulation rate, and biomechanical stability (FR-10).
  - 100% passing tests in `tests/test_tb_pc.py`.

### Phase 6: Fraud Monitoring Portal & Gateway (Weeks 5–7) — Exit Check: Upload-to-Result Works ✅
- [x] **Enterprise Spring Boot Backend (`backend/`):**
  - High-performance Java 21 Spring Boot gateway proxying to FastAPI.
  - Ephemeral HMAC-SHA256 challenge generation and single-use nonce tracking.
  - Cryptographic hash-chained audit logging anchor (`prev_hash`, `record_hash`) storing feature metadata with zero raw audio persistence.
  - Metadata purge endpoint (`DELETE /api/audit/records/{id}`) complying with FR-11 privacy rights.
- [x] **React + Vite + TypeScript Dashboard (`frontend/src/App.tsx`):**
  - Sleek FinTech dark glassmorphism portal.
  - Live sliding-window risk timeline graph with Recharts (FR-05).
  - Interactive TB-PC challenge simulator with 60-second countdown and per-check score diagnostics.
  - Hash-chained audit trail visualizer.
  - Microphone recording consent modal (FR-13).

---

## 3. Technology Stack

| Layer | Technology | Status |
|:---|:---|:---:|
| **Frontend Portal** | React + Vite + TypeScript, Recharts (Fraud Dashboard) | ✅ Active (Port 5173) |
| **Backend Gateway** | Java 21 + Spring Boot 3.4 (HMAC Challenges & Audit Ledger) | ✅ Active (Port 5000) |
| **Audio & AI Engine** | Python 3.13, PyTorch 2.14, FastAPI, Librosa, SciPy | ✅ Active (Port 8000) |
| **Testing** | pytest (19 automated tests passing) | ✅ Active |

---

## 4. Quickstart: Running the Full Stack

### 1. Launch Python AI Microservice (Port 8000)
```bash
uvicorn ai_service.api:app --reload --port 8000
```

### 2. Launch Spring Boot Backend Gateway (Port 5000)
```bash
cd backend
.\gradlew.bat bootRun
```

### 3. Launch React Frontend Portal (Port 5173)
```bash
cd frontend
npm install
npm run dev
```

### 4. Run Automated Tests
```bash
pytest -v
```

---

## 5. Scope Roadmap

- [x] **Phase 1: Define (Weeks 1–2):** Requirements, SIH verification, licences, repo, data indexing.
- [x] **Phase 2: Baseline (Weeks 3–4):** Pretrained baseline model inference, failure inspection, degradation analysis.
- [x] **Phase 3: AI API (Weeks 5–6):** FastAPI streaming service, VAD, 2 s windowed inference (0.5 s hop).
- [x] **Phase 4: Risk Engine (Weeks 7–8):** Transaction contextual fusion, low-confidence guard, decision matrix.
- [x] **Phase 5: TB-PC (Weeks 6–8):** HMAC-SHA256 spoken challenge derivation, cadence timing, and response verification.
- [x] **Phase 6: Dashboard & Audit (Weeks 5–7):** MERN portal, live risk timeline, hash-chained audit trail.
- [ ] **Phase 7: Evaluation & Viva (Weeks 8–12):** Cross-dataset evaluation, demonstration video, viva presentation slides.

---

## 6. License
Released under the [MIT License](LICENSE). See [docs/LICENSES_AND_ATTRIBUTION.md](docs/LICENSES_AND_ATTRIBUTION.md) for third-party attributions.
