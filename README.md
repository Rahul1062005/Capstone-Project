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
- [x] **Manifest Inference Runner (`scripts/run_baseline_inference.py`):**
  - Batch executes baseline model over indexed datasets and outputs structured predictions to `data/predictions/baseline_predictions.json`.
- [x] **Failure Inspection & Evaluation (`scripts/inspect_baseline_failures.py`):**
  - Evaluates performance across clean vs 8 kHz telephone channel degradation (SC3, SC4).
  - Full report generated at [`docs/PHASE_2_BASELINE_REPORT.md`](docs/PHASE_2_BASELINE_REPORT.md).
- [x] **Automated Test Suite (`pytest`):**
  - 100% pass on 6 unit tests covering preprocessing, validation, failure modes, latency, and metadata contracts.

---

## 3. Technology Stack

| Layer | Technology | Status |
|:---|:---|:---:|
| **Audio & ML Service** | Python 3.13, PyTorch 2.14, Librosa 1.0, SoundFile, SciPy | ✅ Active |
| **Testing** | pytest, pytest-asyncio | ✅ Active |
| **Backend API** | Node.js + Express (JWT, HMAC-SHA256 TB-PC generation) | Phase 5–6 |
| **Frontend Portal** | React + Vite + TypeScript, Recharts (Fraud Dashboard) | Phase 5–7 |
| **Database** | MongoDB (Metadata, scan scores, hash-chained audit logs) | Phase 6 |

---

## 4. Quickstart & Verification

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Benchmark Audio Fixtures
```bash
python scripts/generate_benchmark_audio.py
```

### 3. Run Dataset Indexer (Phase 1 Exit Check)
```bash
python scripts/index_data.py
```
Outputs manifests in `data/manifests/dataset_manifest.json` and `.csv`.

### 4. Run Baseline Inference (Phase 2 Exit Check)
```bash
python scripts/run_baseline_inference.py
```
Produces real model predictions in `data/predictions/baseline_predictions.json`.

### 5. Inspect Failures & Channel Degradation
```bash
python scripts/inspect_baseline_failures.py
```
Generates comprehensive analysis in `docs/PHASE_2_BASELINE_REPORT.md`.

### 6. Run Unit Tests
```bash
pytest -v
```

---

## 5. Scope Roadmap

- [x] **Phase 1: Define (Weeks 1–2):** Requirements, SIH verification, licences, repo, data indexing.
- [x] **Phase 2: Baseline (Weeks 3–4):** Pretrained baseline model inference, failure inspection, degradation analysis.
- [ ] **Phase 3: AI API (Weeks 5–6):** FastAPI streaming service, VAD, 2 s windowed inference (0.5 s hop).
- [ ] **Phase 4: Risk Engine (Weeks 7–8):** Transaction contextual fusion, low-confidence guard, decision matrix.
- [ ] **Phase 5: TB-PC (Weeks 6–8):** HMAC-SHA256 spoken challenge derivation, ASR digit verification.
- [ ] **Phase 6: Dashboard & Audit (Weeks 5–7):** MERN portal, live risk timeline, hash-chained audit trail.
- [ ] **Phase 7: Evaluation & Viva (Weeks 8–12):** EER, ROC-AUC, cross-dataset evaluation, demonstration video.

---

## 6. License
Released under the [MIT License](LICENSE). See [docs/LICENSES_AND_ATTRIBUTION.md](docs/LICENSES_AND_ATTRIBUTION.md) for third-party attributions.
