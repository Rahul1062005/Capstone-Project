# VoiceShield (BAFV-PCTA): AI Deepfake & Voice-Clone Fraud Shield

### Biomechanical Acoustic Feature Verification with Physiological Coupled-Trajectory Analysis

> **Live Demo:** [https://voiceshield-fintech.vercel.app](https://voiceshield-fintech.vercel.app)  
> **SIH Problem Alignment:** `SIH26104` — Cybersecurity & FinTech  
> **Status:** Phases 1–6 Complete & Live Deployed · Phase 7 (Large-Scale Dataset Training & Viva Evaluation) Ready  
> **Intended Use:** Academic research prototype and advisory decision-support console for bank fraud analysts (not for autonomous production account freezes).

---

## 1. Problem Formulation & Solution

### The Threat: Vishing & Voice-Clone Impersonation
Fraudsters use neural text-to-speech (TTS) and few-shot voice cloning tools to impersonate account holders or executives over phone channels, tricking victims and call agents into authorizing high-value transfers. Traditional passive detectors struggle when audio passes through lossy 8 kHz telephone codecs or comes from unseen generative vocoders.

### The BAFV-PCTA Architecture
Human speech is constrained by physical vocal-tract biomechanics—glottal micro-perturbations (jitter/shimmer), continuous formant resonances ($F_1, F_2, F_3$), and subglottal airflow coordination—which frame-by-frame neural vocoders often distort.

VoiceShield combines three layers:
1. **Passive Acoustic & Physiological Analysis (`ai_service/`):**
   - **Baseline Detector:** PyTorch spectro-temporal CNN + attentive statistical pooling (`SpectroTemporalBaselineNetwork`), with automatic checkpoint loading from `models/aasist_baseline.pt`.
   - **BAFV & PCTA Engine:** Extracts pitch (`F0`), jitter, shimmer, HNR, LPC formants (`F1–F3`), **PCVS** (6 coupled trajectory pairs + Mahalanobis manifold distance), **RPCI**, and **PTVA** articulator velocity bounds.
   - **Open-Set Detector & Telephony Confidence (`TCS`):** Distinguishes known generator families vs. unseen zero-day synthetic audio and measures 8 kHz narrowband channel degradation.
2. **Dynamic Risk Fusion Engine (`DRFE`) & Spoken Challenge (`TB-PC`):**
   - Fuses voice spoof probability, PCTA score, channel confidence $C$, and transaction context (INR amount, new beneficiary, velocity) using an amount-aware adaptive threshold $\tau_L(a)$, low-confidence channel guard, and 3-of-5 persistence rule into `ALLOW`, `VERIFY`, or `HOLD_FOR_REVIEW`.
   - Issues ephemeral 60-second **HMAC-SHA256** spoken challenges tied to transaction details and single-use nonces.
3. **Enterprise Gateway & Operations Console (`backend/` + `frontend/`):**
   - **Java 21 Spring Boot Gateway:** Proxies analysis to FastAPI, generates cryptographic challenges, and records a **SHA-256 hash-chained audit ledger** with zero raw audio retention and GDPR/DPDP metadata purge (`DELETE /api/audit/records/{id}`).
   - **React + TypeScript Console:** Light/Dark institutional analyst dashboard featuring Call Inspector, 2.0s Windowed Risk Timeline, Voice Challenge simulator, and Audit Ledger.

---

## 2. Live Deployment

| Component | Hosting | URL / Notes |
|:---|:---|:---|
| **Frontend Console** | Vercel | [https://voiceshield-fintech.vercel.app](https://voiceshield-fintech.vercel.app) |
| **Spring Boot Gateway** | Render (Docker) | [Health Check](https://bafv-pcta-gateway.onrender.com/health) |
| **Python AI Microservice** | Render (Docker, CPU PyTorch) | [Health Check](https://bafv-pcta-ai.onrender.com/health) |

> **Free-Tier Cold Start Note:** Both Render backend services sleep after ~15 minutes of inactivity. On the first visit after idle, please allow 45–60 seconds for the backend containers to wake up.

Full deployment instructions are in [`docs/DEPLOYMENT_GUIDE.md`](docs/DEPLOYMENT_GUIDE.md).

---

## 3. Technology Stack

| Layer | Technology |
|:---|:---|
| **Frontend Portal** | React 19, Vite, TypeScript, Recharts, Lucide Icons (Light & Matte Dark themes) |
| **Backend Gateway** | Java 21, Spring Boot 3.4.3, Gradle 9.3.1 (HMAC-SHA256 Challenges & SHA-256 Audit Ledger) |
| **Audio & AI Engine** | Python, PyTorch, FastAPI, Uvicorn, Librosa, SciPy, SoundFile, scikit-learn |
| **Automated Checks** | `pytest` (23 tests), Gradle JUnit (`test`), `tsc -b && vite build`, `oxlint` |

---

## 4. Repository Structure

```text
├── ai_service/                 # FastAPI microservice, PyTorch baseline, BAFV-PCTA, OGD/TCS, DRFE, TB-PC
├── backend/                    # Java 21 Spring Boot gateway, HMAC challenges, SHA-256 audit ledger, Dockerfile
├── frontend/                   # React + Vite + TypeScript analyst workstation, sample WAVs, vercel.json
├── scripts/                    # Dataset indexing, baseline inspection, PyTorch training & manifold calibration
├── tests/                      # 23 automated pytest unit & FastAPI integration tests
├── data/                       # Indexed synthetic benchmark WAVs, manifests, baseline predictions
├── docs/                       # Specifications, SIH alignment, baseline report, training & deployment guides
├── render.yaml                 # Render Blueprint for bafv-pcta-ai and bafv-pcta-gateway
├── requirements.txt            # Python dependencies
└── conftest.py                 # Pytest path and Numba cache configuration
```

---

## 5. Quickstart: Running Locally

Open three terminals in the repository root:

### 1. Launch Python AI Microservice (Port `8000`)
```bash
pip install -r requirements.txt
uvicorn ai_service.api:app --reload --port 8000
```

### 2. Launch Spring Boot Gateway (Port `5000`)
```bash
cd backend
.\gradlew.bat bootRun
```

### 3. Launch React Frontend Console (Port `5173`)
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Sample `.wav` files for manual upload testing (including `corrupted_payload.wav` for FR-13 fail-secure verification) are in `data/test_samples/`.

### 4. Run Automated Tests & Builds
```bash
# Python AI & API tests (23 tests)
pytest -v

# Spring Boot backend build & tests
.\backend\gradlew.bat -p backend test --no-daemon

# Frontend type-check, production build, and lint
npm --prefix frontend run build
npm --prefix frontend run lint
```

---

## 6. Model Training & Calibration (Phase 7)

By default, if no trained checkpoint is present in `models/aasist_baseline.pt`, `BaselineAntiSpoofDetector` falls back to deterministic seed initialization (`weights_source: "untrained_seed_init"` on `/health`) so the full pipeline can be demonstrated safely without fabricated predictions.

To train on real datasets (**ASVspoof 2019 LA**, **In-the-Wild**, or custom `bonafide/` + `spoof/` folders) and calibrate the BAFV/Open-Set distributions:
- Training script: [`scripts/train_detector.py`](scripts/train_detector.py)
- Calibration script: [`scripts/calibrate_manifold.py`](scripts/calibrate_manifold.py)
- Step-by-step Colab & local instructions: [`docs/TRAINING_AND_COLAB_GUIDE.md`](docs/TRAINING_AND_COLAB_GUIDE.md)

---

## 7. Roadmap & Documentation

- [x] **Phase 1: Define** — Requirements, SIH alignment, licensing, data indexing ([`docs/PROJECT_SPECIFICATION.md`](docs/PROJECT_SPECIFICATION.md), [`docs/SIH_PROBLEM_ALIGNMENT.md`](docs/SIH_PROBLEM_ALIGNMENT.md)).
- [x] **Phase 2: Baseline** — Audio validation, baseline detector, degradation inspection ([`docs/PHASE_2_BASELINE_REPORT.md`](docs/PHASE_2_BASELINE_REPORT.md)).
- [x] **Phase 3: AI API & PCTA** — FastAPI REST/WebSocket service, VAD, 2.0s windowed analysis, BAFV-PCTA, OGD, TCS.
- [x] **Phase 4: Risk Engine** — DRFE multimodal fusion, amount-aware adaptive threshold, low-confidence guard, persistence rule.
- [x] **Phase 5: TB-PC Challenge** — HMAC-SHA256 spoken challenge generation and latency/articulation/content verification.
- [x] **Phase 6: Enterprise Gateway, UI & Cloud Deployment** — Java 21 Spring Boot gateway, SHA-256 audit ledger, Light/Dark workstation UI, live on Vercel & Render.
- [ ] **Phase 7: Evaluation & Viva** — Train weights on real benchmark datasets (ASVspoof 2019 LA / In-the-Wild), cross-dataset EER/min-tDCF evaluation, demo video, and viva slides.

---

## 8. License

Released under the [MIT License](LICENSE). See [`docs/LICENSES_AND_ATTRIBUTION.md`](docs/LICENSES_AND_ATTRIBUTION.md) for dataset and library attributions.
