# BAFV-PCTA: Project Specification & Requirements

**Voice-Clone Fraud Shield: Biomechanical Acoustic Feature Verification with Physiological Coupled-Trajectory Analysis**  
*Transaction-Bound Physiological Challenge Verification for Voice-Clone Fraud Prevention*  
**Document Version:** 1.0 (Phase 1 Baseline & Architecture Alignment)  
**SIH Theme:** Cybersecurity and FinTech  

---

## 1. Executive Summary & Problem Formulation
Voice cloning and synthetic speech generators (neural TTS, instant voice cloning, real-time voice conversion) present an acute threat to voice-authorized financial transactions, call center authentication, and high-value wire transfers (Vishing attacks).

Traditional anti-spoofing detectors rely solely on passive acoustic models that:
1. Degrade significantly over telephone channels (8 kHz band-limiting, AMR/G.711 codecs, packet jitter).
2. Suffer from out-of-domain failures when confronted with unseen generative neural vocoders.
3. Lack transaction context, leading to either reckless approvals or unacceptable false alarms for real bank customers.

### The BAFV-PCTA Innovation
BAFV-PCTA introduces a dual-defense paradigm:
1. **Passive Windowed Acoustic & Physiological Trajectory Analysis:** 
   - 2.0-second sliding windows with 0.5-second hop.
   - Pretrained anti-spoofing baseline model coupled with Biomechanical Acoustic Feature Verification (F0/formant dynamics, phase-continuity metrics, vocal-tract resonance, jitter/shimmer/HNR, and PCVS-lite).
2. **Active Transaction-Bound Spoken Challenge (TB-PC):**
   - In ambiguous/suspicious cases (`VERIFY` state), the system generates an ephemeral, transaction-bound challenge sentence derived from a cryptographic HMAC-SHA256 of transaction details (amount, account, nonce).
   - Defeats replay attacks, cached clone libraries, and exposes computational latency (>800ms) or acoustic mismatch in on-demand synthesis engines.
3. **Contextual Risk Engine:**
   - Fuses audio evidence, channel confidence metrics, and simulated transaction velocity/amount into a human analyst recommendation: `ALLOW`, `VERIFY`, or `HOLD FOR REVIEW`.

---

## 2. Functional Requirements Traceability Matrix

| ID | Requirement Specification | Phase | Status |
|:---|:---|:---:|:---:|
| **FR-01** | Accept supported audio (WAV first, auto-transcoded with FFmpeg). Reject malformed/oversized files safely. | Phase 1 & 2 | ✅ Implemented |
| **FR-02** | Resample and normalize audio to model target standard (16 kHz mono PCM float32). | Phase 2 | ✅ Implemented |
| **FR-03** | Output real model predictions with explicit model name, checkpoint version, and device provenance (never hardcoded). | Phase 2 | ✅ Implemented |
| **FR-04** | Show score interpretation, quality warnings, channel confidence, and clear probabilistic disclaimers. | Phase 2 | ✅ Implemented |
| **FR-05** | Analyze continuous speech in 2.0 s windows with 0.5 s hop; stream updates to a live risk timeline. | Phase 3 | Planned |
| **FR-06** | Allow simulation of transaction context: amount (INR/USD), beneficiary status (known/new), velocity. | Phase 4 | Planned |
| **FR-07** | Multimodal risk fusion combining audio evidence, channel SNR/confidence, and mock transaction parameters. | Phase 4 | Planned |
| **FR-08** | Recommend decision states: `ALLOW`, `VERIFY`, or `HOLD FOR REVIEW` (advisory only for bank fraud analysts). | Phase 4 | Planned |
| **FR-09** | TB-PC Challenge Generation: derive dynamic phrases from HMAC-SHA256 of transaction context + single-use nonce. | Phase 5 | Planned |
| **FR-10** | Verify TB-PC challenge response: latency, articulation rate, sustained vowel stability, and digit correctness. | Phase 5 | Planned |
| **FR-11** | Scan metadata logging and deletion support; raw customer audio is not permanently stored without consent. | Phase 6 | Planned |
| **FR-12** | Explicit error handling on decoding/inference failures; fail-secure with no fabricated predictions. | Phase 2 | ✅ Implemented |
| **FR-13** | Authentication, rate limiting, and explicit user consent banner before microphone recording. | Phase 6 | Planned |

---

## 3. Mathematical Formulation of the Risk Engine (Scope Section 9)

$$\text{logit}(s) = b_0 + C \cdot (b_1 R_v + b_2 P_{pcta} + b_3 R_s) + b_4 R_c + b_5 R_t + b_6 (R_v \cdot R_t)$$

Where:
- $R_v$: Raw model spoof score ($0.0 = \text{genuine}, 1.0 = \text{spoof}$)
- $P_{pcta}$: Physiological Coupled-Trajectory anomaly score
- $R_s$: Sustained-feature stability risk
- $C$: Channel confidence index ($0.0 \le C \le 1.0$)
- $R_c$: Channel degradation penalty
- $R_t$: Simulated transaction risk ($f(\text{amount}, \text{velocity}, \text{beneficiary status})$)

Temporal smoothing across consecutive time windows:
$$S_t = 0.6 \cdot S_{t-1} + 0.4 \cdot s_t$$

### Decision Boundaries
- **$\mathbf{S_t < 0.35}$**: `ALLOW`
- **$\mathbf{0.35 \le S_t < 0.75}$**: `VERIFY` (Triggers TB-PC Spoken Challenge)
- **$\mathbf{S_t \ge 0.75}$**: `HOLD FOR REVIEW` (Analyst intervention, requires persistence in $\ge 3$ of last 5 windows unless $R_t > 0.9$).

---

## 4. Phase Plan (Current Milestone: Phase 1 & Phase 2)
- **Phase 1 (Weeks 1-2) — Define:** System specifications, SIH problem mapping, repo setup, licensing, and dataset indexer.
- **Phase 2 (Weeks 3-4) — Baseline:** Baseline pretrained anti-spoofing model inference, real model outputs, validation pipeline, and failure inspection.
- *Phases 3-12 (AI API, UI, TB-PC, Risk Engine, Evaluation, Hardening) are preserved as future milestones as per user directive.*
