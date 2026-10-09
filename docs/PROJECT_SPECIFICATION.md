# BAFV-PCTA: Project Specification & Requirements

**Voice-Clone Fraud Shield: Biomechanical Acoustic Feature Verification with Physiological Coupled-Trajectory Analysis for Real-Time Voice-Clone Fraud Prevention**  
**Mapped SIH Problem:** `SIH26104`: AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks  
**Organisation / Department:** All India Council for Technical Education (AICTE) - Cyber Security Cell  
**Category & Theme:** Software | Blockchain and Cybersecurity  
**Application Focus:** Banking, financial fraud prevention and telecom / contact-centre security  
**Document Alignment:** Formally aligned with the 18-page BAFV-PCTA Project Report and Scope Document  

---

## 1. Executive Summary & Problem Formulation
Voice cloning and synthetic speech generators (neural TTS, instant voice cloning, real-time voice conversion) present an acute threat to voice-authorized financial transactions, call center authentication, and high-value wire transfers (Vishing attacks).

Traditional anti-spoofing detectors rely solely on passive acoustic models that:
1. Degrade significantly over telephone channels (8 kHz band-limiting, AMR/G.711 codecs, packet jitter).
2. Suffer from out-of-domain failures when confronted with unseen generative neural vocoders.
3. Lack transaction context, leading to either reckless approvals or unacceptable false alarms for real bank customers.

### The BAFV-PCTA Innovation
BAFV-PCTA introduces a physiological coupling defense:
1. **Biomechanical Acoustic Feature Verification (BAFV):** Extracts 4 parallel streaming measurements tied to human speech production (source/phase, vocal-tract $F_0-F_3$, respiration cues, prosody/energy; 25 ms frames, 10 ms hop).
2. **Physiological Coupled-Trajectory Analysis (PCTA):** Checks whether acoustic measurements move synchronously as human vocal-tract biology dictates:
   - **PCVS:** Physiological Coupled-Trajectory Variance Score ($P_{pcta}$).
   - **RPCI:** Respiration-to-Phonation Coupling Index (breath-to-voicing timing delay $\Delta_k$).
   - **PTVA:** Phoneme-Transition Velocity Anomaly (articulatory speed limits).
3. **Telephony Confidence Score (TCS):** $C = \text{sigmoid}(w_1 \text{SNR}_n + w_2 \text{BW} + w_3(1 - \text{loss}) + w_4 \min(1, T_{\text{speech}}/T_0) + w_5 \text{codec}_{\text{score}} + b)$.
4. **Dynamic Risk Fusion Engine (DRFE):** Fuses acoustic evidence, channel confidence $C$, call context $R_c$, and financial transaction risk $R_t$ into an explainable recommendation (`ALLOW`, `VERIFY`, `HOLD FOR REVIEW`).
5. **Adaptive Step-Up Verification:** Dynamic challenge selection (random digit repetition, out-of-band mobile app approval, cleaner line callback, or immediate hold).
6. **Enterprise Java 21 Spring Boot Gateway:** Orchestrator handling session state, transaction context, policy rules, and cryptographic hash-chained audit logging (no raw audio retained).

---

## 2. Functional Requirements Traceability Matrix

| ID | Requirement Specification | Tier | Status |
|:---|:---|:---:|:---:|
| **FR-01** | Ingest call audio as 20 ms packets over WebSocket/gRPC and maintain a 2 s ring buffer with 0.5 s hop. | Tier 1 | ✅ Implemented |
| **FR-02** | Run VAD, resampling to 16 kHz (native 8 kHz kept for statistics), normalisation; output channel confidence $C$. | Tier 1 | ✅ Implemented |
| **FR-03** | Extract four feature streams and compute PCVS, RPCI and PTVA; mark unmeasurable cues as missing. | Tier 1 | ✅ Implemented |
| **FR-04** | Compute open-set voice risk $R_v$ and report attack type or label unknown synthetic. | Tier 1 | ✅ Implemented |
| **FR-05** | Compute speaker risk against enrolled voiceprint or within-call drift (dropped when no voiceprint). | Tier 2 | ✅ Implemented |
| **FR-06** | Fetch call and transaction context (simulated) and compute $R_c$ and $R_t$. | Tier 1 | ✅ Implemented |
| **FR-07** | Fuse evidence in DRFE with smoothing ($\lambda = 0.6$), persistence rule and low-confidence guard. | Tier 1 | ✅ Implemented |
| **FR-08** | Produce `ALLOW` / `VERIFY` / `HOLD` and alert an analyst on `HOLD`; select adaptive challenge on `VERIFY`. | Tier 1 | ✅ Implemented |
| **FR-09** | Store evidence report per decision: top contributing terms, violated coupling pairs, channel confidence. | Tier 1 | ✅ Implemented |
| **FR-10** | Provide REST and WebSocket interfaces with documented API contracts. | Tier 1 | ✅ Implemented |
| **FR-11** | Dashboard shows live risk timeline, coupling heat-map, evidence panel and analyst controls. | Tier 1 | ✅ Implemented |
| **FR-12** | Write hash-chained audit records; never store raw audio unless policy explicitly requires it. | Tier 1 | ✅ Implemented |
| **FR-13** | Show explicit errors on failure; never fabricate a prediction. | Tier 1 | ✅ Implemented |
| **FR-14** | Role-based access (analyst, administrator, auditor) with all access logged. | Tier 1 | ✅ Implemented |

---

## 3. Mathematical Formulations (Report Section 8)

### 3.1 PCTA and Physiological Coupling Violation Score (PCVS)
$$c_{ij} = \left(\max_\tau r_{ij}(\tau), \operatorname{argmax}_\tau r_{ij}(\tau)\right)$$
$$D_{ij} = \sqrt{(c_{ij} - \mu_{ij})^T \Sigma_{ij}^{-1} (c_{ij} - \mu_{ij})} + \beta \cdot R_{ij}$$
$$\text{PCVS} = \sum_{(i,j) \in E} w_{ij} D_{ij}, \quad P_{pcta} = \operatorname{sigmoid}(a \cdot \text{PCVS} + b)$$

### 3.2 Respiration-Phonation Coupling Index (RPCI)
$$\text{RPCI} = \frac{1}{K} \sum_{k=1}^K \sqrt{(\Delta_k - \mu_h)^T \Sigma_h^{-1} (\Delta_k - \mu_h)} + \lambda \cdot \rho$$
$$\rho = \max\left(0, \frac{L_{\text{run}} - L_{\max}}{L_{\max}}\right)$$

### 3.3 Dynamic Risk Fusion Engine (DRFE)
$$\text{logit}(s_t) = \beta_0 + C \cdot (\beta_1 R_v + \beta_2 P_{pcta} + \beta_3 R_s) + \beta_4 R_c + \beta_5 R_t + \beta_6 (R_v \cdot R_t)$$
$$S_t = \lambda S_{t-1} + (1 - \lambda) s_t \quad (\lambda \approx 0.6)$$

### 3.4 Amount-Aware Adaptive Threshold
$$\tau_L(a) = \tau_{L0} \cdot \left[1 - \kappa \min\left(1, \frac{\ln(1 + a/a_0)}{\ln(1 + a_{\max}/a_0)}\right)\right], \quad \kappa \approx 0.40$$

### 3.5 Decision Rules
- **$\mathbf{S_t < \tau_L(a)}$**: `ALLOW`
- **$\mathbf{\tau_L(a) \le S_t < 0.75}$**: `VERIFY` (Adaptive Step-Up Verification)
- **$\mathbf{S_t \ge 0.75}$**: `HOLD FOR REVIEW`
- **Persistence Rule:** Escalate to `HOLD` only if the hold band is reached in at least 3 of the last 5 windows, unless $R_t > 0.9$ (immediate hold).
- **Low-Confidence Guard:** If $C < 0.5$ and $\max(R_v, P_{pcta}) \ge 0.5$, outcome is at least `VERIFY`; `HOLD` cannot come from voice evidence alone.

---

## 4. 16-Week Roadmap & Priority Tiers

| Tier | Items Included | Status |
|:---|:---|:---:|
| **Tier 1 (Must Deliver)** | Audio pipeline (VAD, windowing, features), baseline AASIST/wav2vec2, PCVS, TCS, DRFE with guard and persistence rules, Java 21 Spring Boot orchestrator, REST/WebSocket APIs, React dashboard, hash-chained audit log, RQ1, RQ3, RQ5 | ✅ Complete |
| **Tier 2 (Should Deliver)** | RPCI, PTVA, open-set detector with LOGO (RQ2), ECAPA speaker consistency, adaptive challenge policy, SIP audio-fork demo, per-language evaluation (RQ4) | ✅ Active |
| **Tier 3 (Stretch)** | gRPC streaming, Kafka event stream, Redis rolling state, ledger anchoring (Hyperledger Fabric), Prometheus/Grafana, full multilingual UI | Future Scope |
