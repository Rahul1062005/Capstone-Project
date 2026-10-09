# Smart India Hackathon (SIH) Problem Alignment

**Theme:** Cybersecurity and FinTech  
**Project Title:** BAFV-PCTA (Voice-Clone Fraud Shield)  
**Full Title:** Biomechanical Acoustic Feature Verification with Physiological Coupled-Trajectory Analysis for Voice-Clone Fraud Prevention  

---

## 1. Problem Statement Mapping

### The Real-World Attack Scenario (Vishing & Executive Impersonation)
- **Vishing Attack Vector:** Threat actors utilize neural text-to-speech (TTS) engines and few-shot voice cloners (e.g., ElevenLabs, Tortoise, OpenVoice, XTTS) to impersonate corporate CXOs, account holders, or family members over telephone channels.
- **Vulnerability in Financial Authorization:** Modern banking institutions and automated interactive voice response (IVR) systems rely on knowledge-based authentication (OTP, mother's maiden name) or basic voice biometrics. Deepfake audio bypasses human call agents and classical speaker verification by mimicking timbre and pitch profiles.
- **The Core Flaw in Existing Solutions:**
  - Standard ML detectors treat the problem as a single static classification task. When voice cloning models evolve or audio passes through lossy telephone codecs (AMR, G.711, 8 kHz downsampling), passive detectors collapse into high false alarm rates.
  - Furthermore, passive scoring cannot distinguish between an authorized pre-recorded audio authorized by the customer vs an adversarial replay or live attacker.

---

## 2. Solution Overview: BAFV-PCTA

BAFV-PCTA solves this via two complementary defense layers:

### Layer A: Biomechanical & Physiological Acoustic Analysis
Human vocal production is constrained by physical vocal tract biomechanics:
1. Subglottal lung pressure and continuous air-flow aerodynamics.
2. Glottal vocal fold micro-perturbations (pitch jitter, amplitude shimmer).
3. Coupled formants ($F_1, F_2, F_3$) exhibiting continuous vocal-tract geometry transitions governed by physical articulators (tongue, jaw, velum).
4. Harmonic-to-Noise Ratio (HNR) and phase-continuity properties across frame boundaries.

Generative neural vocoders (HiFi-GAN, WaveGlow, Diffusion Vocoders) generate speech frame-by-frame or via spectrogram inversion, leaving phase inconsistencies, synthetic micro-spectral artifacts, and unnatural formant transitions under temporal stress.

### Layer B: Transaction-Bound Spoken Challenge (TB-PC)
When passive confidence indicates ambiguity (`VERIFY` zone, or high transaction value):
- The system generates an un-cacheable, dynamic prompt tied cryptographically to the transaction:
  $$\text{Challenge Phrase} = \text{DerivePhrases}(\text{HMAC-SHA256}(K_{bank}, \text{TxnID} \parallel \text{Amount} \parallel \text{Nonce}))$$
- **Replay Resistance:** An attacker cannot reuse stolen audio snippets because the phrase contains transaction-specific numbers and random words generated seconds prior.
- **On-Demand Cloning Latency Exposure:** Running few-shot voice synthesis in real-time introduces computational delay (minimum 600ms - 2000ms pipeline latency), which violates human conversational timing thresholds.
- **Physiological Coupled Consistency:** The response is evaluated simultaneously for acoustic naturalness, articulation velocity, and digit correctness via local ASR.

---

## 3. Demarcation: Prototype vs. Production Banking Use
To ensure compliance with academic research standards and responsible AI practices:
- **Research Prototype:** Tested on standardized public benchmarks (ASVspoof 2019 LA, In-the-Wild) and consented volunteer recordings.
- **Advisory Role:** The system provides decision support (`ALLOW`, `VERIFY`, `HOLD FOR REVIEW`) for human bank fraud analysts. It does not unilaterally block real customer transactions.
- **Privacy First:** Raw customer audio streams are not retained long-term; only feature-level acoustic fingerprints and audit log hashes are stored.
