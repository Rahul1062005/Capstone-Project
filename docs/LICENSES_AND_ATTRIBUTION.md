# Licenses and Academic Attribution

This project is released under the **MIT License**. Third-party datasets, baseline models, and libraries used across the research and engineering pipeline are attributed below in compliance with their respective academic and open-source licenses.

---

## 1. Baseline Model & Algorithmic Architectures
- **AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks):**
  - Reference: Jung, J., Heo, H., Tak, H., Shim, H., & Chung, J. S. (2022). *AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks*. ICASSP 2022.
  - Repository: [github.com/clovaai/aasist](https://github.com/clovaai/aasist)
  - License: Apache License 2.0 / Academic Research Use.

- **RawNet2 Baseline:**
  - Reference: Tak, H., Patino, J., Todisco, M., Nautsch, A., Evans, N., & Larcher, A. (2021). *End-to-End anti-spoofing with RawNet2*. Interspeech 2021.
  - License: MIT License.

- **SSL Speech Representations (Wav2Vec 2.0 / WavLM):**
  - Reference: Baevski et al., *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*, NeurIPS 2020.
  - License: MIT License.

---

## 2. Evaluation Datasets
- **ASVspoof 2019 Logical Access (LA) Dataset:**
  - Organizer: ASVspoof Consortium (EURECOM, University of Eastern Finland, etc.)
  - Purpose: Official academic anti-spoofing benchmark for synthetic speech and voice conversion detection.
  - Terms: Academic evaluation, strictly non-commercial research.

- **In-The-Wild Audio Deepfake Dataset:**
  - Reference: Müller, N., Dieckmann, F., Timmermann, P., et al. (2022). *Does Audio Deepfake Detection Generalize?*. Interspeech 2022.
  - Purpose: Cross-dataset out-of-domain robustness evaluation.

- **Consented Volunteer Challenge Dataset (TB-PC):**
  - Collected exclusively from consenting participants across varied accents/languages under academic protocol.
  - No synthetic data is generated without explicit consent of the target voice contributor.

---

## 3. Core Software Libraries
- **PyTorch:** BSD 3-Clause License.
- **Librosa:** ISC License.
- **SoundFile:** BSD 3-Clause License.
- **SciPy & NumPy:** BSD 3-Clause License.
- **FastAPI:** MIT License.
- **Express.js & React:** MIT License.
