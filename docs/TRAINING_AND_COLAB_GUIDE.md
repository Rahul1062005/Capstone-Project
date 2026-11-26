# BAFV-PCTA Model Training & Manifold Calibration Guide

This guide explains how to train real anti-spoofing weights (`models/aasist_baseline.pt`) and calibrate empirical biomechanical/open-set distributions (`models/calibration_stats.json`) using real speech datasets.

Once those files exist under `models/`, `ai_service/api.py` loads them **automatically** on startup (or via `BAFV_CHECKPOINT_PATH` and `BAFV_CALIBRATION_PATH`). You can verify active weights anytime via `GET /health` (`"weights_source": "trained_checkpoint"`).

---

## 1. Supported Dataset Formats

### Option A: Simple Folder Layout (In-the-Wild / WaveFake / Custom Recordings)
Organize audio files (`.wav`, `.flac`, `.ogg`, `.mp3`) like this:
```text
dataset/
  bonafide/    # (or real/, human/, genuine/)
    speaker_01.wav
    ...
  spoof/       # (or fake/, synthetic/, deepfake/)
    clone_01.wav
    ...
```

### Option B: ASVspoof 2019 Logical Access (LA)
Use the standard protocol files directly (e.g., `ASVspoof2019.LA.cm.train.trn.txt` and `ASVspoof2019.LA.cm.dev.trl.txt`) pointing to the `flac/` directory.

---

## 2. Training on Google Colab (Recommended Free T4 GPU)

1. Open Google Colab (`Runtime -> Change runtime type -> T4 GPU`).
2. Clone or upload your repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download your dataset (e.g., ASVspoof 2019 LA or In-the-Wild via Kaggle/Drive).
4. Run training:

   **Folder dataset:**
   ```bash
   python scripts/train_detector.py \
     --data-dir /content/dataset \
     --epochs 20 \
     --batch-size 32 \
     --telephony-aug-prob 0.25 \
     --output-checkpoint models/aasist_baseline.pt
   ```

   **ASVspoof 2019 LA protocol:**
   ```bash
   python scripts/train_detector.py \
     --asvspoof-protocol /content/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
     --audio-root /content/LA/ASVspoof2019_LA_train/flac \
     --dev-protocol /content/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
     --dev-audio-root /content/LA/ASVspoof2019_LA_dev/flac \
     --epochs 20 \
     --batch-size 32 \
     --telephony-aug-prob 0.25 \
     --output-checkpoint models/aasist_baseline.pt
   ```

5. Run manifold & generator cluster calibration:
   ```bash
   python scripts/calibrate_manifold.py \
     --bonafide-dir /content/dataset/bonafide \
     --spoof-root /content/dataset/spoof_by_generator \
     --checkpoint models/aasist_baseline.pt \
     --output models/calibration_stats.json \
     --max-files 500
   ```
   *(If you don't have generator-separated subfolders, omit `--spoof-root` to calibrate the human manifold first.)*

6. Download `models/aasist_baseline.pt`, `models/training_metrics.json`, and `models/calibration_stats.json` into:
   `D:\Study_Time\Projects\Capstone Project\models\`

---

## 3. Verifying Loaded Weights Locally

Start the AI service (`uvicorn ai_service.api:app --reload --port 8000`) and visit `http://127.0.0.1:8000/health`:
- `"weights_source": "trained_checkpoint"`
- `"checkpoint_loaded": true`
