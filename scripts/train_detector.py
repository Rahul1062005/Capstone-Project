"""Training Pipeline for BAFV-PCTA SpectroTemporalBaselineNetwork.

Supports:
1. Folder datasets: <data_dir>/{bonafide|real}/*.wav and {spoof|fake}/*.wav
   (works with In-the-Wild, WaveFake, or custom recordings).
2. ASVspoof 2019 LA protocol files + FLAC/WAV root directory.

Saves the best PyTorch state_dict checkpoint to models/aasist_baseline.pt
and evaluation metrics (Loss, Accuracy, ROC-AUC, EER) to models/training_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_service.audio_processor import AudioProcessingError, AudioProcessor
from ai_service.baseline_detector import BaselineAntiSpoofDetector

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}


def collect_folder_samples(data_dir: Path) -> List[Tuple[Path, int]]:
    """Collect (path, label) pairs from bonafide/real (0) and spoof/fake (1) subfolders."""
    label_dirs = {
        0: ("bonafide", "real", "human", "genuine"),
        1: ("spoof", "fake", "synthetic", "deepfake"),
    }
    samples: List[Tuple[Path, int]] = []
    for label, names in label_dirs.items():
        for name in names:
            sub = data_dir / name
            if not sub.exists() or not sub.is_dir():
                continue
            for file_path in sorted(sub.rglob("*")):
                if file_path.is_file() and file_path.suffix.lower() in AUDIO_EXTENSIONS:
                    samples.append((file_path, label))
    return samples


def collect_asvspoof_samples(protocol_path: Path, audio_root: Path) -> List[Tuple[Path, int]]:
    """Parse an ASVspoof 2019 LA CM protocol file and resolve matching audio files."""
    samples: List[Tuple[Path, int]] = []
    with protocol_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            utt_id = parts[1]
            label_str = parts[-1].lower()
            if label_str not in ("bonafide", "spoof"):
                continue
            label = 0 if label_str == "bonafide" else 1
            for ext in (".flac", ".wav"):
                candidate = audio_root / f"{utt_id}{ext}"
                if candidate.exists():
                    samples.append((candidate, label))
                    break
    return samples


def apply_telephony_augmentation(
    waveform: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate 8 kHz narrowband telephone channel + mild additive noise."""
    nyquist = 0.5 * sample_rate
    cutoff = min(3400.0 / nyquist, 0.95)
    sos = signal.butter(4, cutoff, btype="low", output="sos")
    filtered = signal.sosfilt(sos, waveform).astype(np.float32)

    snr_db = float(rng.uniform(12.0, 24.0))
    sig_power = float(np.mean(filtered ** 2)) + 1e-9
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=len(filtered)).astype(np.float32)
    aug = filtered + noise
    max_abs = float(np.max(np.abs(aug))) if len(aug) > 0 else 0.0
    if max_abs > 0.0:
        aug = (aug / max_abs * 0.95).astype(np.float32)
    return aug


def fix_waveform_length(
    waveform: np.ndarray,
    target_samples: int,
    training: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pad or crop 1D waveform to a fixed length for batched training."""
    n = len(waveform)
    if n == target_samples:
        return waveform
    if n < target_samples:
        pad = target_samples - n
        return np.pad(waveform, (0, pad), mode="constant")
    if training:
        start = int(rng.integers(0, n - target_samples + 1))
    else:
        start = (n - target_samples) // 2
    return waveform[start : start + target_samples]


class AntiSpoofAudioDataset(Dataset):
    """PyTorch Dataset extracting normalized 80-bin Log-Mel tensors on the fly."""

    def __init__(
        self,
        samples: Sequence[Tuple[Path, int]],
        feature_extractor: BaselineAntiSpoofDetector,
        clip_seconds: float = 3.0,
        training: bool = True,
        telephony_aug_prob: float = 0.0,
        seed: int = 2026,
    ) -> None:
        self.samples = list(samples)
        self.processor = AudioProcessor(target_sample_rate=feature_extractor.TARGET_SR)
        self.feature_extractor = feature_extractor
        self.target_samples = int(clip_seconds * feature_extractor.TARGET_SR)
        self.training = training
        self.telephony_aug_prob = telephony_aug_prob
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        try:
            waveform, _ = self.processor.load_and_preprocess(path)
        except AudioProcessingError:
            waveform = np.zeros(self.target_samples, dtype=np.float32)

        if self.training and self.telephony_aug_prob > 0.0 and self.rng.random() < self.telephony_aug_prob:
            waveform = apply_telephony_augmentation(
                waveform, self.feature_extractor.TARGET_SR, self.rng
            )

        fixed = fix_waveform_length(waveform, self.target_samples, self.training, self.rng)
        features = self.feature_extractor.extract_features(fixed).squeeze(0).cpu()  # [1, 80, T]
        return features, label


def compute_eer(y_true: np.ndarray, spoof_scores: np.ndarray) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER) and ROC-AUC from binary labels (1=spoof) and scores."""
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5
    fpr, tpr, _ = roc_curve(y_true, spoof_scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    auc = float(roc_auc_score(y_true, spoof_scores))
    return eer, auc


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate validation loss, accuracy, ROC-AUC, and EER."""
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_labels: List[int] = []
    all_spoof_probs: List[float] = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels_t = labels.to(device, dtype=torch.long)
            logits = model(inputs)
            loss = criterion(logits, labels_t)
            probs = F.softmax(logits, dim=1)[:, 1]

            batch_size = labels_t.size(0)
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
            all_labels.extend(labels_t.cpu().tolist())
            all_spoof_probs.extend(probs.cpu().tolist())

    y_true = np.array(all_labels, dtype=np.int64)
    y_score = np.array(all_spoof_probs, dtype=np.float64)
    preds = (y_score >= 0.5).astype(np.int64)
    acc = float(np.mean(preds == y_true)) if len(y_true) > 0 else 0.0
    eer, auc = compute_eer(y_true, y_score)

    return {
        "val_loss": round(total_loss / max(1, total_items), 4),
        "val_accuracy": round(acc, 4),
        "val_roc_auc": round(auc, 4),
        "val_eer": round(eer, 4),
    }


def train_pipeline(
    train_samples: Sequence[Tuple[Path, int]],
    val_samples: Optional[Sequence[Tuple[Path, int]]] = None,
    output_checkpoint: Path = PROJECT_ROOT / "models" / "aasist_baseline.pt",
    metrics_out: Path = PROJECT_ROOT / "models" / "training_metrics.json",
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    clip_seconds: float = 3.0,
    telephony_aug_prob: float = 0.25,
    val_fraction: float = 0.2,
    seed: int = 2026,
) -> Dict[str, object]:
    """Train SpectroTemporalBaselineNetwork and save best checkpoint + metrics."""
    if len(train_samples) < 4:
        raise ValueError(f"Need at least 4 training samples; got {len(train_samples)}.")

    if val_samples is None:
        labels = np.array([lbl for _, lbl in train_samples])
        if len(np.unique(labels)) < 2 or len(train_samples) < 6:
            split_idx = max(1, int(len(train_samples) * (1.0 - val_fraction)))
            tr_list = list(train_samples[:split_idx])
            va_list = list(train_samples[split_idx:]) or list(train_samples[:1])
        else:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
            train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))
            tr_list = [train_samples[i] for i in train_idx]
            va_list = [train_samples[i] for i in val_idx]
    else:
        tr_list = list(train_samples)
        va_list = list(val_samples)

    torch.manual_seed(seed)
    # Build detector wrapper without loading an old checkpoint so weights start fresh
    detector = BaselineAntiSpoofDetector(checkpoint_path="__none__")
    device = detector.device
    model = detector.model

    train_ds = AntiSpoofAudioDataset(
        tr_list,
        feature_extractor=detector,
        clip_seconds=clip_seconds,
        training=True,
        telephony_aug_prob=telephony_aug_prob,
        seed=seed,
    )
    val_ds = AntiSpoofAudioDataset(
        va_list,
        feature_extractor=detector,
        clip_seconds=clip_seconds,
        training=False,
        telephony_aug_prob=0.0,
        seed=seed,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # Compute inverse-frequency class weights for imbalanced datasets
    tr_labels = np.array([lbl for _, lbl in tr_list])
    n_bonafide = max(1, int(np.sum(tr_labels == 0)))
    n_spoof = max(1, int(np.sum(tr_labels == 1)))
    total = float(n_bonafide + n_spoof)
    class_weights = torch.tensor(
        [total / (2.0 * n_bonafide), total / (2.0 * n_spoof)],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    best_eer = float("inf")
    best_loss = float("inf")
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels_t = labels.to(device, dtype=torch.long)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels_t)
            loss.backward()
            optimizer.step()

            bs = labels_t.size(0)
            running_loss += float(loss.item()) * bs
            seen += bs

        scheduler.step()
        train_loss = round(running_loss / max(1, seen), 4)
        val_metrics = evaluate_epoch(model, val_loader, criterion, device)
        epoch_stats = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(epoch_stats)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_metrics['val_loss']:.4f} | "
            f"val_acc={val_metrics['val_accuracy']:.4f} | val_eer={val_metrics['val_eer']:.4f} | "
            f"val_auc={val_metrics['val_roc_auc']:.4f}"
        )

        is_better = (val_metrics["val_eer"] < best_eer) or (
            val_metrics["val_eer"] == best_eer and val_metrics["val_loss"] < best_loss
        )
        if is_better:
            best_eer = val_metrics["val_eer"]
            best_loss = val_metrics["val_loss"]
            torch.save(model.state_dict(), output_checkpoint)

    summary = {
        "checkpoint_path": str(output_checkpoint),
        "device": str(device),
        "train_samples": len(tr_list),
        "val_samples": len(va_list),
        "best_val_eer": best_eer,
        "best_val_loss": best_loss,
        "history": history,
    }
    metrics_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BAFV-PCTA Baseline Anti-Spoofing Detector")
    parser.add_argument("--data-dir", type=Path, help="Folder with bonafide/ and spoof/ subdirectories")
    parser.add_argument("--val-dir", type=Path, help="Optional separate validation folder")
    parser.add_argument("--asvspoof-protocol", type=Path, help="ASVspoof 2019 LA train protocol txt")
    parser.add_argument("--audio-root", type=Path, help="Audio directory for ASVspoof train protocol")
    parser.add_argument("--dev-protocol", type=Path, help="Optional ASVspoof 2019 LA dev protocol txt")
    parser.add_argument("--dev-audio-root", type=Path, help="Audio directory for ASVspoof dev protocol")
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models" / "aasist_baseline.pt",
        help="Destination path for best PyTorch state_dict",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=PROJECT_ROOT / "models" / "training_metrics.json",
        help="Destination path for JSON training report",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip-seconds", type=float, default=3.0)
    parser.add_argument("--telephony-aug-prob", type=float, default=0.25)
    args = parser.parse_args()

    if args.data_dir:
        train_samples = collect_folder_samples(args.data_dir)
        val_samples = collect_folder_samples(args.val_dir) if args.val_dir else None
    elif args.asvspoof_protocol and args.audio_root:
        train_samples = collect_asvspoof_samples(args.asvspoof_protocol, args.audio_root)
        val_samples = (
            collect_asvspoof_samples(args.dev_protocol, args.dev_audio_root)
            if args.dev_protocol and args.dev_audio_root
            else None
        )
    else:
        parser.error("Provide either --data-dir or (--asvspoof-protocol and --audio-root).")
        return

    summary = train_pipeline(
        train_samples=train_samples,
        val_samples=val_samples,
        output_checkpoint=args.output_checkpoint,
        metrics_out=args.metrics_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        clip_seconds=args.clip_seconds,
        telephony_aug_prob=args.telephony_aug_prob,
    )
    print(f"Saved best checkpoint to {summary['checkpoint_path']} (best_val_eer={summary['best_val_eer']:.4f})")


if __name__ == "__main__":
    main()
