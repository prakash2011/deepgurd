"""
DeepGuard Model Training Script
Implements Section 9 specification:
- Recursively collects video files from FakeAVCeleb dataset
- Label extraction: 'RealVideo-RealAudio' -> 0 (real), all others -> 1 (fake)
- GroupShuffleSplit (80/20) by subject identity with zero-leakage assertion
- Feature extraction & disk caching (resume-capable)
- BCEWithLogitsLoss with class-balancing pos_weight (n_real / n_fake)
- AdamW optimizer, lr 1e-3, weight decay 1e-4, early stopping (patience 5)
- Saves best-AUC model checkpoint to weights.pt
- Writes comprehensive training_report.json
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm

import config
import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("deepguard-train")


def extract_identity_and_label(path: Path, root: Path) -> Tuple[str, int]:
    """
    Extracts subject identity folder and binary label (0=Real, 1=Fake).
    Convention: paths containing 'RealVideo-RealAudio' -> 0, else 1.
    Identity is the subject folder name (e.g. 'id00017' or direct parent).
    """
    path_str = str(path).replace("\\", "/")
    label = 0 if "RealVideo-RealAudio" in path_str else 1
    # Parent directory in FakeAVCeleb structure is always the person identity
    identity = path.parent.name
    return identity, label


class CachedFeatureDataset(Dataset):
    def __init__(self, samples: List[Tuple[torch.Tensor, int]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feat, label = self.samples[idx]
        return feat, torch.tensor(label, dtype=torch.float32)


def train_model(
    dataset_dir: str | Path,
    epochs: int = 30,
    batch_size: int = 32,
    cache_path: str = "features_cache.pt",
    weights_path: str = config.WEIGHTS_PATH,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    dataset_path = Path(dataset_dir).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory '{dataset_path}' does not exist.")

    # 1. Collect video files
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_files: List[Path] = [
        p for p in dataset_path.rglob("*")
        if p.is_file() and p.suffix.lower() in video_extensions
    ]
    video_files.sort()

    if not video_files:
        raise ValueError(f"No video files found in {dataset_path}")

    logger.info("Found %d video files in %s", len(video_files), dataset_path)

    # Balanced cross-demographic multimodal sampling across Real, Audio-Fake, Video-Fake, and Multi-Fake
    if max_samples and max_samples > 0 and len(video_files) > max_samples:
        import random
        random.seed(42)
        eths = ['African', 'Asian (East)', 'Asian (South)', 'Caucasian (American)', 'Caucasian (European)']

        # Real samples: stratified equally across all 5 ethnicities
        n_real = max_samples // 2
        per_eth_real = max(1, n_real // len(eths))
        sampled_reals: List[Path] = []
        for eth in eths:
            v_eth = [p for p in video_files if "RealVideo-RealAudio" in str(p) and eth in str(p)]
            if v_eth:
                sampled_reals.extend(random.sample(v_eth, min(len(v_eth), per_eth_real)))

        # Fake samples: stratified across modalities and ethnicities
        n_fake_target = max_samples - len(sampled_reals)
        n_fake_per_cat = max(1, n_fake_target // 3)
        per_eth_fake = max(1, n_fake_per_cat // len(eths))

        sampled_fakes: List[Path] = []
        for cat in ["RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]:
            for eth in eths:
                v_cat_eth = [p for p in video_files if cat in str(p) and eth in str(p)]
                if v_cat_eth:
                    sampled_fakes.extend(random.sample(v_cat_eth, min(len(v_cat_eth), per_eth_fake)))

        video_files = sorted(sampled_reals + sampled_fakes)
        logger.info(
            "Selected balanced cross-demographic dataset: %d videos (%d Reals across all 5 ethnicities, %d Fakes across audio/visual/multimodal)",
            len(video_files), len(sampled_reals), len(sampled_fakes),
        )

    # 2. Extract identities and labels
    samples_info: List[Dict[str, Any]] = []
    for p in video_files:
        identity, label = extract_identity_and_label(p, dataset_path)
        samples_info.append({
            "path": p,
            "identity": identity,
            "label": label,
        })

    # 3. Load or initialize feature cache
    cache_file = Path(cache_path)
    feature_cache: Dict[str, torch.Tensor] = {}
    if cache_file.exists():
        try:
            logger.info("Loading existing feature cache from %s", cache_file)
            feature_cache = torch.load(cache_file, map_location="cpu")
        except Exception as e:
            logger.warning("Could not load cache file %s: %s. Starting fresh.", cache_file, e)
            feature_cache = {}

    device = pipeline.get_device()
    logger.info("Using device: %s for feature extraction and training", device)

    valid_samples: List[Tuple[torch.Tensor, int, str, Path]] = []
    newly_cached = 0

    for info in tqdm(samples_info, desc="Extracting / Loading Features"):
        p = info["path"]
        key = str(p)
        feat: Optional[torch.Tensor] = feature_cache.get(key)
        if feat is None:
            try:
                feat = pipeline.features(p, device=device)  # (T, 2560)
                feature_cache[key] = feat
                newly_cached += 1
            except Exception as e:
                logger.warning("Skipping unreadable/corrupt file %s: %s", p.name, e)
                continue

        valid_samples.append((feat, info["label"], info["identity"], p))

    if newly_cached > 0 or not cache_file.exists():
        try:
            torch.save(feature_cache, cache_file)
            logger.info("Saved %d features to cache %s", len(feature_cache), cache_file)
        except Exception as e:
            logger.warning("Failed to save feature cache: %s", e)

    if not valid_samples:
        raise RuntimeError("No valid features could be extracted from dataset.")

    # 4. GroupShuffleSplit by identity (80/20, seed 0)
    identities = [s[2] for s in valid_samples]
    labels = [s[1] for s in valid_samples]
    indices = np.arange(len(valid_samples))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, val_idx = next(gss.split(indices, labels, groups=identities))

    train_groups: Set[str] = {identities[i] for i in train_idx}
    val_groups: Set[str] = {identities[i] for i in val_idx}

    # Assertion: zero identity overlap
    overlap = train_groups.intersection(val_groups)
    assert len(overlap) == 0, f"Identity overlap detected! Leaked identities: {overlap}"
    logger.info(
        "GroupShuffleSplit confirmed zero leakage: Train identities=%d (%d samples), Val identities=%d (%d samples)",
        len(train_groups), len(train_idx), len(val_groups), len(val_idx),
    )

    train_data = [valid_samples[i] for i in train_idx]
    val_data = [valid_samples[i] for i in val_idx]

    n_train_real = sum(1 for s in train_data if s[1] == 0)
    n_train_fake = sum(1 for s in train_data if s[1] == 1)
    logger.info("Train class distribution: Real=%d, Fake=%d", n_train_real, n_train_fake)

    # 5. Class-balanced loss & optimizer
    pos_weight_val = float(n_train_real) / max(1.0, float(n_train_fake))
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_dataset = CachedFeatureDataset([(s[0], s[1]) for s in train_data])
    val_dataset = CachedFeatureDataset([(s[0], s[1]) for s in val_data])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = pipeline.Detector(input_dim=config.FEAT_DIM * 2, hidden_dim=config.HIDDEN).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-5)

    best_auc = -1.0
    best_f1 = 0.0
    best_acc = 0.0
    best_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    logger.info("Starting training for %d epochs...", epochs)

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        total_loss = 0.0
        for feats, targets in train_loader:
            feats = feats.to(device)
            targets = targets.to(device)

            # Feature augmentation (slight Gaussian noise) to prevent overfitting and improve generalization
            if feats.size(0) > 1:
                noise = torch.randn_like(feats) * 0.005
                feats = feats + noise

            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)

        scheduler.step()
        avg_loss = total_loss / len(train_dataset)

        # Validation phase
        model.eval()
        val_probs: List[float] = []
        val_targets: List[int] = []

        with torch.no_grad():
            for feats, targets in val_loader:
                feats = feats.to(device)
                logits = model(feats)
                probs = torch.sigmoid(logits).cpu().numpy().tolist()
                val_probs.extend(probs)
                val_targets.extend(targets.int().numpy().tolist())

        val_probs_arr = np.array(val_probs)
        val_targets_arr = np.array(val_targets)
        val_preds_arr = (val_probs_arr >= config.THRESHOLD).astype(int)

        # Calculate metrics
        try:
            if len(np.unique(val_targets_arr)) > 1:
                val_auc = float(roc_auc_score(val_targets_arr, val_probs_arr))
            else:
                val_auc = 0.5
        except Exception:
            val_auc = 0.5

        val_f1 = float(f1_score(val_targets_arr, val_preds_arr, zero_division=0))
        val_acc = float(accuracy_score(val_targets_arr, val_preds_arr))

        print(
            f"Epoch {epoch:02d}/{epochs:02d} - Loss: {avg_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Model selection: best AUC, breaking ties with F1 and loss
        improved = False
        if val_auc > best_auc + 1e-4:
            improved = True
        elif abs(val_auc - best_auc) <= 1e-4:
            if val_f1 > best_f1 + 1e-4:
                improved = True
            elif abs(val_f1 - best_f1) <= 1e-4 and avg_loss < best_loss:
                improved = True

        if improved:
            best_auc = val_auc
            best_f1 = val_f1
            best_acc = val_acc
            best_loss = avg_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), weights_path)
            logger.info("--> Checkpoint saved to %s (Epoch %d | Val AUC: %.4f | F1: %.4f | Loss: %.4f)", weights_path, epoch, best_auc, best_f1, avg_loss)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", patience)
                break

    # 7. Write training_report.json
    report = {
        "dataset_path": str(dataset_path),
        "total_videos": len(valid_samples),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "train_identities": len(train_groups),
        "val_identities": len(val_groups),
        "identity_overlap": 0,
        "best_epoch": best_epoch,
        "best_auc": round(best_auc, 4),
        "best_f1": round(best_f1, 4),
        "best_accuracy": round(best_acc, 4),
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "patience": patience,
            "hidden_dim": config.HIDDEN,
            "pos_weight": round(pos_weight_val, 4),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    report_path = Path("training_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved training report to %s", report_path)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepGuard BiLSTM+MLP detector on FakeAVCeleb")
    parser.add_argument("dataset", type=str, help="Path to FakeAVCeleb dataset root")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs (default: 30)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--cache", type=str, default="features_cache.pt", help="Path to feature cache file")
    parser.add_argument("--weights", type=str, default=config.WEIGHTS_PATH, help="Path to save best weights")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max balanced samples to train on")

    args = parser.parse_args()
    train_model(
        dataset_dir=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch,
        cache_path=args.cache,
        weights_path=args.weights,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        max_samples=args.max_samples,
    )
