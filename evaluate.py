"""
DeepGuard Standalone Model Evaluation Script
Implements Section 10 specification:
- Loads weights.pt and evaluates on a dataset directory
- Labels: 'RealVideo-RealAudio' -> 0 (real), all others -> 1 (fake)
- Calculates AUC, F1, Accuracy, Precision, Recall, Confusion Matrix
- Renders and saves confusion matrix plot as PNG
- Saves evaluation_report.json
- Supports cross-dataset evaluation with --cross-dataset flag
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from tqdm import tqdm

import config
import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("deepguard-evaluate")


def plot_confusion_matrix(cm: np.ndarray, save_path: Path, title: str = "Confusion Matrix") -> None:
    """Plots and saves confusion matrix as a clean PNG image."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ["Real (0)", "Fake (1)"]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    # Annotate values
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    logger.info("Saved confusion matrix figure to %s", save_path)


def evaluate(
    dataset_dir: str | Path,
    weights_path: str = config.WEIGHTS_PATH,
    cross_dataset: bool = False,
    output_dir: str | Path = ".",
) -> Dict[str, Any]:
    dataset_path = Path(dataset_dir).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_file = Path(weights_path).resolve()
    if not weights_file.exists():
        raise FileNotFoundError(f"Model weights file '{weights_file}' not found.")

    device = pipeline.get_device()
    detector = pipeline.Detector(input_dim=config.FEAT_DIM * 2, hidden_dim=config.HIDDEN).to(device)
    state_dict = torch.load(weights_file, map_location=device)
    detector.load_state_dict(state_dict)
    detector.eval()
    logger.info("Loaded trained detector weights from %s", weights_file)

    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_files = [
        p for p in dataset_path.rglob("*")
        if p.is_file() and p.suffix.lower() in video_extensions
    ]
    video_files.sort()

    if not video_files:
        raise ValueError(f"No video files found in {dataset_path}")

    logger.info("Found %d video files to evaluate in %s", len(video_files), dataset_path)

    y_true: List[int] = []
    y_scores: List[float] = []
    skipped = 0

    for p in tqdm(video_files, desc="Evaluating"):
        path_str = str(p).replace("\\", "/")
        label = 0 if "RealVideo-RealAudio" in path_str else 1

        try:
            feats = pipeline.features(p, device=device)  # (T, 2560)
            feats_batch = feats.unsqueeze(0).to(device)
            with torch.no_grad():
                prob = float(detector.predict_proba(feats_batch).item())
            y_true.append(label)
            y_scores.append(prob)
        except Exception as e:
            logger.warning("Could not evaluate file %s: %s", p.name, e)
            skipped += 1

    if not y_true:
        raise RuntimeError("No files could be evaluated successfully.")

    y_true_arr = np.array(y_true)
    y_scores_arr = np.array(y_scores)
    y_pred_arr = (y_scores_arr >= config.THRESHOLD).astype(int)

    # Compute metrics
    try:
        if len(np.unique(y_true_arr)) > 1:
            auc = float(roc_auc_score(y_true_arr, y_scores_arr))
        else:
            auc = 0.5
    except Exception:
        auc = 0.5

    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])

    dataset_type = "cross-dataset" if cross_dataset else "standard"
    logger.info("Evaluation Complete [%s]:", dataset_type.upper())
    logger.info("AUC: %.4f | F1: %.4f | Accuracy: %.4f | Precision: %.4f | Recall: %.4f", auc, f1, acc, prec, rec)
    logger.info("Confusion Matrix:\n%s", cm)

    # Save confusion matrix plot
    cm_name = "confusion_matrix_cross.png" if cross_dataset else "confusion_matrix.png"
    cm_path = out_dir / cm_name
    title_suffix = " (Cross-Dataset)" if cross_dataset else " (In-Distribution)"
    plot_confusion_matrix(cm, cm_path, title=f"DeepGuard Confusion Matrix{title_suffix}")

    report = {
        "dataset_path": str(dataset_path),
        "is_cross_dataset": cross_dataset,
        "evaluation_type": dataset_type,
        "weights_path": str(weights_file),
        "total_evaluated": len(y_true),
        "skipped_files": skipped,
        "real_count": int(np.sum(y_true_arr == 0)),
        "fake_count": int(np.sum(y_true_arr == 1)),
        "metrics": {
            "auc": round(auc, 4),
            "f1": round(f1, 4),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
        },
        "confusion_matrix": cm.tolist(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": "A performance drop is expected when evaluating cross-dataset models on unseen generators." if cross_dataset else "Standard test split evaluation.",
    }

    report_name = "evaluation_report_cross.json" if cross_dataset else "evaluation_report.json"
    report_path = out_dir / report_name
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved evaluation report to %s", report_path)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DeepGuard BiLSTM+MLP detector")
    parser.add_argument("dataset", type=str, help="Path to evaluation dataset root")
    parser.add_argument("--weights", type=str, default=config.WEIGHTS_PATH, help="Path to weights.pt")
    parser.add_argument("--cross-dataset", action="store_true", help="Flag indicating cross-dataset evaluation")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save report and matrix")

    args = parser.parse_args()
    evaluate(
        dataset_dir=args.dataset,
        weights_path=args.weights,
        cross_dataset=args.cross_dataset,
        output_dir=args.output_dir,
    )
