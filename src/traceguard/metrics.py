from __future__ import annotations

import numpy as np


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if not len(positives) or not len(negatives):
        return float("nan")
    comparisons = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    return float((comparisons + 0.5 * ties) / (len(positives) * len(negatives)))


def binary_metrics(labels, scores, threshold: float = 0.5) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels and scores must be non-empty one-dimensional arrays of equal size")
    predictions = (scores >= threshold).astype(np.int64)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "samples": int(labels.size),
        "accuracy": float((tp + tn) / labels.size),
        "balanced_accuracy": float((recall + specificity) / 2),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "roc_auc": _roc_auc(labels, scores),
        "brier": float(np.mean((scores - labels) ** 2)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "false_negative_rate": float(fn / max(fn + tp, 1)),
    }
