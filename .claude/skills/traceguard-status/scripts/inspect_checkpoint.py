#!/usr/bin/env python3
"""Inspect a TraceGuard training run: print per-epoch metrics and an overfit/underfit read.

Usage:
    python inspect_checkpoint.py <output_dir>          # e.g. checkpoints/cifake
    python inspect_checkpoint.py <output_dir> --json    # raw data instead of a table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

METRIC_COLUMNS = [
    ("epoch", "{:d}", "epoch"),
    ("train_loss", "{:.4f}", "train_loss"),
    ("validation_loss", "{:.4f}", "val_loss"),
    ("precision", "{:.3f}", "precision"),
    ("recall", "{:.3f}", "recall"),
    ("f1", "{:.3f}", "f1"),
    ("roc_auc", "{:.3f}", "auc"),
    ("balanced_accuracy", "{:.3f}", "bal_acc"),
]


def load_history(output_dir: Path) -> list[dict]:
    history_path = output_dir / "history.json"
    if not history_path.is_file():
        raise FileNotFoundError(
            f"No history.json found in {output_dir}. Has training finished? "
            "history.json is only written once, after the last epoch completes."
        )
    return json.loads(history_path.read_text(encoding="utf-8"))


def load_checkpoint_summary(output_dir: Path) -> dict | None:
    checkpoint_path = output_dir / "best.pt"
    if not checkpoint_path.is_file():
        return None
    try:
        import torch
    except ImportError:
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return {
        "epoch": checkpoint.get("epoch"),
        "threshold": checkpoint.get("threshold"),
        "temperature": checkpoint.get("temperature"),
        "training_source": checkpoint.get("training_source"),
        "validation_metrics": checkpoint.get("validation_metrics"),
    }


def diagnose(history: list[dict]) -> str:
    if len(history) < 2:
        return "Only one epoch recorded — not enough data to judge a trend yet."

    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["validation_loss"] for row in history]
    aucs = [row["roc_auc"] for row in history]

    train_improving = train_losses[-1] < train_losses[0]
    val_best_epoch = min(range(len(val_losses)), key=lambda i: val_losses[i])
    val_worsening_at_end = val_losses[-1] > val_losses[val_best_epoch] * 1.02
    auc_near_chance = max(aucs) < 0.6

    if auc_near_chance:
        return (
            "UNDERFITTING signal: validation ROC-AUC never rose much above chance (0.5). "
            "Consider more epochs, a higher learning rate, checking the data actually has a "
            "learnable signal, or verifying labels are correct."
        )
    if train_improving and val_worsening_at_end and val_best_epoch < len(history) - 1:
        return (
            f"OVERFITTING signal: validation loss bottomed out at epoch {history[val_best_epoch]['epoch']} "
            f"and rose afterward while training loss kept falling. The saved best.pt checkpoint "
            f"(epoch {history[val_best_epoch]['epoch']}) already protects against this, but consider "
            "stopping around there, adding regularization, or using a source-disjoint validation split "
            "if this trend looks aggressive."
        )
    if val_best_epoch == len(history) - 1:
        return (
            "Still improving: validation loss/AUC was still getting better at the final epoch. "
            "Training longer (more --epochs) may help further — rerun with a higher --epochs value "
            "to see where it actually plateaus or turns over."
        )
    return "Loss curves look like a reasonable, healthy fit — no strong over/underfitting signal detected."


def print_table(history: list[dict]) -> None:
    header = " | ".join(label for _, _, label in METRIC_COLUMNS)
    print(header)
    print("-" * len(header))
    for row in history:
        cells = [fmt.format(row[key]) for key, fmt, _ in METRIC_COLUMNS]
        print(" | ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Training --output-dir, e.g. checkpoints/cifake")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of a table")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    history = load_history(output_dir)
    checkpoint_summary = load_checkpoint_summary(output_dir)
    diagnosis = diagnose(history)

    if args.json:
        print(
            json.dumps(
                {"history": history, "checkpoint": checkpoint_summary, "diagnosis": diagnosis},
                indent=2,
            )
        )
        return

    print(f"Training run: {output_dir}\n")
    print_table(history)
    print()
    if checkpoint_summary:
        print(f"Best checkpoint: epoch {checkpoint_summary['epoch']}, "
              f"threshold={checkpoint_summary['threshold']:.4f}, "
              f"source={checkpoint_summary['training_source']}")
    print(f"\nDiagnosis: {diagnosis}")
    print(
        "\nReminder: these are the training run's own validation numbers. If that validation split "
        "was also used to pick the checkpoint/threshold (true for the Kaggle/CIFAKE path by default), "
        "treat this as a training-time sanity check, not a final generalization number — use "
        "traceguard-evaluate against a genuinely held-out dataset for that."
    )


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
