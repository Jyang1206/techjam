from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

from .data import discover_labeled_images
from .inference import Predictor
from .metrics import binary_metrics
from .transforms import ROBUSTNESS_SUITE, apply_degradation


def condition_name(transform: str, value: float) -> str:
    if transform == "clean":
        return "clean"
    return f"{transform}_{value:g}"


def write_summary(rows: list[dict[str, object]], output: Path) -> None:
    headers = ["Condition", "Accuracy", "Balanced accuracy", "ROC-AUC", "FPR", "FNR"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["condition"]),
                    f"{row['accuracy']:.3f}",
                    f"{row['balanced_accuracy']:.3f}",
                    f"{row['roc_auc']:.3f}",
                    f"{row['false_positive_rate']:.3f}",
                    f"{row['false_negative_rate']:.3f}",
                ]
            )
            + " |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> list[dict[str, object]]:
    records = discover_labeled_images(args.data_dir)
    predictor = Predictor.from_checkpoint(args.checkpoint, device=args.device)
    conditions = [("clean", 1.0)]
    selected = (
        ROBUSTNESS_SUITE
        if args.transforms == ["all"]
        else {name: ROBUSTNESS_SUITE[name] for name in args.transforms}
    )
    conditions.extend((name, value) for name, values in selected.items() for value in values)
    metric_rows: list[dict[str, object]] = []
    clean_predictions: list[dict[str, object]] = []

    for transform, value in conditions:
        labels: list[int] = []
        scores: list[float] = []
        for record in records:
            with Image.open(record.path) as source:
                image = apply_degradation(source.convert("RGB"), transform, value)
            score = predictor.score_image(image, tta=args.tta)
            labels.append(record.label)
            scores.append(score)
            if transform == "clean":
                clean_predictions.append(
                    {"image_path": str(record.path), "label": record.label, "pred": score}
                )
        metrics = binary_metrics(labels, scores, predictor.threshold)
        metric_rows.append({"condition": condition_name(transform, value), **metrics})
        print(
            f"{metric_rows[-1]['condition']:<14} "
            f"bal_acc={metrics['balanced_accuracy']:.3f} auc={metrics['roc_auc']:.3f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metric_rows, indent=2), encoding="utf-8")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_rows[0].keys())
        writer.writeheader()
        writer.writerows(metric_rows)
    write_summary(metric_rows, output_dir / "robustness_table.md")

    false_positives = sorted(
        (
            row
            for row in clean_predictions
            if row["label"] == 0 and row["pred"] >= predictor.threshold
        ),
        key=lambda row: row["pred"],
        reverse=True,
    )[: args.error_examples]
    false_negatives = sorted(
        (
            row
            for row in clean_predictions
            if row["label"] == 1 and row["pred"] < predictor.threshold
        ),
        key=lambda row: row["pred"],
    )[: args.error_examples]
    (output_dir / "error_analysis.json").write_text(
        json.dumps(
            {"false_positives": false_positives, "false_negatives": false_negatives}, indent=2
        ),
        encoding="utf-8",
    )
    return metric_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate clean and transformed image accuracy.")
    parser.add_argument("data_dir", help="Folder containing real/ and fake/")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument(
        "--transforms",
        nargs="+",
        choices=["all", *ROBUSTNESS_SUITE],
        default=["all"],
    )
    parser.add_argument("--tta", choices=("none", "robust"), default="robust")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--error-examples", type=int, default=12)
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
