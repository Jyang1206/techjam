from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import (
    HuggingFaceStreamDataset,
    LabeledImageDataset,
    discover_labeled_images,
    download_kaggle_dataset,
    find_split_directory,
    limit_records,
    stratified_split,
    wildfake_records_from_manifest,
)
from .inference import resolve_device
from .metrics import binary_metrics
from .model import ModelConfig, TraceGuard, trainable_parameters
from .transforms import build_eval_transform, build_train_transform


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validation_scores(model, loader, device) -> tuple[list[int], list[float], float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    labels: list[int] = []
    scores: list[float] = []
    losses: list[float] = []
    with torch.inference_mode():
        for images, targets, _ in loader:
            images = images.to(device)
            targets = targets.float().to(device)
            logits = model(images)
            losses.append(float(criterion(logits, targets).cpu()))
            labels.extend(targets.int().cpu().tolist())
            scores.extend(torch.sigmoid(logits).cpu().tolist())
    return labels, scores, float(np.mean(losses))


def select_threshold(labels: list[int], scores: list[float]) -> float:
    candidates = np.unique(np.r_[0.5, np.asarray(scores)])
    return float(
        max(
            candidates,
            key=lambda candidate: binary_metrics(labels, scores, candidate)["balanced_accuracy"],
        )
    )


def _sample_limit(value: int) -> int | None:
    return value if value > 0 else None


def build_training_datasets(args: argparse.Namespace):
    selected_sources = sum(
        bool(source)
        for source in (args.data_dir, args.hf_dataset, args.kaggle_dataset, args.wildfake_root)
    )
    if selected_sources != 1:
        raise ValueError("Choose exactly one local, Hugging Face, Kaggle, or WildFake source.")
    if args.hf_dataset:
        common = {
            "dataset_id": args.hf_dataset,
            "config": args.hf_config,
            "image_column": args.hf_image_column,
            "label_column": args.hf_label_column,
            "id_column": args.hf_id_column,
            "positive_labels": args.hf_positive_labels,
            "seed": args.seed,
        }
        train_dataset = HuggingFaceStreamDataset(
            split=args.hf_train_split,
            transform=build_train_transform(),
            max_samples=_sample_limit(args.max_train_samples),
            shuffle_buffer=args.hf_shuffle_buffer,
            **common,
        )
        validation_dataset = HuggingFaceStreamDataset(
            split=args.hf_validation_split,
            transform=build_eval_transform(),
            max_samples=_sample_limit(args.max_validation_samples),
            **common,
        )
        train_count = args.max_train_samples if args.max_train_samples > 0 else "full stream"
        validation_count = (
            args.max_validation_samples if args.max_validation_samples > 0 else "full stream"
        )
        positive_weight = args.positive_weight if args.positive_weight is not None else 0.5
        source_name = f"Hugging Face {args.hf_dataset}"
        return (
            train_dataset,
            validation_dataset,
            train_count,
            validation_count,
            positive_weight,
            source_name,
        )

    if args.kaggle_dataset:
        dataset_root = download_kaggle_dataset(args.kaggle_dataset)
        train_root = find_split_directory(dataset_root, args.kaggle_train_split)
        validation_root = find_split_directory(dataset_root, args.kaggle_validation_split)
        train_records = limit_records(
            discover_labeled_images(train_root),
            _sample_limit(args.max_train_samples),
            args.seed,
        )
        validation_records = limit_records(
            discover_labeled_images(validation_root),
            _sample_limit(args.max_validation_samples),
            args.seed,
        )
        train_dataset = LabeledImageDataset(train_records, build_train_transform())
        validation_dataset = LabeledImageDataset(validation_records, build_eval_transform())
        positive_weight = args.positive_weight if args.positive_weight is not None else 1.0
        source_name = f"Kaggle {args.kaggle_dataset}"
        return (
            train_dataset,
            validation_dataset,
            len(train_records),
            len(validation_records),
            positive_weight,
            source_name,
        )

    if args.wildfake_root:
        if args.max_train_samples <= 0 or args.max_validation_samples <= 0:
            raise ValueError("WildFake requires positive sample limits for bounded memory use.")
        root = Path(args.wildfake_root)
        images_root = root / "Images" if (root / "Images").is_dir() else root
        split_root = root / "split_train_test" / "csv_file" / "total_split"
        train_manifest = Path(args.wildfake_train_manifest or split_root / "train_metadata.csv")
        validation_manifest = Path(
            args.wildfake_validation_manifest or split_root / "test_metadata.csv"
        )
        train_records = wildfake_records_from_manifest(
            train_manifest,
            images_root,
            maximum=args.max_train_samples,
            seed=args.seed,
            exclude_protected=not args.allow_protected_wildfake,
        )
        validation_records = wildfake_records_from_manifest(
            validation_manifest,
            images_root,
            maximum=args.max_validation_samples,
            seed=args.seed,
            exclude_protected=not args.allow_protected_wildfake,
        )
        positive_count = sum(record.label == 1 for record in train_records)
        negative_count = len(train_records) - positive_count
        positive_weight = (
            args.positive_weight
            if args.positive_weight is not None
            else negative_count / max(positive_count, 1)
        )
        return (
            LabeledImageDataset(train_records, build_train_transform()),
            LabeledImageDataset(validation_records, build_eval_transform()),
            len(train_records),
            len(validation_records),
            positive_weight,
            "ModelScope hy2628982280/WildFake",
        )

    records = discover_labeled_images(args.data_dir)
    train_records, validation_records = stratified_split(records, args.val_fraction, args.seed)
    train_dataset = LabeledImageDataset(train_records, build_train_transform())
    validation_dataset = LabeledImageDataset(validation_records, build_eval_transform())
    fake_count = sum(record.label == 1 for record in train_records)
    real_count = len(train_records) - fake_count
    positive_weight = (
        args.positive_weight
        if args.positive_weight is not None
        else real_count / max(fake_count, 1)
    )
    return (
        train_dataset,
        validation_dataset,
        len(train_records),
        len(validation_records),
        positive_weight,
        str(args.data_dir),
    )


def train(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    previous_results = [
        path for path in (output_dir / "best.pt", output_dir / "history.json") if path.is_file()
    ]
    if previous_results and not args.overwrite:
        found = ", ".join(str(path) for path in previous_results)
        raise FileExistsError(
            f"{output_dir} already has training results ({found}). Refusing to overwrite them. "
            "Pick a new --output-dir for this run, e.g. checkpoints/<source>/run_XXX such as "
            "checkpoints/cifake/run_002, or pass --overwrite if replacing this run's results is "
            "intentional."
        )

    seed_everything(args.seed)
    device = resolve_device(args.device)
    (
        train_dataset,
        validation_dataset,
        train_count,
        validation_count,
        positive_weight,
        source_name,
    ) = build_training_datasets(args)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=not isinstance(train_dataset, HuggingFaceStreamDataset),
        **loader_options,
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    config = ModelConfig(backbone=args.backbone, pretrained=not args.no_pretrained)
    model = TraceGuard(config).to(device)
    pos_weight = torch.tensor([positive_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    best_auc = -1.0
    history: list[dict[str, float | int]] = []
    print(
        f"Training {trainable_parameters(model):,} parameters on {device}; "
        f"{train_count} train / {validation_count} validation images from {source_name}."
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.detach().cpu()))
        scheduler.step()

        labels, scores, val_loss = validation_scores(model, validation_loader, device)
        threshold = select_threshold(labels, scores)
        metrics = binary_metrics(labels, scores, threshold)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "validation_loss": val_loss,
            **metrics,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} loss={row['train_loss']:.4f} "
            f"val_auc={row['roc_auc']:.4f} val_bal_acc={row['balanced_accuracy']:.4f}"
        )
        if float(metrics["roc_auc"]) > best_auc:
            best_auc = float(metrics["roc_auc"])
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model.checkpoint_config(),
                    "threshold": threshold,
                    "temperature": 1.0,
                    "epoch": epoch,
                    "validation_metrics": metrics,
                    "training_source": source_name,
                },
                best_path,
            )

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {best_path} (validation ROC-AUC {best_auc:.4f})")
    return best_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train TraceGuard from local, Hugging Face, Kaggle, or WildFake data."
    )
    parser.add_argument("data_dir", nargs="?", help="Local folder containing real/ and fake/")
    parser.add_argument("--hf-dataset", help="Hub dataset ID, for example saberzl/SID_Set")
    parser.add_argument("--hf-config")
    parser.add_argument("--hf-train-split", default="train")
    parser.add_argument("--hf-validation-split", default="validation")
    parser.add_argument("--hf-image-column", default="image")
    parser.add_argument("--hf-label-column", default="label")
    parser.add_argument("--hf-id-column", default="img_id")
    parser.add_argument("--hf-positive-labels", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--hf-shuffle-buffer", type=int, default=2048)
    parser.add_argument(
        "--kaggle-dataset",
        help="Kaggle dataset handle, for example birdy654/cifake-real-and-ai-generated-synthetic-images",
    )
    parser.add_argument("--kaggle-train-split", default="train")
    parser.add_argument("--kaggle-validation-split", default="test")
    parser.add_argument(
        "--wildfake-root",
        help="Local WildFake root containing Images/ and official split manifests",
    )
    parser.add_argument("--wildfake-train-manifest")
    parser.add_argument("--wildfake-validation-manifest")
    parser.add_argument(
        "--allow-protected-wildfake",
        action="store_true",
        help="Include COCO real and advanced DALL-E rows; never use for challenge training",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=20000,
        help="Source sample cap; WildFake requires a positive value",
    )
    parser.add_argument(
        "--max-validation-samples",
        type=int,
        default=4000,
        help="Source sample cap; WildFake requires a positive value",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints",
        help="Where best.pt/history.json are written. Use a distinct directory per run, e.g. "
        "checkpoints/<source>/run_XXX such as checkpoints/cifake/run_002 — an existing run in "
        "this directory is protected unless --overwrite is passed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing best.pt/history.json already in --output-dir",
    )
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-weight", type=float)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
