from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .data import (
    HuggingFaceStreamDataset,
    LabeledImageDataset,
    discover_labeled_images,
    download_kaggle_dataset,
    external_validation_split,
    fake_generator_disjoint_split,
    fake_generator_holdout_split,
    find_split_directory,
    group_disjoint_split,
    limit_records,
    stratified_split,
    wildfake_records_from_manifest,
)
from .inference import resolve_device
from .metrics import binary_metrics
from .model import ModelConfig, TraceGuard, total_parameters, trainable_parameters
from .transforms import build_eval_transform, build_train_transform, normalization_for_backbone


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CachedFeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor, identifiers: list[str]) -> None:
        self.features = features
        self.targets = targets
        self.identifiers = identifiers

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.features[index], self.targets[index], self.identifiers[index]


def cache_features(
    model: TraceGuard,
    loader: DataLoader,
    device: torch.device,
    *,
    views: int = 1,
) -> CachedFeatureDataset:
    model.eval()
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    identifiers: list[str] = []
    with torch.inference_mode():
        for view in range(views):
            for batch_index, (images, batch_targets, batch_identifiers) in enumerate(loader):
                images = images.to(device)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    batch_features = model.extract_features(images)
                features.append(batch_features.float().cpu())
                targets.append(batch_targets.cpu())
                identifiers.extend(batch_identifiers)
                if batch_index % 50 == 0:
                    print(
                        f"  caching view {view + 1}/{views}: "
                        f"{batch_index * loader.batch_size}/{len(loader.dataset)}",
                        flush=True,
                    )
    return CachedFeatureDataset(torch.cat(features), torch.cat(targets), identifiers)


def validation_scores(
    model,
    loader,
    device,
    *,
    cached_features: bool = False,
) -> tuple[list[int], list[float], float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    labels: list[int] = []
    scores: list[float] = []
    losses: list[float] = []
    with torch.inference_mode():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device)
            targets = targets.float().to(device)
            logits = model.classify_features(inputs) if cached_features else model(inputs)
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


def balanced_group_weights(records) -> list[float]:
    """Give every label equal mass, then every source/generator within a label equal mass."""
    group_counts = Counter((record.label, record.group or "unknown") for record in records)
    groups_per_label = Counter(label for label, _ in group_counts)
    return [
        1.0
        / (
            2
            * groups_per_label[record.label]
            * group_counts[(record.label, record.group or "unknown")]
        )
        for record in records
    ]


def build_training_datasets(args: argparse.Namespace, model_config: ModelConfig | None = None):
    normalization = (model_config or ModelConfig()).normalization
    train_transform = lambda: build_train_transform(
        normalization=normalization,
        robustness_profile=args.robustness_profile,
    )
    eval_crop_pct = (model_config or ModelConfig()).eval_crop_pct
    eval_transform = lambda: build_eval_transform(
        normalization=normalization, crop_pct=eval_crop_pct
    )
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
            transform=train_transform(),
            max_samples=_sample_limit(args.max_train_samples),
            shuffle_buffer=args.hf_shuffle_buffer,
            **common,
        )
        validation_dataset = HuggingFaceStreamDataset(
            split=args.hf_validation_split,
            transform=eval_transform(),
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
        train_dataset = LabeledImageDataset(train_records, train_transform())
        validation_dataset = LabeledImageDataset(validation_records, eval_transform())
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
            LabeledImageDataset(train_records, train_transform()),
            LabeledImageDataset(validation_records, eval_transform()),
            len(train_records),
            len(validation_records),
            positive_weight,
            "ModelScope hy2628982280/WildFake",
        )

    records = discover_labeled_images(args.data_dir)
    if args.validation_dir:
        if (
            args.generator_disjoint_split
            or args.fake_generator_disjoint_split
            or args.validation_fake_groups
        ):
            raise ValueError(
                "--validation-dir replaces generator split flags and --validation-fake-groups"
            )
        train_records, validation_records = external_validation_split(
            records, discover_labeled_images(args.validation_dir)
        )
    else:
        train_records = validation_records = None
    if args.generator_disjoint_split and args.fake_generator_disjoint_split:
        raise ValueError("Choose only one generator-disjoint split mode")
    if args.validation_fake_groups and not args.fake_generator_disjoint_split:
        raise ValueError(
            "--validation-fake-groups requires --fake-generator-disjoint-split"
        )
    if args.validation_dir:
        pass
    elif args.fake_generator_disjoint_split:
        if args.validation_fake_groups:
            train_records, validation_records = fake_generator_holdout_split(
                records,
                args.val_fraction,
                args.seed,
                args.validation_fake_groups,
            )
        else:
            train_records, validation_records = fake_generator_disjoint_split(
                records, args.val_fraction, args.seed
            )
    elif args.generator_disjoint_split:
        train_records, validation_records = group_disjoint_split(
            records, args.val_fraction, args.seed
        )
    else:
        train_records, validation_records = stratified_split(
            records, args.val_fraction, args.seed
        )
    train_dataset = LabeledImageDataset(train_records, train_transform())
    validation_dataset = LabeledImageDataset(validation_records, eval_transform())
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
    initial_checkpoint = None
    if args.init_checkpoint:
        initial_checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        initial_config = replace(
            ModelConfig(**initial_checkpoint.get("model_config", {})), pretrained=False
        )
        if (
            args.low_resolution_size is not None
            and initial_config.low_resolution_size not in (0, args.low_resolution_size)
        ):
            raise ValueError(
                "Cannot change the low-resolution size of an existing multi-scale checkpoint"
            )
        config = replace(
            initial_config,
            low_resolution_size=(
                args.low_resolution_size
                if args.low_resolution_size is not None
                else initial_config.low_resolution_size
            ),
        )
    else:
        normalization = (
            normalization_for_backbone(args.backbone)
            if args.normalization == "auto"
            else args.normalization
        )
        config = ModelConfig(
            backbone=args.backbone,
            pretrained=not args.no_pretrained,
            dropout=args.dropout,
            normalization=normalization,
            use_frequency=not args.no_frequency_branch,
            backbone_projection=args.use_backbone_projection,
            classifier_layernorm=not args.no_classifier_layernorm,
            eval_crop_pct=1.0 if args.use_backbone_projection else 0.875,
            low_resolution_size=args.low_resolution_size or 0,
        )
    if config.low_resolution_size < 0 or config.low_resolution_size >= 224:
        raise ValueError("--low-resolution-size must be between 1 and 223, or 0 to disable")
    (
        train_dataset,
        validation_dataset,
        train_count,
        validation_count,
        positive_weight,
        source_name,
    ) = build_training_datasets(args, config)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    sampler = None
    if args.balance_groups:
        if not isinstance(train_dataset, LabeledImageDataset):
            raise ValueError("--balance-groups currently requires a local labeled-image dataset")
        sampler = WeightedRandomSampler(
            balanced_group_weights(train_dataset.records),
            num_samples=len(train_dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    train_loader = DataLoader(
        train_dataset,
        shuffle=sampler is None and not isinstance(train_dataset, HuggingFaceStreamDataset),
        sampler=sampler,
        **loader_options,
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    model = TraceGuard(config).to(device)
    if initial_checkpoint is not None:
        load_result = model.load_state_dict(initial_checkpoint["model_state"], strict=False)
        expected_missing = (
            {
                key
                for key in model.state_dict()
                if key.startswith("low_resolution_classifier.")
            }
            if config.low_resolution_size > initial_config.low_resolution_size
            else set()
        )
        if set(load_result.missing_keys) != expected_missing or load_result.unexpected_keys:
            raise RuntimeError(
                "Initializer is not architecture-compatible: "
                f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
            )
    if args.freeze_backbone:
        model.backbone.requires_grad_(False)
    if args.freeze_base_classifier:
        if model.low_resolution_classifier is None or initial_checkpoint is None:
            raise ValueError(
                "--freeze-base-classifier requires --init-checkpoint and --low-resolution-size"
            )
        model.classifier.requires_grad_(False)
        if model.frequency_head is not None:
            model.frequency_head.requires_grad_(False)
    using_cached_features = args.cache_frozen_features
    if using_cached_features:
        if args.feature_cache_views < 1 or args.head_batch_size < 1:
            raise ValueError("Feature-cache views and head batch size must both be positive")
        if not args.freeze_backbone or model.frequency is not None:
            raise ValueError(
                "--cache-frozen-features requires --freeze-backbone and --no-frequency-branch"
            )
        print(f"Caching {args.feature_cache_views} augmented training feature view(s).")
        cached_train = cache_features(
            model, train_loader, device, views=args.feature_cache_views
        )
        cached_validation = cache_features(model, validation_loader, device)
        cache_loader_options = {
            "batch_size": args.head_batch_size,
            "num_workers": 0,
            "pin_memory": device.type == "cuda",
        }
        train_loader = DataLoader(cached_train, shuffle=True, **cache_loader_options)
        validation_loader = DataLoader(
            cached_validation, shuffle=False, **cache_loader_options
        )
    pos_weight = torch.tensor([positive_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    best_auc = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    print(
        f"Training {trainable_parameters(model):,} of {total_parameters(model):,} parameters "
        f"on {device}; "
        f"{train_count} train / {validation_count} validation images from {source_name}."
    )

    if args.evaluate_initial:
        labels, scores, val_loss = validation_scores(
            model,
            validation_loader,
            device,
            cached_features=using_cached_features,
        )
        threshold = select_threshold(labels, scores)
        metrics = binary_metrics(labels, scores, threshold)
        best_auc = float(metrics["roc_auc"])
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_config": model.checkpoint_config(),
                "threshold": threshold,
                "temperature": 1.0,
                "epoch": 0,
                "validation_metrics": metrics,
                "training_source": source_name,
                "training_config": {
                    "initialized_from": Path(args.init_checkpoint).name
                    if args.init_checkpoint
                    else None,
                    "freeze_backbone": args.freeze_backbone,
                    "epochs_requested": args.epochs,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "early_stopping_patience": args.early_stopping_patience,
                    "cache_frozen_features": args.cache_frozen_features,
                    "feature_cache_views": args.feature_cache_views,
                    "freeze_base_classifier": args.freeze_base_classifier,
                    "low_resolution_size": config.low_resolution_size,
                    "robustness_profile": args.robustness_profile,
                },
            },
            best_path,
        )
        print(
            f"epoch=00 loss=n/a val_auc={best_auc:.4f} "
            f"val_bal_acc={metrics['balanced_accuracy']:.4f}"
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_backbone:
            model.backbone.eval()
        epoch_losses: list[float] = []
        for inputs, targets, _ in train_loader:
            inputs = inputs.to(device)
            targets = targets.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = (
                    model.classify_features(inputs) if using_cached_features else model(inputs)
                )
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.detach().cpu()))
        scheduler.step()

        labels, scores, val_loss = validation_scores(
            model,
            validation_loader,
            device,
            cached_features=using_cached_features,
        )
        threshold = select_threshold(labels, scores)
        metrics = binary_metrics(labels, scores, threshold)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "validation_loss": val_loss,
            **metrics,
        }
        history.append(row)
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(
            f"epoch={epoch:02d} loss={row['train_loss']:.4f} "
            f"val_auc={row['roc_auc']:.4f} val_bal_acc={row['balanced_accuracy']:.4f}"
        )
        if float(metrics["roc_auc"]) > best_auc:
            best_auc = float(metrics["roc_auc"])
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model.checkpoint_config(),
                    "threshold": threshold,
                    "temperature": 1.0,
                    "epoch": epoch,
                    "validation_metrics": metrics,
                    "training_source": source_name,
                    "training_config": {
                        "initialized_from": Path(args.init_checkpoint).name
                        if args.init_checkpoint
                        else None,
                        "freeze_backbone": args.freeze_backbone,
                        "epochs_requested": args.epochs,
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "early_stopping_patience": args.early_stopping_patience,
                        "cache_frozen_features": args.cache_frozen_features,
                        "feature_cache_views": args.feature_cache_views,
                        "freeze_base_classifier": args.freeze_base_classifier,
                        "low_resolution_size": config.low_resolution_size,
                        "robustness_profile": args.robustness_profile,
                    },
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                print(
                    f"Early stopping after {epoch} epochs; validation ROC-AUC did not improve "
                    f"for {args.early_stopping_patience} consecutive epochs."
                )
                break

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
    parser.add_argument(
        "--init-checkpoint",
        help="Initialize the full model from an existing compatible TraceGuard checkpoint.",
    )
    parser.add_argument(
        "--evaluate-initial",
        action="store_true",
        help="Evaluate/save epoch 0 before optimization, preserving the initializer if tuning hurts.",
    )
    parser.add_argument(
        "--normalization",
        choices=("auto", "imagenet", "clip"),
        default="auto",
        help="Input normalization; auto selects CLIP statistics for CLIP backbones.",
    )
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument(
        "--robustness-profile",
        choices=("standard", "low_resolution", "none"),
        default="standard",
        help="Training augmentation profile; low_resolution emphasizes label-symmetric "
        "32/56/112-pixel resize and JPEG artifacts.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Train only the detection head (and frequency branch, if enabled).",
    )
    parser.add_argument(
        "--no-frequency-branch",
        action="store_true",
        help="Use only spatial backbone features; useful for a frozen CLIP linear probe.",
    )
    parser.add_argument(
        "--use-backbone-projection",
        action="store_true",
        help="Keep a pretrained backbone's native projection head, required for official CLIP "
        "embedding probes.",
    )
    parser.add_argument(
        "--no-classifier-layernorm",
        action="store_true",
        help="Use a plain linear detection head, matching UniversalFakeDetect's CLIP probe.",
    )
    parser.add_argument(
        "--low-resolution-size",
        type=int,
        default=None,
        help="Add a shared-backbone residual expert over an image downsampled to this square size; "
        "32 targets severe low-resolution domain shift.",
    )
    parser.add_argument(
        "--freeze-base-classifier",
        action="store_true",
        help="When adding a low-resolution expert to an initializer, preserve its original head "
        "and train only the new residual classifier.",
    )
    parser.add_argument(
        "--cache-frozen-features",
        action="store_true",
        help="Cache frozen spatial features once, then tune the head without repeat backbone passes.",
    )
    parser.add_argument(
        "--feature-cache-views",
        type=int,
        default=1,
        help="Number of independently augmented training views to cache.",
    )
    parser.add_argument(
        "--head-batch-size",
        type=int,
        default=1024,
        help="Batch size for cached-feature head optimization.",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many non-improving validation epochs; 0 disables early stopping.",
    )
    parser.add_argument("--positive-weight", type=float)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--validation-dir",
        help="Local-folder mode only: use this labeled real/fake folder as the exact validation "
        "set and exclude matching materialized records from training.",
    )
    parser.add_argument(
        "--generator-disjoint-split",
        action="store_true",
        help="Local-folder mode only: hold out whole generator/source groups for validation "
        "instead of a random per-image split, so validation genuinely tests generators unseen "
        "during training. Only meaningful when images were named by traceguard-materialize "
        "(which encodes generator identity in the filename) - plain local folders with no "
        "recoverable group identity will raise an error rather than silently no-op.",
    )
    parser.add_argument(
        "--fake-generator-disjoint-split",
        action="store_true",
        help="Hold out fake generator groups but stratify authentic sources across both splits, "
        "avoiding real-source/content confounding.",
    )
    parser.add_argument(
        "--validation-fake-groups",
        nargs="+",
        help="With --fake-generator-disjoint-split, hold out these exact fake groups instead "
        "of selecting a subset based on size. Keeps data-scale experiments comparable.",
    )
    parser.add_argument(
        "--balance-groups",
        action="store_true",
        help="Sample local training records so each label and source/generator has equal mass.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
