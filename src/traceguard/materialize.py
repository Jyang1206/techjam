from __future__ import annotations

import argparse
import hashlib
import shutil
from collections.abc import Mapping
from pathlib import Path

from .data import decode_dataset_image, wildfake_records_from_manifest


def _target_dir(output_root: Path, label: int) -> Path:
    directory = output_root / ("fake" if label else "real")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _wildfake_target_name(record_path: Path, images_root: Path, group: str) -> str:
    """Return a stable name that remains unique when WildFake basenames collide."""
    relative_path = record_path.resolve().relative_to(images_root.resolve())
    digest = hashlib.sha256(relative_path.as_posix().encode("utf-8")).hexdigest()[:12]
    return f"{group}__{digest}__{record_path.name}"


def materialize_huggingface(
    dataset_id: str,
    *,
    output_root: Path,
    split: str = "train",
    config: str | None = None,
    image_column: str = "image",
    label_column: str = "label",
    id_column: str = "img_id",
    positive_labels: tuple[int, ...] = (1, 2),
    samples_per_class: int,
    samples_per_label: Mapping[int, int] | None = None,
    group_by_label: bool = False,
    shuffle_buffer: int = 200,
    skip_samples: int = 0,
    seed: int = 42,
) -> tuple[int, int]:
    """Stream a balanced sample from a Hugging Face dataset onto disk as real/fake JPEGs.

    Unlike HuggingFaceStreamDataset (built for direct streaming into training), this walks the
    stream once and writes files to a shared local folder, so the result can be merged with other
    sources and trained on together via plain local-folder mode.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency error is user-facing
        raise RuntimeError("Install Hugging Face support with `pip install -e .`") from exc

    stream = load_dataset(dataset_id, config, split=split, streaming=True)
    if shuffle_buffer > 0:
        stream = stream.shuffle(buffer_size=shuffle_buffer, seed=seed)
    if skip_samples < 0:
        raise ValueError("skip_samples cannot be negative")
    if skip_samples:
        stream = stream.skip(skip_samples)

    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive")
    label_targets = dict(samples_per_label or {})
    if any(count < 1 for count in label_targets.values()):
        raise ValueError("Every per-label sample target must be positive")

    slug = dataset_id.split("/")[-1]
    counts = {0: 0, 1: 0}
    label_counts = {label: 0 for label in label_targets}
    if label_targets and group_by_label:
        for binary_label, directory_name in ((0, "real"), (1, "fake")):
            directory = output_root / directory_name
            if not directory.is_dir():
                continue
            for path in directory.glob(f"hf__{slug}__label_*__*"):
                parts = path.name.split("__", 3)
                try:
                    source_label = int(parts[2].removeprefix("label_"))
                except (IndexError, ValueError):
                    continue
                expected_binary_label = 1 if source_label in positive_labels else 0
                if source_label in label_targets and expected_binary_label == binary_label:
                    label_counts[source_label] += 1
                    counts[binary_label] += 1
        if any(label_counts.values()):
            detail = " ".join(
                f"label_{key}={value}" for key, value in sorted(label_counts.items())
            )
            print(f"  {dataset_id}: resuming from {detail}", flush=True)

    for index, row in enumerate(stream):
        if label_targets:
            if all(label_counts[label] >= target for label, target in label_targets.items()):
                break
        elif counts[0] >= samples_per_class and counts[1] >= samples_per_class:
            break
        source_label = int(row[label_column])
        label = 1 if source_label in positive_labels else 0
        if label_targets and (
            source_label not in label_targets
            or label_counts[source_label] >= label_targets[source_label]
        ):
            continue
        if not label_targets and counts[label] >= samples_per_class:
            continue
        identifier = str(row.get(id_column, index))
        group = f"hf__{slug}"
        if group_by_label:
            group = f"{group}__label_{source_label}"
        target = _target_dir(output_root, label) / f"{group}__{identifier}.jpg"
        if target.is_file():
            continue
        try:
            image = decode_dataset_image(row[image_column])
        except (KeyError, OSError, TypeError, ValueError):
            continue
        image.convert("RGB").save(target, format="JPEG", quality=95)
        counts[label] += 1
        if label_targets:
            label_counts[source_label] += 1
        if sum(counts.values()) % 500 == 0:
            detail = (
                " ".join(f"label_{key}={value}" for key, value in sorted(label_counts.items()))
                if label_targets
                else f"real={counts[0]} fake={counts[1]}"
            )
            print(f"  {dataset_id}: {detail}", flush=True)
    incomplete = {
        label: target - label_counts[label]
        for label, target in label_targets.items()
        if label_counts[label] < target
    }
    if incomplete:
        shortfall = ", ".join(
            f"label {label}: {missing} missing" for label, missing in sorted(incomplete.items())
        )
        raise ValueError(f"Hugging Face stream ended before targets were met ({shortfall})")
    return counts[0], counts[1]


def _label_sample_target(value: str) -> tuple[int, int]:
    try:
        label_text, count_text = value.split("=", 1)
        label, count = int(label_text), int(count_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected LABEL=COUNT, for example 1=25000") from exc
    if count < 1:
        raise argparse.ArgumentTypeError("COUNT must be positive")
    return label, count


def materialize_wildfake(
    manifest: Path,
    images_root: Path,
    *,
    output_root: Path,
    samples_per_class: int,
    seed: int = 42,
    exclude_protected: bool = True,
    include_groups: frozenset[str] | None = None,
) -> tuple[int, int]:
    """Copy a balanced WildFake sample from its official manifest onto disk as real/fake files."""
    records = wildfake_records_from_manifest(
        manifest,
        images_root,
        maximum=samples_per_class * 2,
        seed=seed,
        exclude_protected=exclude_protected,
        include_groups=include_groups,
    )
    counts = {0: 0, 1: 0}
    for record in records:
        # record.group is already "wildfake__<generator>__<architecture>" (see
        # _wildfake_group in data.py). Include a digest of the path relative to Images/ because
        # WildFake reuses basenames across category subdirectories within one architecture.
        target_name = _wildfake_target_name(record.path, images_root, record.group or "unknown")
        target = _target_dir(output_root, record.label) / target_name
        shutil.copyfile(record.path, target)
        counts[record.label] += 1
    return counts[0], counts[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize a balanced, bounded sample from remote datasets into one local "
        "real/fake folder, suitable for `traceguard-train <output-dir>` afterward. Run once per "
        "source you want included; each call adds to the same --output-dir without touching what "
        "a previous call already wrote."
    )
    parser.add_argument("--output-dir", default="data/merged")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--hf-dataset", help="e.g. saberzl/SID_Set")
    parser.add_argument("--hf-config")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-image-column", default="image")
    parser.add_argument("--hf-label-column", default="label")
    parser.add_argument("--hf-id-column", default="img_id")
    parser.add_argument("--hf-positive-labels", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--hf-samples-per-class", type=int, default=5000)
    parser.add_argument(
        "--hf-samples-per-label",
        nargs="+",
        type=_label_sample_target,
        metavar="LABEL=COUNT",
        help="Sample exact original-label counts instead of binary class counts, e.g. "
        "0=25000 1=25000 2=25000 for SID_Set.",
    )
    parser.add_argument(
        "--hf-group-by-label",
        action="store_true",
        help="Encode each original HF label as a separate balancing/splitting group.",
    )
    parser.add_argument(
        "--hf-shuffle-buffer",
        type=int,
        default=200,
        help="Lower = faster start (less data fetched just to prime randomization), less uniform "
        "sampling. Streaming a large buffer over an unauthenticated, rate-limited connection can "
        "dominate total runtime; 0 disables shuffling entirely for maximum speed.",
    )

    parser.add_argument("--wildfake-manifest")
    parser.add_argument("--wildfake-images-root")
    parser.add_argument("--wildfake-samples-per-class", type=int, default=5000)
    parser.add_argument(
        "--wildfake-include-groups",
        nargs="+",
        help="Optional exact group allowlist, e.g. wildfake__Real__imagenet "
        "wildfake__Diffusion_based__DDPM.",
    )
    parser.add_argument(
        "--hf-skip-samples",
        type=int,
        default=0,
        help="Skip this many rows after deterministic streaming shuffle; useful when resuming "
        "from a previously materialized prefix.",
    )
    parser.add_argument(
        "--allow-protected-wildfake",
        action="store_true",
        help="Include COCO real and advanced DALL-E rows; never use for challenge submission",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_dir)

    if args.hf_dataset:
        per_label_targets = dict(args.hf_samples_per_label or [])
        if len(per_label_targets) != len(args.hf_samples_per_label or []):
            raise ValueError("Each --hf-samples-per-label label may only be specified once")
        real, fake = materialize_huggingface(
            args.hf_dataset,
            output_root=output_root,
            split=args.hf_split,
            config=args.hf_config,
            image_column=args.hf_image_column,
            label_column=args.hf_label_column,
            id_column=args.hf_id_column,
            positive_labels=tuple(args.hf_positive_labels),
            samples_per_class=args.hf_samples_per_class,
            samples_per_label=per_label_targets,
            group_by_label=args.hf_group_by_label,
            shuffle_buffer=args.hf_shuffle_buffer,
            skip_samples=args.hf_skip_samples,
            seed=args.seed,
        )
        print(f"{args.hf_dataset}: wrote {real} real / {fake} fake images to {output_root}")

    if args.wildfake_manifest:
        real, fake = materialize_wildfake(
            Path(args.wildfake_manifest),
            Path(args.wildfake_images_root),
            output_root=output_root,
            samples_per_class=args.wildfake_samples_per_class,
            seed=args.seed,
            exclude_protected=not args.allow_protected_wildfake,
            include_groups=(
                frozenset(args.wildfake_include_groups)
                if args.wildfake_include_groups
                else None
            ),
        )
        print(f"WildFake: wrote {real} real / {fake} fake images to {output_root}")

    if not args.hf_dataset and not args.wildfake_manifest:
        raise ValueError("Pass --hf-dataset and/or --wildfake-manifest to materialize something.")


if __name__ == "__main__":
    main()
