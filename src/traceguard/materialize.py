from __future__ import annotations

import argparse
import hashlib
import shutil
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
    shuffle_buffer: int = 200,
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

    counts = {0: 0, 1: 0}
    for index, row in enumerate(stream):
        if counts[0] >= samples_per_class and counts[1] >= samples_per_class:
            break
        label = 1 if row[label_column] in positive_labels else 0
        if counts[label] >= samples_per_class:
            continue
        try:
            image = decode_dataset_image(row[image_column])
        except (KeyError, OSError, TypeError, ValueError):
            continue
        identifier = str(row.get(id_column, index))
        slug = dataset_id.split("/")[-1]
        # "hf__<dataset_slug>__<identifier>" - parsed back out by infer_group() for
        # group_disjoint_split. HF datasets don't carry a per-image generator/architecture column
        # the way WildFake's manifest does, so the whole dataset becomes one group.
        target = _target_dir(output_root, label) / f"hf__{slug}__{identifier}.jpg"
        image.convert("RGB").save(target, format="JPEG", quality=95)
        counts[label] += 1
        if sum(counts.values()) % 500 == 0:
            print(f"  {dataset_id}: real={counts[0]} fake={counts[1]}", flush=True)
    return counts[0], counts[1]


def materialize_wildfake(
    manifest: Path,
    images_root: Path,
    *,
    output_root: Path,
    samples_per_class: int,
    seed: int = 42,
    exclude_protected: bool = True,
) -> tuple[int, int]:
    """Copy a balanced WildFake sample from its official manifest onto disk as real/fake files."""
    records = wildfake_records_from_manifest(
        manifest,
        images_root,
        maximum=samples_per_class * 2,
        seed=seed,
        exclude_protected=exclude_protected,
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
        "--allow-protected-wildfake",
        action="store_true",
        help="Include COCO real and advanced DALL-E rows; never use for challenge submission",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_dir)

    if args.hf_dataset:
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
            shuffle_buffer=args.hf_shuffle_buffer,
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
        )
        print(f"WildFake: wrote {real} real / {fake} fake images to {output_root}")

    if not args.hf_dataset and not args.wildfake_manifest:
        raise ValueError("Pass --hf-dataset and/or --wildfake-manifest to materialize something.")


if __name__ == "__main__":
    main()
