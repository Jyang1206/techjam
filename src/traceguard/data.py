from __future__ import annotations

import csv
import io
import random
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFile
from torch.utils.data import Dataset, IterableDataset, get_worker_info

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    group: str | None = None
    """Best-effort generator/source identity, e.g. "wildfake__Diffusion_based__SD" or
    "hf__SID_Set". Used by group_disjoint_split to hold out whole generators rather than random
    individual images. None when no such identity is known or inferable."""


def infer_group(path: Path) -> str:
    """Best-effort group key parsed from a materialized filename (see materialize.py).

    Recognizes traceguard-materialize's own naming conventions
    (wildfake__<generator>__<architecture>__<name>, hf__<dataset_slug>__<identifier>) so a
    generator-disjoint split still works after re-discovering images from a plain folder, where
    per-record metadata from the original manifest/dataset no longer exists. Falls back to
    "unknown" for anything else - such records will all land in a single group, effectively
    falling back to per-image behavior for that subset when used with group_disjoint_split.
    """
    parts = path.name.split("__")
    if parts[0] == "wildfake" and len(parts) >= 3:
        return "__".join(parts[:3])
    if parts[0] == "hf" and len(parts) >= 2:
        return "__".join(parts[:2])
    return "unknown"


def image_paths(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def child_directory(root: str | Path, name: str) -> Path:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    match = next(
        (
            child
            for child in root.iterdir()
            if child.is_dir() and child.name.casefold() == name.casefold()
        ),
        None,
    )
    if match is None:
        raise FileNotFoundError(f"Could not find a `{name}` directory in {root}")
    return match


def find_split_directory(root: str | Path, split: str) -> Path:
    root = Path(root)
    candidates = [root, *sorted(path for path in root.rglob("*") if path.is_dir())]
    for candidate in candidates:
        if candidate.name.casefold() != split.casefold():
            continue
        child_names = {path.name.casefold() for path in candidate.iterdir() if path.is_dir()}
        if {"real", "fake"}.issubset(child_names):
            return candidate
    raise FileNotFoundError(
        f"Could not find a `{split}` split containing REAL/ and FAKE/ under {root}"
    )


def discover_labeled_images(root: str | Path) -> list[ImageRecord]:
    root = Path(root)
    real_directory = child_directory(root, "real")
    fake_directory = child_directory(root, "fake")
    records = [ImageRecord(path, 0, infer_group(path)) for path in image_paths(real_directory)]
    records.extend(ImageRecord(path, 1, infer_group(path)) for path in image_paths(fake_directory))
    if not records:
        raise ValueError(f"No images found. Expected `{root / 'real'}` and `{root / 'fake'}`.")
    if not any(record.label == 0 for record in records) or not any(
        record.label == 1 for record in records
    ):
        raise ValueError("Both real/ and fake/ must contain at least one supported image.")
    return records


def limit_records(
    records: Sequence[ImageRecord], maximum: int | None, seed: int
) -> list[ImageRecord]:
    if maximum is None or maximum >= len(records):
        return list(records)
    if maximum < 2:
        raise ValueError("A labeled sample limit must be at least 2")
    generator = random.Random(seed)
    selected: list[ImageRecord] = []
    per_label = maximum // 2
    for label in (0, 1):
        group = [record for record in records if record.label == label]
        generator.shuffle(group)
        selected.extend(group[:per_label])
    if len(selected) < maximum:
        remaining = [record for record in records if record not in selected]
        generator.shuffle(remaining)
        selected.extend(remaining[: maximum - len(selected)])
    generator.shuffle(selected)
    return selected


def download_kaggle_dataset(handle: str) -> Path:
    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - dependency error is user-facing
        raise RuntimeError("Install Kaggle support with `pip install -e .`") from exc
    return Path(kagglehub.dataset_download(handle))


def _wildfake_group(row: dict[str, str]) -> str:
    generator = row.get("Generator", "unknown").strip() or "unknown"
    architecture = row.get("Architecture", "unknown").strip() or "unknown"
    return f"wildfake__{generator}__{architecture}"


def _wildfake_protected_row(row: dict[str, str]) -> bool:
    label = int(float(row["IsFake"]))
    searchable = " ".join(
        row.get(column, "")
        for column in ("Generator", "Architecture", "Weight", "Category", "Image_path")
    ).casefold()
    is_advanced = row.get("IsAdvanced", "0").strip().casefold() in {"1", "1.0", "true"}
    return (label == 0 and "coco" in searchable) or (
        label == 1 and "dalle" in searchable and is_advanced
    )


def _wildfake_image_path(images_root: Path, raw_path: str) -> Path:
    normalized = raw_path.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    relative = Path(normalized)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe WildFake image path: {raw_path}")
    if relative.parts and relative.parts[0].casefold() == "images":
        relative = Path(*relative.parts[1:])
    return images_root / relative


def wildfake_records_from_manifest(
    manifest: str | Path,
    images_root: str | Path,
    *,
    maximum: int,
    seed: int,
    exclude_protected: bool = True,
) -> list[ImageRecord]:
    """Build a balanced reservoir sample from an official WildFake split manifest."""
    if maximum < 2:
        raise ValueError("WildFake sample limit must be at least 2")
    manifest = Path(manifest)
    images_root = Path(images_root)
    if not manifest.is_file():
        raise FileNotFoundError(f"WildFake manifest not found: {manifest}")
    if not images_root.is_dir():
        raise FileNotFoundError(f"WildFake Images directory not found: {images_root}")

    targets = {0: maximum // 2, 1: maximum - maximum // 2}
    reservoirs: dict[int, list[ImageRecord]] = {0: [], 1: []}
    seen = {0: 0, 1: 0}
    generators = {0: random.Random(seed), 1: random.Random(seed + 1)}
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"IsFake", "Image_path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"WildFake manifest must contain columns: {sorted(required)}")
        for row in reader:
            if exclude_protected and _wildfake_protected_row(row):
                continue
            label = int(float(row["IsFake"]))
            if label not in (0, 1):
                continue
            image_path = _wildfake_image_path(images_root, row["Image_path"])
            if not image_path.is_file():
                continue
            seen[label] += 1
            record = ImageRecord(image_path, label, _wildfake_group(row))
            reservoir = reservoirs[label]
            if len(reservoir) < targets[label]:
                reservoir.append(record)
                continue
            replacement = generators[label].randrange(seen[label])
            if replacement < targets[label]:
                reservoir[replacement] = record

    records = reservoirs[0] + reservoirs[1]
    if len(records) < maximum:
        raise ValueError(
            f"WildFake manifest resolved {len(records)} of {maximum} requested images. "
            "Download more matching image archives or lower the sample limit."
        )
    random.Random(seed).shuffle(records)
    return records


def stratified_split(
    records: Sequence[ImageRecord], validation_fraction: float, seed: int
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    generator = random.Random(seed)
    train: list[ImageRecord] = []
    validation: list[ImageRecord] = []
    for label in (0, 1):
        group = [record for record in records if record.label == label]
        generator.shuffle(group)
        count = max(1, round(len(group) * validation_fraction))
        validation.extend(group[:count])
        train.extend(group[count:])
    generator.shuffle(train)
    generator.shuffle(validation)
    return train, validation


def group_disjoint_split(
    records: Sequence[ImageRecord], validation_fraction: float, seed: int
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Like stratified_split, but holds out whole generator/source groups for validation instead
    of individual images.

    A random image-level split can put near-identical images from the same generator on both
    sides, so a high validation score may just mean "this generator's style is memorized" rather
    than genuine generalization. This holds out entire groups (parsed via each record's `group`
    field, falling back to infer_group(record.path) when not set) so validation only ever contains
    generators/sources absent from training - a real test of whether the model generalizes beyond
    what it trained on, not just whether it recognizes held-back examples of the same thing.

    Records with no identifiable group (`group` is None and infer_group returns "unknown") are all
    treated as one group and will land entirely on one side - effectively falling back to
    per-image behavior for that unlabeled subset. Raises if a label has fewer than 2 distinct
    groups, since a group-disjoint split is meaningless with nothing to hold out.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    generator = random.Random(seed)
    train: list[ImageRecord] = []
    validation: list[ImageRecord] = []
    for label in (0, 1):
        label_records = [record for record in records if record.label == label]
        by_group: dict[str, list[ImageRecord]] = {}
        for record in label_records:
            key = record.group or infer_group(record.path)
            by_group.setdefault(key, []).append(record)
        if len(by_group) < 2:
            raise ValueError(
                f"Only {len(by_group)} distinct group(s) found for label {label} - a "
                "group-disjoint split needs at least 2 to hold one out. Use stratified_split "
                "instead if your data has no recoverable generator/source identity."
            )
        group_keys = list(by_group)
        generator.shuffle(group_keys)
        target = round(len(label_records) * validation_fraction)
        validation_keys: set[str] = set()
        validation_count = 0
        for key in group_keys:
            if validation_count >= target:
                break
            validation_keys.add(key)
            validation_count += len(by_group[key])
        for key, items in by_group.items():
            (validation if key in validation_keys else train).extend(items)
    generator.shuffle(train)
    generator.shuffle(validation)
    return train, validation


class LabeledImageDataset(Dataset):
    def __init__(self, records: Sequence[ImageRecord], transform: Callable | None = None) -> None:
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as source:
            image = source.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, record.label, str(record.path)


def decode_dataset_image(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        with Image.open(io.BytesIO(value)) as source:
            return source.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes"):
            return decode_dataset_image(value["bytes"])
        if value.get("path"):
            with Image.open(value["path"]) as source:
                return source.convert("RGB")
    if isinstance(value, (str, Path)):
        with Image.open(value) as source:
            return source.convert("RGB")
    raise TypeError(f"Unsupported Hugging Face image value: {type(value).__name__}")


class HuggingFaceStreamDataset(IterableDataset):
    """Stream an image dataset from the Hub and convert multiclass labels to binary."""

    def __init__(
        self,
        dataset_id: str,
        *,
        split: str,
        transform: Callable | None,
        config: str | None = None,
        image_column: str = "image",
        label_column: str = "label",
        id_column: str = "img_id",
        positive_labels: Sequence[int] = (1, 2),
        max_samples: int | None = None,
        shuffle_buffer: int = 0,
        seed: int = 42,
    ) -> None:
        self.dataset_id = dataset_id
        self.config = config
        self.split = split
        self.transform = transform
        self.image_column = image_column
        self.label_column = label_column
        self.id_column = id_column
        self.positive_labels = frozenset(positive_labels)
        self.max_samples = max_samples
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def _source(self):
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - dependency error is user-facing
            raise RuntimeError("Install Hugging Face support with `pip install -e .`") from exc
        source = load_dataset(
            self.dataset_id,
            self.config,
            split=self.split,
            streaming=True,
        )
        if self.shuffle_buffer:
            source = source.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)
        if self.max_samples is not None:
            source = source.take(self.max_samples)
        worker = get_worker_info()
        if worker is not None:
            source = source.shard(num_shards=worker.num_workers, index=worker.id)
        return source

    def __iter__(self):
        warning_count = 0
        for index, row in enumerate(self._source()):
            try:
                image = decode_dataset_image(row[self.image_column])
                label = int(row[self.label_column] in self.positive_labels)
                identifier = str(row.get(self.id_column, f"{self.split}_{index}"))
                if self.transform:
                    image = self.transform(image)
                yield image, label, f"hf://{self.dataset_id}/{self.split}/{identifier}"
            except (KeyError, OSError, TypeError, ValueError) as exc:
                if warning_count < 5:
                    warnings.warn(f"Skipping unreadable row {index}: {exc}", stacklevel=2)
                    warning_count += 1

    def __len__(self) -> int:
        if self.max_samples is None:
            raise TypeError("An unbounded streaming dataset has no local length")
        return self.max_samples
