import csv
import io
from pathlib import Path

import pytest

pytest.importorskip("torch")
from PIL import Image

from traceguard.data import (
    child_directory,
    decode_dataset_image,
    discover_labeled_images,
    find_split_directory,
    image_paths,
    limit_records,
    stratified_split,
    wildfake_records_from_manifest,
)


def create_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), "white").save(path)


def test_discovery_and_stratified_split(tmp_path):
    for label in ("real", "fake"):
        for index in range(4):
            create_image(tmp_path / label / f"{index}.png")
    records = discover_labeled_images(tmp_path)
    train, validation = stratified_split(records, validation_fraction=0.25, seed=3)
    assert len(records) == 8
    assert len(train) == 6
    assert len(validation) == 2
    assert {record.label for record in validation} == {0, 1}


def test_image_paths_ignores_non_images(tmp_path):
    create_image(tmp_path / "photo.jpg")
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")
    assert [path.name for path in image_paths(tmp_path)] == ["photo.jpg"]


def test_hugging_face_image_bytes_are_decoded():
    buffer = io.BytesIO()
    Image.new("RGB", (9, 7), "blue").save(buffer, format="PNG")
    result = decode_dataset_image({"bytes": buffer.getvalue(), "path": None})
    assert result.mode == "RGB"
    assert result.size == (9, 7)


def test_uppercase_cifake_structure_is_discovered(tmp_path):
    for split in ("train", "test"):
        for label in ("REAL", "FAKE"):
            create_image(tmp_path / "CIFAKE" / split / label / f"{label}.jpg")
    train_root = find_split_directory(tmp_path, "train")
    records = discover_labeled_images(train_root)
    assert child_directory(train_root, "real").name == "REAL"
    assert sorted(record.label for record in records) == [0, 1]


def test_record_limit_stays_balanced(tmp_path):
    for label in ("real", "fake"):
        for index in range(6):
            create_image(tmp_path / label / f"{index}.png")
    limited = limit_records(discover_labeled_images(tmp_path), maximum=6, seed=4)
    assert len(limited) == 6
    assert sum(record.label for record in limited) == 3


def test_wildfake_manifest_is_balanced_and_protected_rows_are_excluded(tmp_path):
    images = tmp_path / "Images"
    rows = [
        ["GAN_based", "BigGAN", "x", "x", 0, 1, "./GAN_based/fake.jpg", 1],
        ["Diffusion_based", "dalle2", "x", "x", 1, 1, "./Diffusion_based/dalle.jpg", 2],
        ["Real", "imagenet", "x", "x", 0, 0, "./Real/real.jpg", 3],
        ["Real", "coco", "x", "x", 0, 0, "./Real/coco.jpg", 4],
    ]
    for row in rows:
        create_image(images / row[6].removeprefix("./"))
    manifest = tmp_path / "train_metadata.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Generator",
                "Architecture",
                "Weight",
                "Category",
                "IsAdvanced",
                "IsFake",
                "Image_path",
                "Num",
            ]
        )
        writer.writerows(rows)

    records = wildfake_records_from_manifest(manifest, images, maximum=2, seed=9)
    assert sorted(record.path.name for record in records) == ["fake.jpg", "real.jpg"]
    assert sorted(record.label for record in records) == [0, 1]
