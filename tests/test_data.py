import csv
import io
from pathlib import Path

import pytest

pytest.importorskip("torch")
from PIL import Image

from traceguard.data import (
    ImageRecord,
    child_directory,
    decode_dataset_image,
    discover_labeled_images,
    fake_generator_disjoint_split,
    find_split_directory,
    group_disjoint_split,
    image_paths,
    infer_group,
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


def test_wildfake_manifest_records_carry_generator_group(tmp_path):
    images = tmp_path / "Images"
    rows = [
        ["GAN_based", "BigGAN", "x", "x", 0, 1, "./GAN_based/fake.jpg", 1],
        ["Real", "imagenet", "x", "x", 0, 0, "./Real/real.jpg", 3],
    ]
    for row in rows:
        create_image(images / row[6].removeprefix("./"))
    manifest = tmp_path / "train_metadata.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["Generator", "Architecture", "Weight", "Category", "IsAdvanced", "IsFake",
             "Image_path", "Num"]
        )
        writer.writerows(rows)

    records = wildfake_records_from_manifest(manifest, images, maximum=2, seed=9)
    groups = {record.path.name: record.group for record in records}
    assert groups["fake.jpg"] == "wildfake__GAN_based__BigGAN"
    assert groups["real.jpg"] == "wildfake__Real__imagenet"


def test_infer_group_parses_materialize_naming_conventions():
    assert infer_group(Path("wildfake__Diffusion_based__SD__abc123.png")) == (
        "wildfake__Diffusion_based__SD"
    )
    assert infer_group(Path("hf__SID_Set__0000a1229c025d92.jpg")) == "hf__SID_Set"
    assert infer_group(Path("random_photo.jpg")) == "unknown"


def test_discover_labeled_images_infers_group_from_filenames(tmp_path):
    create_image(tmp_path / "real" / "wildfake__Real__laion5b__001.jpg")
    create_image(tmp_path / "real" / "wildfake__Real__imagenet__002.jpg")
    create_image(tmp_path / "fake" / "wildfake__Diffusion_based__SD__003.png")
    create_image(tmp_path / "fake" / "wildfake__GAN_based__BigGAN__004.png")

    records = discover_labeled_images(tmp_path)
    groups = {record.path.name: record.group for record in records}
    assert groups["wildfake__Real__laion5b__001.jpg"] == "wildfake__Real__laion5b"
    assert groups["wildfake__Diffusion_based__SD__003.png"] == "wildfake__Diffusion_based__SD"


def test_group_disjoint_split_holds_out_whole_groups(tmp_path):
    records = []
    for group_index in range(4):
        for label in (0, 1):
            for item in range(5):
                records.append(
                    ImageRecord(
                        Path(f"/fake/{label}_{group_index}_{item}.jpg"),
                        label,
                        f"group_{label}_{group_index}",
                    )
                )

    train, validation = group_disjoint_split(records, validation_fraction=0.25, seed=1)

    train_groups = {record.group for record in train}
    validation_groups = {record.group for record in validation}
    # no group should ever appear on both sides of the split
    assert not (train_groups & validation_groups)
    assert len(train) + len(validation) == len(records)
    assert validation  # something actually landed in validation


def test_group_disjoint_split_chooses_group_subset_closest_to_target():
    records = []
    for label in (0, 1):
        for group_index, size in enumerate((8, 5, 3, 2)):
            records.extend(
                ImageRecord(
                    Path(f"/{label}/{group_index}_{item}.jpg"),
                    label,
                    f"group_{label}_{group_index}",
                )
                for item in range(size)
            )

    _, validation = group_disjoint_split(records, validation_fraction=5 / 18, seed=42)

    # Five images per label is exactly reachable, so the splitter should not greedily overshoot.
    assert sum(record.label == 0 for record in validation) == 5
    assert sum(record.label == 1 for record in validation) == 5


def test_group_disjoint_split_requires_at_least_two_groups_per_label():
    records = [
        ImageRecord(Path("/fake/a.jpg"), 0, "only_group"),
        ImageRecord(Path("/fake/b.jpg"), 1, "only_group"),
    ]
    with pytest.raises(ValueError, match="distinct group"):
        group_disjoint_split(records, validation_fraction=0.2, seed=1)


def test_fake_generator_split_stratifies_reals_and_holds_out_fake_groups():
    records = []
    for source in ("real_faces", "real_objects"):
        records.extend(ImageRecord(Path(f"/{source}/{i}.jpg"), 0, source) for i in range(20))
    for generator in ("fake_a", "fake_b", "fake_c"):
        records.extend(ImageRecord(Path(f"/{generator}/{i}.jpg"), 1, generator) for i in range(10))

    train, validation = fake_generator_disjoint_split(records, 0.25, seed=42)

    train_fake_groups = {record.group for record in train if record.label == 1}
    validation_fake_groups = {record.group for record in validation if record.label == 1}
    assert not train_fake_groups & validation_fake_groups
    assert {record.group for record in train if record.label == 0} == {
        "real_faces",
        "real_objects",
    }
    assert {record.group for record in validation if record.label == 0} == {
        "real_faces",
        "real_objects",
    }
