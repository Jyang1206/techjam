import csv

from PIL import Image

from traceguard.materialize import materialize_wildfake


def create_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), "white").save(path)


def test_materialize_wildfake_writes_balanced_real_fake_folders(tmp_path):
    images = tmp_path / "Images"
    rows = [
        ["GAN_based", "BigGAN", "x", "x", 0, 1, "./GAN_based/fake1.jpg", 1],
        ["Diffusion_based", "dalle2", "x", "x", 1, 1, "./Diffusion_based/dalle_advanced.jpg", 2],
        ["Real", "imagenet", "x", "x", 0, 0, "./Real/real1.jpg", 3],
        ["Real", "coco", "x", "x", 0, 0, "./Real/coco.jpg", 4],
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

    output_root = tmp_path / "merged"
    real_count, fake_count = materialize_wildfake(
        manifest, images, output_root=output_root, samples_per_class=1, seed=9
    )

    assert (real_count, fake_count) == (1, 1)
    real_files = list((output_root / "real").iterdir())
    fake_files = list((output_root / "fake").iterdir())
    assert len(real_files) == 1
    assert len(fake_files) == 1
    # the protected COCO-real row and the advanced-DALL-E row must never be selectable
    assert "coco" not in real_files[0].name
    assert "dalle_advanced" not in fake_files[0].name
    # copies are independent files, not the same inode/path as the source
    assert real_files[0].parent == output_root / "real"
    # filenames encode generator identity for group_disjoint_split to recover later
    assert real_files[0].name.startswith("wildfake__Real__imagenet__")
    assert fake_files[0].name.startswith("wildfake__GAN_based__BigGAN__")


def test_materialize_wildfake_preserves_duplicate_basenames(tmp_path):
    images = tmp_path / "Images"
    rows = [
        ["Real", "imagenet", "x", "x", 0, 0, "./Real/imagenet/a/shared.jpg", 1],
        ["Real", "imagenet", "x", "x", 0, 0, "./Real/imagenet/b/shared.jpg", 2],
        ["Other_based", "MAGE", "x", "x", 0, 1, "./Other_based/MAGE/a/shared.jpg", 3],
        ["Other_based", "MAGE", "x", "x", 0, 1, "./Other_based/MAGE/b/shared.jpg", 4],
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

    output_root = tmp_path / "merged"
    counts = materialize_wildfake(
        manifest, images, output_root=output_root, samples_per_class=2, seed=42
    )

    assert counts == (2, 2)
    real_names = {path.name for path in (output_root / "real").iterdir()}
    fake_names = {path.name for path in (output_root / "fake").iterdir()}
    assert len(real_names) == 2
    assert len(fake_names) == 2
    assert all(name.endswith("__shared.jpg") for name in real_names | fake_names)
