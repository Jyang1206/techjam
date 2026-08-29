from pathlib import Path

from traceguard.evaluate import _relative_image_path


def test_relative_image_path_does_not_expose_dataset_root(tmp_path):
    image = tmp_path / "REAL" / "image.jpg"

    assert _relative_image_path(image, tmp_path) == "REAL/image.jpg"


def test_relative_image_path_falls_back_to_name_outside_dataset_root(tmp_path):
    image = Path(tmp_path.anchor) / "outside" / "image.jpg"

    assert _relative_image_path(image, tmp_path) == "image.jpg"
