import numpy as np
import pytest

pytest.importorskip("PIL")
from PIL import Image

from traceguard.transforms import ROBUSTNESS_SUITE, apply_degradation


@pytest.fixture
def image():
    pixels = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    return Image.fromarray(pixels)


@pytest.mark.parametrize(
    ("name", "value"),
    [(name, values[0]) for name, values in ROBUSTNESS_SUITE.items()],
)
def test_degradations_preserve_size_and_rgb(image, name, value):
    result = apply_degradation(image, name, value)
    assert result.size == image.size
    assert result.mode == "RGB"


def test_noise_is_deterministic_for_evaluation(image):
    first = np.asarray(apply_degradation(image, "noise", 0.05, seed=7))
    second = np.asarray(apply_degradation(image, "noise", 0.05, seed=7))
    assert np.array_equal(first, second)


def test_unknown_degradation_fails_loudly(image):
    with pytest.raises(ValueError, match="Unknown transform"):
        apply_degradation(image, "mystery", 1)
