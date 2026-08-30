import numpy as np
import pytest

pytest.importorskip("PIL")
from PIL import Image

from traceguard.transforms import (
    CLIP_MEAN,
    ROBUSTNESS_SUITE,
    apply_degradation,
    build_eval_transform,
    normalization_for_backbone,
)


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


def test_clip_backbone_uses_clip_normalization():
    assert normalization_for_backbone("vit_base_patch16_clip_224.openai") == "clip"
    assert normalization_for_backbone("efficientnet_b0") == "imagenet"
    normalized = build_eval_transform(normalization="clip")(Image.new("RGB", (256, 256), "white"))
    expected = [(1 - mean) / std for mean, std in zip(CLIP_MEAN, (0.26862954, 0.26130258, 0.27577711))]
    assert np.allclose(normalized.mean(dim=(1, 2)).numpy(), expected, atol=1e-5)


def test_clip_eval_can_preserve_native_crop_percentage():
    transform = build_eval_transform(normalization="clip", crop_pct=1.0)
    resized = transform.transforms[0]
    assert resized.size == 224
