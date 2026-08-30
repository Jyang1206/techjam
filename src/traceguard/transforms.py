from __future__ import annotations

import io
import random
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
NORMALIZATION_STATS = {
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
    "clip": (CLIP_MEAN, CLIP_STD),
}

ROBUSTNESS_SUITE: dict[str, tuple[float, ...]] = {
    "jpeg": (90, 70, 50, 30),
    "blur": (0.5, 1.0, 2.0),
    "resize": (0.5, 0.25),
    "noise": (0.02, 0.05, 0.10),
    "color": (0.8, 1.2),
    "crop": (0.8,),
}


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A"))
        return background
    return image.convert("RGB")


def apply_degradation(
    image: Image.Image,
    transform: str,
    value: float,
    *,
    seed: int = 0,
) -> Image.Image:
    image = ensure_rgb(image)
    if transform == "clean":
        return image.copy()
    if transform == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(value))
        buffer.seek(0)
        with Image.open(buffer) as encoded:
            return encoded.convert("RGB")
    if transform == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(value)))
    if transform == "resize":
        original = image.size
        reduced = (max(1, round(original[0] * value)), max(1, round(original[1] * value)))
        return image.resize(reduced, Image.Resampling.BILINEAR).resize(
            original, Image.Resampling.BICUBIC
        )
    if transform == "noise":
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        noise = np.random.default_rng(seed).normal(0, value, size=pixels.shape)
        return Image.fromarray(np.uint8(np.clip(pixels + noise, 0, 1) * 255))
    if transform == "color":
        adjusted = ImageEnhance.Brightness(image).enhance(value)
        adjusted = ImageEnhance.Contrast(adjusted).enhance(value)
        return ImageEnhance.Color(adjusted).enhance(value)
    if transform == "crop":
        width, height = image.size
        crop_width, crop_height = round(width * value), round(height * value)
        left, top = (width - crop_width) // 2, (height - crop_height) // 2
        return image.crop((left, top, left + crop_width, top + crop_height)).resize(
            (width, height), Image.Resampling.BICUBIC
        )
    raise ValueError(f"Unknown transform: {transform}")


class RandomRobustnessTransform:
    def __init__(self, probability: float = 0.8) -> None:
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        image = ensure_rgb(image)
        if random.random() > self.probability:
            return image
        name = random.choice(tuple(ROBUSTNESS_SUITE))
        value = random.choice(ROBUSTNESS_SUITE[name])
        return apply_degradation(image, name, value, seed=random.randrange(2**32))


class RandomLowResolutionTransform:
    """Expose both labels to realistic low-resolution and recompression artifacts."""

    def __init__(self, probability: float = 0.8) -> None:
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        image = ensure_rgb(image)
        if random.random() > self.probability:
            return image
        target_size = random.choice((32, 56, 112))
        scale = min(1.0, target_size / max(min(image.size), 1))
        image = apply_degradation(image, "resize", scale)
        if random.random() < 0.5:
            image = apply_degradation(image, "jpeg", random.choice((50, 70, 90)))
        return image


def normalization_for_backbone(backbone: str) -> str:
    """Choose the native normalization used by the requested pretrained visual encoder."""
    return "clip" if "clip" in backbone.casefold() or "mclip" in backbone.casefold() else "imagenet"


def normalization_stats(name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    try:
        return NORMALIZATION_STATS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown normalization: {name}") from exc


def build_train_transform(
    image_size: int = IMAGE_SIZE,
    *,
    normalization: str = "imagenet",
    robustness_profile: str = "standard",
) -> Callable[[Image.Image], object]:
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    mean, std = normalization_stats(normalization)
    interpolation = (
        InterpolationMode.BICUBIC if normalization == "clip" else InterpolationMode.BILINEAR
    )
    if robustness_profile == "standard":
        robustness_transform = RandomRobustnessTransform()
    elif robustness_profile == "low_resolution":
        robustness_transform = RandomLowResolutionTransform()
    elif robustness_profile == "none":
        robustness_transform = ensure_rgb
    else:
        raise ValueError(f"Unknown training robustness profile: {robustness_profile}")
    return transforms.Compose(
        [
            robustness_transform,
            transforms.RandomResizedCrop(
                image_size, scale=(0.75, 1.0), interpolation=interpolation
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_eval_transform(
    image_size: int = IMAGE_SIZE,
    *,
    normalization: str = "imagenet",
    crop_pct: float = 0.875,
) -> Callable[[Image.Image], object]:
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    mean, std = normalization_stats(normalization)
    if not 0 < crop_pct <= 1:
        raise ValueError("crop_pct must be in (0, 1]")
    resize_size = round(image_size / crop_pct)
    interpolation = (
        InterpolationMode.BICUBIC if normalization == "clip" else InterpolationMode.BILINEAR
    )
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
