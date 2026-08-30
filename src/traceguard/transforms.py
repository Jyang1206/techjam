from __future__ import annotations

import io
import random
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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


def build_train_transform(
    image_size: int = IMAGE_SIZE,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
    degradation_probability: float = 0.8,
) -> Callable[[Image.Image], object]:
    """Training pipeline: degrade, crop, flip, normalize.

    `mean`/`std` must match what the chosen backbone's pretrained weights expect - pass
    `TraceGuard.normalization()` rather than assuming ImageNet constants, since CLIP backbones use
    their own. `degradation_probability` is the share of training images that get a random
    real-world degradation applied; the NTIRE 2026 robustness challenge found that training through
    these transforms mattered more than any post-hoc robustification.
    """
    from torchvision import transforms

    return transforms.Compose(
        [
            RandomRobustnessTransform(degradation_probability),
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_eval_transform(
    image_size: int = IMAGE_SIZE,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
) -> Callable[[Image.Image], object]:
    """Deterministic pipeline used for validation, evaluation, and inference.

    See build_train_transform for why `mean`/`std` should come from the model rather than defaults.
    """
    from torchvision import transforms

    resize_size = round(image_size / 0.875)
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
