from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    frequency_bins: int = 8
    dropout: float = 0.25
    freeze_backbone: bool = False
    """Train only the fusion head and leave the pretrained backbone untouched.

    Fine-tuning every weight on a hackathon-sized dataset lets the backbone overwrite its general
    visual knowledge with generator-specific trivia, which is why the EfficientNet baseline in
    checkpoints/merged/run_001 scored 0.71 on held-out generators while its training loss kept
    falling. Freezing makes that failure mode structurally impossible: the backbone cannot forget
    what it knew, so only the small head has to generalize. See Ojha et al., CVPR 2023.
    """
    use_frequency: bool = True
    """Include the radial-FFT branch. Set False to ablate it and measure what it actually adds."""


class FrequencyStatistics(nn.Module):
    """Extract compact radial-spectrum and color statistics from an image batch."""

    def __init__(
        self,
        bins: int = 8,
        mean: tuple[float, ...] = IMAGENET_MEAN,
        std: tuple[float, ...] = IMAGENET_STD,
    ) -> None:
        super().__init__()
        self.bins = bins
        # Buffers must match the normalization the incoming tensors were built with, otherwise the
        # un-normalization below reconstructs the wrong pixel values. CLIP backbones use their own
        # constants rather than the ImageNet ones.
        self.register_buffer("mean", torch.tensor(mean)[None, :, None, None])
        self.register_buffer("std", torch.tensor(std)[None, :, None, None])

    @property
    def output_dim(self) -> int:
        return self.bins * 2 + 6

    def forward(self, normalized: torch.Tensor) -> torch.Tensor:
        images = (normalized * self.std + self.mean).clamp(0, 1)
        color_mean = images.mean(dim=(-2, -1))
        color_std = images.std(dim=(-2, -1), unbiased=False)

        gray = images[:, 0] * 0.299 + images[:, 1] * 0.587 + images[:, 2] * 0.114
        spectrum = torch.log1p(
            torch.abs(torch.fft.fftshift(torch.fft.fft2(gray.float()), dim=(-2, -1)))
        )
        height, width = spectrum.shape[-2:]
        yy = torch.linspace(-1, 1, height, device=spectrum.device)[:, None]
        xx = torch.linspace(-1, 1, width, device=spectrum.device)[None, :]
        radius = torch.sqrt(xx.square() + yy.square()).clamp(max=1)

        radial_features: list[torch.Tensor] = []
        edges = torch.linspace(0, 1, self.bins + 1, device=spectrum.device)
        for index in range(self.bins):
            mask = (radius >= edges[index]) & (radius < edges[index + 1])
            values = spectrum[:, mask]
            radial_features.extend(
                [values.mean(dim=1, keepdim=True), values.std(dim=1, keepdim=True, unbiased=False)]
            )
        return torch.cat([color_mean, color_std, *radial_features], dim=1)


class TraceGuard(nn.Module):
    """A pretrained spatial backbone fused with explicit frequency evidence.

    The backbone is either fine-tuned end to end (the original behaviour, kept as a baseline) or
    frozen so that only the fusion head learns. See ModelConfig.freeze_backbone.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.config = config
        try:
            import timm
            from timm.data import resolve_model_data_config
        except ImportError as exc:  # pragma: no cover - dependency error is user-facing
            raise RuntimeError(
                "timm is required; install the project with `pip install -e .`"
            ) from exc

        # No explicit global_pool: each architecture's own default is the right one. EfficientNet
        # defaults to average pooling (unchanged from before), while ViT/CLIP models default to
        # their class token followed by the final norm - which is how CLIP embeddings are actually
        # meant to be read. Forcing "avg" here silently discarded that norm layer and mean-pooled
        # patch tokens instead, producing features unlike the ones CLIP was trained to produce.
        self.backbone = timm.create_model(
            config.backbone,
            pretrained=config.pretrained,
            num_classes=0,
        )
        backbone_dim = self.backbone.num_features

        # Each pretrained backbone declares the normalization and input size it was trained with.
        # Feeding CLIP weights ImageNet-normalized tensors silently degrades their features, so the
        # transforms and the frequency branch both follow this instead of hardcoded constants.
        data_config = resolve_model_data_config(self.backbone)
        self.data_mean: tuple[float, ...] = tuple(data_config["mean"])
        self.data_std: tuple[float, ...] = tuple(data_config["std"])
        self.input_size: int = int(data_config["input_size"][-1])

        if config.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self.backbone.eval()

        head_dim = backbone_dim
        if config.use_frequency:
            self.frequency = FrequencyStatistics(
                config.frequency_bins, self.data_mean, self.data_std
            )
            self.frequency_head = nn.Sequential(
                nn.LayerNorm(self.frequency.output_dim),
                nn.Linear(self.frequency.output_dim, 64),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            head_dim += 64
        else:
            self.frequency = None
            self.frequency_head = None

        self.classifier = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Dropout(config.dropout),
            nn.Linear(head_dim, 1),
        )

    def train(self, mode: bool = True):
        """Keep a frozen backbone in eval mode even when the rest of the model is training.

        Without this, `model.train()` would re-enable dropout and running-statistics updates inside
        the backbone, so a "frozen" backbone would still drift and produce different features from
        one epoch to the next.
        """
        super().train(mode)
        if self.config.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.config.freeze_backbone:
            with torch.no_grad():
                spatial = self.backbone(images)
        else:
            spatial = self.backbone(images)

        if self.frequency is not None:
            frequency = self.frequency_head(self.frequency(images))
            spatial = torch.cat([spatial, frequency], dim=1)
        return self.classifier(spatial).squeeze(1)

    def checkpoint_config(self) -> dict[str, object]:
        return asdict(self.config)

    def normalization(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Mean/std this backbone's pretrained weights expect, for building matching transforms."""
        return self.data_mean, self.data_std


def trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def frozen_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)
