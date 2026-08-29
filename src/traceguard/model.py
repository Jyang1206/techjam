from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    frequency_bins: int = 8
    dropout: float = 0.25


class FrequencyStatistics(nn.Module):
    """Extract compact radial-spectrum and color statistics from an image batch."""

    def __init__(self, bins: int = 8) -> None:
        super().__init__()
        self.bins = bins
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

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
    """A lightweight spatial backbone fused with explicit frequency evidence."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.config = config
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - dependency error is user-facing
            raise RuntimeError(
                "timm is required; install the project with `pip install -e .`"
            ) from exc

        self.backbone = timm.create_model(
            config.backbone,
            pretrained=config.pretrained,
            num_classes=0,
            global_pool="avg",
        )
        backbone_dim = self.backbone.num_features
        self.frequency = FrequencyStatistics(config.frequency_bins)
        self.frequency_head = nn.Sequential(
            nn.LayerNorm(self.frequency.output_dim),
            nn.Linear(self.frequency.output_dim, 64),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(backbone_dim + 64),
            nn.Dropout(config.dropout),
            nn.Linear(backbone_dim + 64, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        spatial = self.backbone(images)
        frequency = self.frequency_head(self.frequency(images))
        return self.classifier(torch.cat([spatial, frequency], dim=1)).squeeze(1)

    def checkpoint_config(self) -> dict[str, object]:
        return asdict(self.config)


def trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
