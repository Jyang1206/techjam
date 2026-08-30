from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    frequency_bins: int = 8
    dropout: float = 0.25
    normalization: str = "imagenet"
    use_frequency: bool = True
    backbone_projection: bool = False
    classifier_layernorm: bool = True
    eval_crop_pct: float = 0.875
    low_resolution_size: int = 0


class FrequencyStatistics(nn.Module):
    """Extract compact radial-spectrum and color statistics from an image batch."""

    def __init__(
        self,
        bins: int = 8,
        *,
        mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        std: tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.bins = bins
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

        if config.backbone_projection:
            self.backbone = timm.create_model(config.backbone, pretrained=config.pretrained)
            backbone_dim = self.backbone.num_classes
        else:
            self.backbone = timm.create_model(
                config.backbone,
                pretrained=config.pretrained,
                num_classes=0,
                global_pool="avg",
            )
            backbone_dim = self.backbone.num_features
        self.backbone_dim = backbone_dim
        from .transforms import normalization_stats

        if config.use_frequency:
            mean, std = normalization_stats(config.normalization)
            self.frequency = FrequencyStatistics(config.frequency_bins, mean=mean, std=std)
            self.frequency_head = nn.Sequential(
                nn.LayerNorm(self.frequency.output_dim),
                nn.Linear(self.frequency.output_dim, 64),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            classifier_dim = backbone_dim + 64
        else:
            self.frequency = None
            self.frequency_head = None
            classifier_dim = backbone_dim
        classifier_layers: list[nn.Module] = []
        if config.classifier_layernorm:
            classifier_layers.append(nn.LayerNorm(classifier_dim))
        classifier_layers.extend([nn.Dropout(config.dropout), nn.Linear(classifier_dim, 1)])
        self.classifier = nn.Sequential(*classifier_layers)
        self.base_feature_dim = classifier_dim
        self.low_resolution_classifier: nn.Sequential | None = None
        if config.low_resolution_size > 0:
            low_resolution_layers: list[nn.Module] = []
            if config.classifier_layernorm:
                low_resolution_layers.append(nn.LayerNorm(backbone_dim))
            low_resolution_layers.extend(
                [nn.Dropout(config.dropout), nn.Linear(backbone_dim, 1, bias=False)]
            )
            self.low_resolution_classifier = nn.Sequential(*low_resolution_layers)
            nn.init.zeros_(self.low_resolution_classifier[-1].weight)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        spatial = self.backbone(images)
        low_resolution = None
        if self.low_resolution_classifier is not None:
            low_size = self.config.low_resolution_size
            reduced = F.interpolate(
                images,
                size=(low_size, low_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            restored = F.interpolate(
                reduced,
                size=images.shape[-2:],
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            low_resolution = self.backbone(restored)
        if self.frequency is not None and self.frequency_head is not None:
            frequency = self.frequency_head(self.frequency(images))
            spatial = torch.cat([spatial, frequency], dim=1)
        if low_resolution is not None:
            spatial = torch.cat([spatial, low_resolution], dim=1)
        return spatial

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        base_features = features[:, : self.base_feature_dim]
        logits = self.classifier(base_features).squeeze(1)
        if self.low_resolution_classifier is not None:
            low_resolution_features = features[:, self.base_feature_dim :]
            logits = logits + self.low_resolution_classifier(low_resolution_features).squeeze(1)
        return logits

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classify_features(self.extract_features(images))

    def checkpoint_config(self) -> dict[str, object]:
        return asdict(self.config)


def trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def total_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
