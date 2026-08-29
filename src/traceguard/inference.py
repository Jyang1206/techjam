from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .model import ModelConfig, TraceGuard
from .transforms import apply_degradation, build_eval_transform, ensure_rgb

TTA_PRESETS: dict[str, list[tuple[str, float]]] = {
    "none": [("clean", 1.0)],
    "robust": [("clean", 1.0), ("jpeg", 70), ("resize", 0.5), ("crop", 0.8)],
}


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Predictor:
    def __init__(
        self,
        model: TraceGuard,
        *,
        device: torch.device,
        threshold: float = 0.5,
        temperature: float = 1.0,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.threshold = threshold
        self.temperature = max(float(temperature), 1e-6)
        self.transform = build_eval_transform()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, device: str = "auto") -> Predictor:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. Train a model or pass --checkpoint."
            )
        target_device = resolve_device(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = replace(ModelConfig(**checkpoint.get("model_config", {})), pretrained=False)
        model = TraceGuard(config)
        model.load_state_dict(checkpoint["model_state"])
        return cls(
            model,
            device=target_device,
            threshold=float(checkpoint.get("threshold", 0.5)),
            temperature=float(checkpoint.get("temperature", 1.0)),
        )

    @torch.inference_mode()
    def score_image(self, image: Image.Image, tta: str = "robust") -> float:
        if tta not in TTA_PRESETS:
            raise ValueError(f"Unknown TTA preset: {tta}")
        image = ensure_rgb(image)
        tensors = [
            self.transform(apply_degradation(image, name, value))
            for name, value in TTA_PRESETS[tta]
        ]
        batch = torch.stack(tensors).to(self.device)
        logits = self.model(batch) / self.temperature
        return float(torch.sigmoid(logits).mean().cpu())

    def stability_profile(self, image: Image.Image) -> list[dict[str, float | str]]:
        probes = [
            ("Clean", "clean", 1.0),
            ("JPEG 70", "jpeg", 70),
            ("Blur 1.0", "blur", 1.0),
            ("Resize 0.5x", "resize", 0.5),
            ("Crop 80%", "crop", 0.8),
        ]
        return [
            {
                "condition": label,
                "aigc_probability": self.score_image(
                    apply_degradation(image, transform, value), tta="none"
                ),
            }
            for label, transform, value in probes
        ]

    def predict_image(self, image: Image.Image, tta: str = "robust") -> dict[str, float | str]:
        score = self.score_image(image, tta=tta)
        return {
            "label": "AIGC" if score >= self.threshold else "authentic",
            "pred": score,
            "threshold": self.threshold,
        }


def score_spread(profile: list[dict[str, float | str]]) -> float:
    scores = np.asarray([row["aigc_probability"] for row in profile], dtype=float)
    return float(scores.max() - scores.min())
