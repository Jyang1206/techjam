from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from .model import ModelConfig, TraceGuard


def import_universal_fake_detect(head_path: str | Path, output_path: str | Path) -> Path:
    """Build a self-contained TraceGuard checkpoint from UniversalFakeDetect's official head."""
    config = ModelConfig(
        backbone="vit_large_patch14_clip_quickgelu_224.openai",
        pretrained=True,
        dropout=0.0,
        normalization="clip",
        use_frequency=False,
        backbone_projection=True,
        classifier_layernorm=False,
        eval_crop_pct=1.0,
    )
    model = TraceGuard(config)
    state = torch.load(head_path, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    linear = model.classifier[-1]
    if not isinstance(linear, nn.Linear):
        raise TypeError("Expected the final classifier layer to be linear")
    linear.load_state_dict(state)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model.checkpoint_config(),
            "threshold": 0.5,
            "temperature": 1.0,
            "epoch": 0,
            "validation_metrics": {},
            "training_source": "UniversalFakeDetect official CVPR 2023 pretrained head",
        },
        output_path,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Import UniversalFakeDetect's official CLIP head")
    parser.add_argument("head_path")
    parser.add_argument("output_path")
    args = parser.parse_args()
    output = import_universal_fake_detect(args.head_path, args.output_path)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
