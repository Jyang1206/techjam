from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .data import image_paths
from .inference import Predictor


def predict_directory(
    image_directory: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    *,
    device: str = "auto",
    tta: str = "robust",
) -> list[dict[str, object]]:
    paths = image_paths(image_directory)
    if not paths:
        raise ValueError(f"No supported images found in {image_directory}")
    predictor = Predictor.from_checkpoint(checkpoint, device=device)
    results = []
    for path in paths:
        try:
            with Image.open(path) as source:
                score = predictor.score_image(source.convert("RGB"), tta=tta)
            results.append({"image_path": str(path), "pred": score})
        except (OSError, ValueError) as exc:
            results.append({"image_path": str(path), "pred": None, "error": str(exc)})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score every image in a directory.")
    parser.add_argument("image_directory", help="Directory searched recursively for images")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--output", default="predictions.json")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--tta", choices=("none", "robust"), default="robust")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = predict_directory(
        args.image_directory,
        args.checkpoint,
        args.output,
        device=args.device,
        tta=args.tta,
    )
    failures = sum(row.get("pred") is None for row in results)
    print(f"Wrote {len(results)} predictions to {args.output} ({failures} failures).")


if __name__ == "__main__":
    main()
