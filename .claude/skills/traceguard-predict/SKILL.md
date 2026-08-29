---
name: traceguard-predict
description: Run TraceGuard's required inference deliverable script (`traceguard-predict`) to score every image in a directory and produce the challenge's mandated `predictions.json` (a JSON list of `{"image_path", "pred"}` objects). Use this whenever the user wants to run inference/scoring on a folder of images for this project, generate submission predictions, test a trained checkpoint on new images, or asks how to run the required directory-to-JSON script.
---

# Run TraceGuard inference

This is the literal script the hackathon challenge requires as a deliverable: point it at a
directory of images, it recursively scans for supported image files and writes a JSON file with
one entry per image.

Run from the `techjam` repo root with its virtualenv active:

```bash
source .venv/bin/activate
traceguard-predict path/to/image_directory \
  --checkpoint checkpoints/<run-name>/best.pt \
  --output predictions.json
```

## What you need first

A trained checkpoint (`.pt` file) from the `traceguard-train` skill/command. There is no shipped
default checkpoint — model weights are gitignored and never committed to this repository, so a
fresh checkout has nothing at `checkpoints/best.pt` until someone trains locally. If no checkpoint
exists yet, run training first rather than expecting this command to work out of the box.

## Flags worth knowing

- `--tta {robust,none}` (default `robust`): `robust` averages the prediction across 4 views (clean,
  JPEG-70, half-resolution, 80%-crop) — this is the "redistribution-robust consensus" behavior
  described in the README. `none` scores only the original image, which is faster (1 forward pass
  instead of 4) but less robust to the transformations the challenge specifically tests for.
- `--device {auto,cpu,cuda,mps,...}` (default `auto`): auto-resolves to CUDA, then Apple's MPS, then
  CPU — you rarely need to set this explicitly.

## Output format (do not deviate from this — it's the exact contract the challenge grades against)

```json
[
  {"image_path": "path/to/image_directory/photo.jpg", "pred": 0.0831},
  {"image_path": "path/to/image_directory/render.png", "pred": 0.9472}
]
```

`pred` is the probability the image is AI-generated (0 = confidently real, 1 = confidently fake).
If an image fails to open/decode, its entry gets `"pred": null` plus an `"error"` field explaining
why, rather than crashing the whole run — check the console summary line
(`Wrote N predictions to ... (K failures)`) after running to see if anything needs attention.

## Note on the input directory

Unlike `traceguard-evaluate`, this command does **not** need a labeled `real/`/`fake/` folder
structure — it just needs any folder of images, since it has no ground truth to compare against and
only produces per-image scores, not accuracy metrics. If the user wants accuracy/robustness numbers
against known-labeled data, that's the `traceguard-evaluate` skill instead.
