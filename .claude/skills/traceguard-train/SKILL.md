---
name: traceguard-train
description: Train the TraceGuard AIGC image detector (`traceguard-train` CLI) on one of its supported data sources — a local real/fake folder, SID_Set via Hugging Face, CIFAKE via Kaggle, or WildFake via ModelScope. Use this whenever the user wants to train, retrain, fine-tune, or kick off a training run for TraceGuard in this repo, asks which dataset flag to use, or wants to combine multiple datasets into one training run. Trigger on mentions of training on CIFAKE, SID_Set, or WildFake, or phrases like "train the model", "start a training run", "how do I train on my own images".
---

# Train TraceGuard

`traceguard-train` fine-tunes the full two-branch model (EfficientNet-B0 + frequency
statistics, ~4M params) end-to-end — nothing is frozen. Before anything else, make sure you're in the `techjam` repo root and its virtualenv is active:

```bash
source .venv/bin/activate   # create it first with: python -m venv .venv && pip install -e ".[dev]"
```

## Pick exactly one data source

The code enforces exactly one of these four per run (`train.py` raises an error otherwise — it
will not silently combine sources):

| Source | Flag | Notes |
|---|---|---|
| Local folder | `data_dir` (positional) | Needs `real/` and `fake/` subfolders |
| SID_Set | `--hf-dataset saberzl/SID_Set` | Streams from Hugging Face, ~140GB dataset, never fully downloaded |
| CIFAKE | `--kaggle-dataset birdy654/cifake-real-and-ai-generated-synthetic-images` | Auto-downloads via kagglehub, uses the dataset's own train/test split |
| WildFake | `--wildfake-root data/WildFake` | Requires manually downloading the manifest + chosen image archives first (full dataset is ~1.29TB) |

Example commands:

```bash
# Local folder
traceguard-train data/train --epochs 12 --batch-size 32 --output-dir checkpoints/local/run_001

# SID_Set (streamed)
traceguard-train --hf-dataset saberzl/SID_Set \
  --epochs 5 --batch-size 24 --workers 2 --output-dir checkpoints/sid/run_001

# CIFAKE (Kaggle)
traceguard-train --kaggle-dataset birdy654/cifake-real-and-ai-generated-synthetic-images \
  --epochs 5 --batch-size 64 --workers 2 --output-dir checkpoints/cifake/run_001

# WildFake (after downloading manifests + chosen image archives — see project README)
traceguard-train --wildfake-root data/WildFake \
  --max-train-samples 20000 --max-validation-samples 4000 \
  --epochs 5 --batch-size 24 --workers 2 --output-dir checkpoints/wildfake/run_001
```

Default sample caps are `--max-train-samples 20000 --max-validation-samples 4000`. Pass `0` to use
every available image (only meaningful for Kaggle/HF; WildFake requires a positive value since it
streams a bounded reservoir sample).

## Never lose a good checkpoint by accident

`--output-dir` has no memory of previous runs on its own — a fresh `traceguard-train` call resets
its "best so far" tracker to nothing, so if you reuse an `--output-dir` that already has results,
the new run's very first epoch would silently replace the old `best.pt`/`history.json`, even if the
old run scored better. To prevent exactly this, **the training code now refuses to start if
`--output-dir` already contains a `best.pt` or `history.json`**, and tells you so instead of
overwriting silently:

```
FileExistsError: checkpoints/cifake/run_001 already has training results (...). Refusing to
overwrite them. Pick a new --output-dir for this run, e.g. checkpoints/<source>/run_XXX ...
```

So: **always pick a new `--output-dir` for each run** rather than reusing one — this is required
now, not just good practice. The convention is **`checkpoints/<source>/run_XXX`** — group every run
on the same dataset under one source folder, numbering runs sequentially within it:

- `checkpoints/cifake/run_001`, `checkpoints/cifake/run_002` (same source, e.g. more epochs the
  second time)
- `checkpoints/wildfake/run_001`, `checkpoints/sid/run_001`
- `checkpoints/merged_wildfake_sidset/run_001` (a manually merged multi-source local folder — the
  "source" name here just describes what's in the merged folder)

Before starting a new run, check what run numbers already exist under that source
(`ls checkpoints/<source>/`) and pick the next one — the guard above will catch it anyway if a
number gets reused by mistake. If you genuinely want to replace a run's results in place (rare —
usually you'd rather keep both and compare via the `traceguard-status` skill), pass `--overwrite`
explicitly instead of picking a new run number.

## Wanting data from multiple sources combined?

There is no `--multi-source` flag. The one supported way to train on a genuine mix without writing
new code is to **materialize each source into one shared local folder first** (download/export each
dataset's images into the same `data/train/real/`, `data/train/fake/` pool), then run plain
local-folder mode against that merged folder. Before doing this, be aware of two things worth
mentioning to the user if relevant:

- **SID_Set's "fake" label conflates two different things**: fully-synthetic images (label 1) and
  AI-tampered real photos (label 2), both collapsed to `1`. CIFAKE and WildFake's fakes are purely
  fully-synthetic. Mixing sources means your "fake" class becomes slightly less homogeneous.
- **CIFAKE images are only 32×32 pixels**, while SID_Set/WildFake are full resolution. Mixing them
  risks the model learning "blurry/upscaled → CIFAKE" as a shortcut rather than a genuine forensic
  signal. A safer alternative to blending it in is to hold CIFAKE out entirely and use it as a
  cross-distribution generalization *test set* instead (pair this with the `traceguard-evaluate`
  skill) rather than training data.

## Things this skill should always flag before running

- **Never pass `--allow-protected-wildfake`** for anything meant for challenge submission. WildFake
  training excludes real-COCO and advanced-DALL·E rows by default because that is the challenge's
  held-out demonstration set — training on it would be leakage against the actual scoring benchmark.
- **The Kaggle path's "validation" split is CIFAKE's own `test/` folder**, and it gets used *during*
  training to pick the best checkpoint (`best.pt`) and the decision threshold. That means it is not
  a clean, untouched test set anymore — don't also point `traceguard-evaluate` at that same folder
  and call the result a final/reportable number. Point final evaluation at data the training run
  never touched.
- **Apple Silicon (MPS) is auto-detected and works**, but the code's mixed-precision path
  (`torch.autocast`/`GradScaler`) is gated to `device.type == "cuda"` only — an M-series Mac trains
  in full FP32. This is a real but usually acceptable slowdown for a model this small; if timing is
  a concern, run one epoch on a small `--max-train-samples` slice first to estimate total run time
  before committing to a big job.

## What gets produced

- `<output-dir>/best.pt` — the checkpoint from whichever epoch had the highest validation ROC-AUC,
  including model config, chosen threshold, and that epoch's full metrics.
- `<output-dir>/history.json` — every epoch's train/validation loss plus accuracy, balanced
  accuracy, precision, recall, F1, ROC-AUC, Brier score, FPR, FNR. Written once, after the last
  epoch finishes (not incrementally) — use the `traceguard-status` skill to inspect this afterward.

Training prints one summary line per epoch to the terminal it was launched from; that live output
is not captured anywhere else, so check the launching terminal for progress if the run is still
going, or inspect `history.json` once it completes.
