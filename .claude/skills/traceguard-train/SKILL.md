---
name: traceguard-train
description: Train the TraceGuard AIGC image detector (`traceguard-train` CLI) on one of its supported data sources — a local real/fake folder, SID_Set via Hugging Face, CIFAKE via Kaggle, or WildFake via ModelScope. Use this whenever the user wants to train, retrain, fine-tune, or kick off a training run for TraceGuard in this repo, asks which dataset flag to use, or wants to combine multiple datasets into one training run. Trigger on mentions of training on CIFAKE, SID_Set, or WildFake, or phrases like "train the model", "start a training run", "how do I train on my own images".
---

# Train TraceGuard

By default, `traceguard-train` fine-tunes the full two-branch model (EfficientNet-B0 + frequency
statistics, ~4M params) end-to-end. It also supports a low-capacity frozen-CLIP probe for stronger
cross-generator generalization. Before anything else, make sure you're in the `techjam` repo root
and its virtualenv is active:

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

# Cross-generator CLIP probe: only LayerNorm + binary head are trained (2,305 params)
traceguard-train data/merged --generator-disjoint-split \
  --backbone vit_base_patch16_clip_224.openai \
  --freeze-backbone --no-frequency-branch --cache-frozen-features \
  --lr 1e-3 --weight-decay 1e-2 --early-stopping-patience 5 \
  --output-dir checkpoints/merged/run_002

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

## Training on multiple sources combined

There is still no `--multi-source` flag on `traceguard-train` itself — the mutual-exclusivity check
at the top of this skill still applies to every training invocation. But combining sources is now a
two-step workflow instead of a manual, ad-hoc one: use the **`traceguard-materialize` skill** first
to pull a balanced, bounded sample from SID_Set and/or WildFake into one shared local folder
(`data/merged/real/`, `data/merged/fake/`), then train on that folder exactly like any other local
dataset:

```bash
# Step 1 (see the traceguard-materialize skill for full detail on both sources)
traceguard-materialize --hf-dataset saberzl/SID_Set --hf-samples-per-class 5000 \
  --output-dir data/merged
traceguard-materialize --wildfake-manifest ... --wildfake-images-root ... \
  --wildfake-samples-per-class 5000 --output-dir data/merged

# Step 2 - plain local-folder training. Add --generator-disjoint-split (see below) once you
# have images from more than one generator/source per label.
traceguard-train data/merged --output-dir checkpoints/merged/run_001
```

## Generator-disjoint validation (`--generator-disjoint-split`)

By default, local-folder training's internal train/validation split is random and blind to which
generator or source an image came from (`stratified_split`) — meaning the same generator can (and
usually will) show up on both sides. That validation score then mostly answers "does the model
recognize more examples of a generator it already trained on," not "does it generalize to a
generator it's never seen" — the thing this whole project is actually trying to measure.

`--generator-disjoint-split` fixes this by holding out **whole generator/source groups** for
validation instead of random individual images — `group_disjoint_split` in `data.py`. It only works
when images were named by `traceguard-materialize`, since that's what encodes generator identity
into the filename (`wildfake__<generator>__<architecture>__...`, `hf__<dataset_slug>__...`) —
`infer_group()` parses it back out when re-discovering images from a plain folder. Plain images with
no such naming fall into a single "unknown" group.

The splitter searches for the whole-group subset whose image count is closest to the requested
`--val-fraction`; it does not greedily accumulate shuffled groups. This matters when group sizes are
uneven, because a greedy split can accidentally place most minority generators in validation and
leave training dominated by one generator family.

**This needs at least 2 distinct groups per label to mean anything** — it raises a clear error
otherwise rather than silently falling back to something else. A SID_Set-only merge has exactly one
group per label (the whole HF dataset has no per-image generator column), so this flag only becomes
usable once you've added a second source with real generator/architecture variety — WildFake, whose
manifest tracks this per image. Once you have, say, 5 WildFake generator families plus SID_Set
mixed in, validation will hold out some of those generators entirely, giving a real read on
cross-generator generalization *during training itself* — not just from the separate CIFAKE
evaluation afterward.

Two things worth knowing about what ends up in that merged folder:

- **SID_Set's "fake" label conflates two different things**: fully-synthetic images (label 1) and
  AI-tampered real photos (label 2), both collapsed to `1`. WildFake's fakes are purely
  fully-synthetic. Mixing them means your "fake" class becomes slightly less homogeneous.
- **CIFAKE is deliberately excluded from this merge, not just unsupported.** It's natively 32×32
  pixels, while SID_Set/WildFake are full resolution — mixing it in risks the model learning
  "blurry/upscaled → CIFAKE's fakes" as a shortcut rather than a genuine forensic signal. It's more
  valuable held out entirely as a cross-generator generalization *test set* instead (pair with the
  `traceguard-evaluate` skill) — since a model trained only on SID_Set/WildFake would never have
  seen it, evaluating against it afterward is a genuine "does this generalize" check, not just
  another training input.

## Frozen CLIP probe for cross-generator generalization

The original EfficientNet/frequency model can memorize generator-specific high-frequency traces.
Use `--backbone vit_base_patch16_clip_224.openai --freeze-backbone --no-frequency-branch` to train a
small classifier over a pretrained CLIP visual representation instead. `--normalization auto`
(the default) detects CLIP backbones and applies CLIP's native mean/std during both training and
checkpoint inference; this choice is stored in `model_config`, so older ImageNet-normalized
checkpoints remain compatible.

`--freeze-backbone` removes the visual encoder from optimization, and `--no-frequency-branch`
prevents the head from falling back to brittle spectrum shortcuts. For this probe, `--lr 1e-3
--weight-decay 1e-2 --dropout 0.25` is a reasonable starting point. Use
`--early-stopping-patience 5` on generator-disjoint validation rather than assuming more epochs are
better. These settings are an ablation, not a guarantee: only the untouched evaluation result can
establish whether generalization improved.

For large frozen encoders, add `--cache-frozen-features`. It performs the expensive backbone pass
once, stores its output in RAM, and then optimizes the head with `--head-batch-size` (default 1024).
`--feature-cache-views N` caches N independently augmented versions of the training set; three
views are a useful robustness/compute compromise. Caching is intentionally rejected unless both
`--freeze-backbone` and `--no-frequency-branch` are set, because otherwise cached tensors would
silently prevent trainable backbone/frequency parameters from receiving gradients.

For a single-pass detector that must tolerate downsampling, add `--robustness-profile
low_resolution`. It applies the same label-symmetric 32/56/112-pixel resize and optional JPEG
pipeline to authentic and fake training images; use multiple cached views so one random severity
does not define an image permanently. `standard` remains the default and `none` is available for
controlled ablations.

`--low-resolution-size 32` adds a zero-initialized residual expert that reuses the same backbone on
a 32-pixel downsampled view. With `--init-checkpoint ... --freeze-base-classifier`, epoch 0 exactly
reproduces the initializer and only the new residual vector trains. This doubles encoder inference,
so compare it against augmentation-only training before promoting it.

To adapt an existing detector, pass `--init-checkpoint <best.pt> --evaluate-initial`. The complete
architecture and weights come from that checkpoint; epoch 0 is evaluated and saved before any
optimizer step. Fine-tuning therefore replaces `best.pt` only if validation ROC-AUC genuinely
improves over the initializer. This is the safe path for adapting the official UniversalFakeDetect
head without accidentally destroying its broader pretrained generalization.

When authentic source domains are visibly different (for example CelebA-HQ faces versus ImageNet
objects), prefer `--fake-generator-disjoint-split` over `--generator-disjoint-split`. It keeps fake
generator families completely disjoint but randomly stratifies authentic images across train and
validation. This avoids making real-source semantics a proxy for the label while retaining the
important unseen-generator test. Add `--balance-groups` when generator sizes are uneven; sampling
then assigns equal total mass to real/fake and equal mass to every source/generator within its
label, preventing a 5,000-image generator from drowning out a 200-image generator.

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
- `<output-dir>/history.json` — every completed epoch's train/validation loss plus accuracy,
  balanced accuracy, precision, recall, F1, ROC-AUC, Brier score, FPR, FNR. Rewritten after each
  epoch, so completed-epoch progress survives an interrupted longer run. Use the `traceguard-status`
  skill to inspect it afterward.

Training prints one summary line per epoch to the terminal it was launched from; that live output
is not captured anywhere else, so check the launching terminal for progress if the run is still
going, or inspect `history.json` once it completes.

## Sharing a checkpoint with the team

`checkpoints/` is version-controlled and `.pt` files are tracked through **Git LFS** (not plain
git) — this keeps repo history lean even as more runs get added. If a run is worth sharing, commit
it like any other file:

```bash
git add checkpoints/<source>/run_XXX/best.pt checkpoints/<source>/run_XXX/history.json
git commit -m "add <source> run_XXX checkpoint (val AUC X.XXXX)"
```

**Anyone pulling the repo needs Git LFS installed once** (`brew install git-lfs && git lfs install`,
or the equivalent for their OS) before cloning/pulling — without it they'll only get small pointer
files instead of the actual model weights. Not every experimental run needs to be pushed — use
judgment about which checkpoints are actually worth the team having (e.g. the best one per source,
not every intermediate epoch-count experiment).
