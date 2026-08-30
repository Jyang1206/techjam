---
name: traceguard-materialize
description: Build a merged local training folder for TraceGuard by pulling a balanced, bounded sample of images from SID_Set (Hugging Face) and/or WildFake (ModelScope) onto disk via `traceguard-materialize`, so they can be combined and trained on together with plain local-folder mode. Use whenever the user wants to combine multiple datasets into one training run, mix SID_Set and WildFake, or asks how to merge data sources for TraceGuard since the training CLI itself only accepts one source at a time.
---

# Materialize a merged multi-source dataset

`traceguard-train` enforces exactly one data source per run (local folder, SID_Set, CIFAKE, or
WildFake — see the `traceguard-train` skill). The only way to train on a genuine mix without new
training code is to first write a bounded, balanced sample from each remote source onto disk as
plain `real/`/`fake/` files, then point plain local-folder training at the merged result.
`traceguard-materialize` does that writing step; it does not train anything itself.

Run from the `techjam` repo root with its virtualenv active:

```bash
source .venv/bin/activate

# SID_Set: streams from Hugging Face, writes JPEGs to data/merged/{real,fake}/
traceguard-materialize --hf-dataset saberzl/SID_Set \
  --hf-samples-per-class 5000 --output-dir data/merged

# WildFake: needs the manifest + chosen image archives downloaded first (see traceguard-train
# skill for the `modelscope download` step) — this only copies from what's already local
traceguard-materialize --wildfake-manifest data/WildFake/split_train_test/csv_file/total_split/train_metadata.csv \
  --wildfake-images-root data/WildFake/Images \
  --wildfake-samples-per-class 5000 --output-dir data/merged
```

When an Images root contains several downloaded generators but an experiment needs one exact new
family, pass `--wildfake-include-groups` with exact manifest-derived keys. Include at least one real
and one fake group because materialization is balanced, for example:

```bash
traceguard-materialize --wildfake-manifest ... --wildfake-images-root data/WildFake/Images \
  --wildfake-include-groups wildfake__Real__imagenet wildfake__Diffusion_based__DDPM \
  --wildfake-samples-per-class 5000 --output-dir data/merged_v3
```

Run it once per source, pointing every call at the **same `--output-dir`** — each call only adds
its own images, it never touches or clears what a previous call wrote. Once both have run, train
on the combined folder like any local dataset:

```bash
traceguard-train data/merged --output-dir checkpoints/merged/run_001
```

## Recommended proportion (a starting point, not a hard rule)

~20,000 total images, split evenly: 5,000 real + 5,000 fake from SID_Set, 5,000 real + 5,000 fake
from WildFake. This matches the scale already validated on CIFAKE (`run_001`/`run_002`), so results
stay comparable. There's a real argument for weighting WildFake higher instead — it's explicitly
built around in-the-wild, redistributed images, closer to this challenge's actual theme than
SID_Set — but 50/50 is the simpler, more defensible default absent an actual experiment showing
otherwise. If time allows, this is empirically testable later with the same `traceguard-evaluate`
pipeline rather than something to guess hard about upfront.

## Why CIFAKE is deliberately left out of the merge

Don't materialize or fold CIFAKE into this mix. It's natively 32×32 pixels, while SID_Set and
WildFake are full resolution — blending them risks the model learning "blurry/upscaled → CIFAKE's
fakes" as a shortcut instead of a genuine forensic signal. CIFAKE is more valuable held out
entirely, as a cross-generator generalization test: since a WildFake/SID_Set-trained model would
never have seen it, evaluating against it afterward (`traceguard-evaluate` against CIFAKE's test
folder) becomes a genuine "does this generalize to an unseen generator" check, not just another
training input.

## Flags worth knowing

- `--hf-positive-labels` (default `1 2`) — SID_Set's label scheme: `0`=authentic, `1`=fully
  synthetic, `2`=tampered, with `1`/`2` both mapped to the `fake` folder here, matching how
  `traceguard-train`'s own `--hf-dataset` path treats SID_Set.
- `--allow-protected-wildfake` — **never pass this for anything meant for challenge submission.**
  Same protection as `traceguard-train`'s WildFake path: excludes real-COCO and advanced-DALL·E
  rows by default, since that's the challenge's held-out demonstration set.
- Images are re-saved as JPEG quality 95 regardless of source format — a deliberate, disk-space-
  conscious normalization, not a bug.

## What this does NOT do

It doesn't dedupe across sources, doesn't check for near-duplicate images between the two, and
doesn't guarantee generator-family separation for a later train/validation split — the same
leakage caveat that applies to any local-folder run applies here too (see `traceguard-train`'s
notes on this). It also doesn't download WildFake's raw archives for you — that manual step (via
the `modelscope` CLI) has to happen first, same as plain WildFake training.
