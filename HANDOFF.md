# Handoff: merged multi-source training (needs more disk/bandwidth)

Status as of the latest push (`e83fb28`). Written for a teammate picking this up on a machine with
more disk space than the one this was developed on (which is stuck at 15GB free — see below).

## TL;DR — what's blocked and what to do about it

We want to train TraceGuard on a merged **SID_Set + WildFake** dataset (CIFAKE is deliberately held
out as a generalization test, not training data — see "Why CIFAKE is excluded" below). SID_Set's
half is done. **WildFake is blocked purely on disk space** on this machine — everything needed to
unblock it is below. If you have 30-50GB+ free, you can just proceed.

## What's already done

1. **Two CIFAKE-only baselines trained and evaluated**, both committed (weights via Git LFS):
   - `checkpoints/cifake/run_001/` — 5 epochs, val ROC-AUC 0.9975
   - `checkpoints/cifake/run_002/` — 20 epochs, best at epoch 15, val ROC-AUC 0.998, beats run_001
     on every robustness condition (see `outputs/evaluation/run_001_vs_run_002.md`)
2. **`traceguard-materialize`** — a new CLI tool (not part of the original repo) that pulls a
   balanced, bounded sample from a remote dataset and writes it to a local `real/`/`fake/` folder,
   so multiple sources can be combined and trained on via plain local-folder mode. See the
   `traceguard-materialize` skill in `.claude/skills/` for full usage.
3. **SID_Set half of the merge is materialized**: 5,000 real + 5,000 fake full-resolution images.
   **This is NOT in the repo** (`data/` is gitignored, correctly — it's raw dataset content, not a
   deliverable). You'll need to regenerate it yourself:
   ```bash
   source .venv/bin/activate
   traceguard-materialize --hf-dataset saberzl/SID_Set --hf-samples-per-class 5000 \
     --output-dir data/merged
   ```
   Get a free Hugging Face token first (unauthenticated streaming is ~13x slower) — see the
   "Stream SID_Set" section of `README.md` for the exact steps (`hf auth login --token ...`).
4. **Git LFS is set up** for checkpoint sharing — run `git lfs install` once before pulling if you
   haven't already, or you'll only get pointer files instead of real weights.

## Why WildFake is blocked here (read before attempting)

WildFake's images are **not individually downloadable** — they're packed into large ZIP archives,
and the smallest useful ones are still substantial:

| Category | Size |
|---|---|
| Real: celebahq | 351 MB |
| Real: afhq | 452 MB |
| Real: ffhq | 819 MB |
| Real: church | 1.16 GB |
| Real: imagenet | 1.38 GB |
| Real: laion5b | 24.8 GB |
| Real: coco | 2.35 GB — **never download, this is the challenge's protected demo set** |
| Fake (Diffusion): DDIM | 6.05 GB — smallest fake option |
| Fake (Diffusion): DDPM | 8.14 GB |
| Fake (Diffusion): ADM / Imagen / VQDM / DALLE | 17–26 GB each |
| Fake: GAN_based (whole category, one file) | 47.3 GB |
| Fake: Other_based (whole category, one file) | 13.3 GB |
| Fake: SD / Midjourney | split into ~50GB chunks each |

A minimal real+fake pull (imagenet + DDIM) is already ~7.4GB, and ZIP extraction needs roughly 2x
that in transient disk space. This machine has 15GB free — too tight to safely attempt even the
minimal option. **If you have 30GB+ free, this is very doable.**

### How to actually pull WildFake (once you have the disk space)

```bash
pip install "modelscope>=1.34,<2"

# 1. Manifest only, first (small-ish, ~312MB) - lets you inspect what's available
modelscope download --dataset hy2628982280/WildFake \
  --local_dir data/WildFake \
  --include "split_train_test/csv_file/total_split/*"

# 2. Pick and download specific category archives. Recommended minimal starting point:
modelscope download hy2628982280/WildFake --repo-type dataset \
  "Images/Real/imagenet.zip" "Images/Diffusion_based/DDIM.zip" \
  --local-dir data/WildFake

# 3. Extract them under data/WildFake/Images/... matching the manifest's expected layout
#    (Real/imagenet/..., Diffusion_based/DDIM/...) - check the manifest's Image_path column
#    for the exact relative structure each archive should unzip to.

# 4. Materialize a balanced sample from what you've downloaded (protected rows excluded by default)
traceguard-materialize \
  --wildfake-manifest data/WildFake/split_train_test/csv_file/total_split/train_metadata.csv \
  --wildfake-images-root data/WildFake/Images \
  --wildfake-samples-per-class 5000 \
  --output-dir data/merged
```

**Never pass `--allow-protected-wildfake`** — that flag exists only to disable the exclusion of
real-COCO and advanced-DALL·E rows, which is the challenge's own held-out demonstration set. Using
it would be training on data you're supposed to be evaluated against.

If disk allows for more variety than just DDIM, adding one GAN-family archive would help
cross-generator diversity a lot (this project's core theme) — `Other_based.zip` (13.3GB) is the
cheapest way to get generator diversity beyond pure diffusion models, since `GAN_based.zip` (47GB)
is the only other option and is a single non-subdividable file.

**This is also exactly why you should pull from more than one WildFake generator family if at all
possible** (not just DDIM alone): once you have 2+ distinct generators/sources, you can pass
`--generator-disjoint-split` to `traceguard-train` (added after this handoff was first written -
see the `traceguard-train` skill for full detail). Instead of the default random per-image
train/validation split, it holds out **entire generator groups** for validation, so the validation
score actually measures "does this generalize to a generator it never trained on," not just "does
it recognize more examples of a generator it already trained on." Neither WildFake's own official
train/test manifest split nor our default local-folder split does this (verified directly: all 26
generator/architecture combinations in WildFake's manifest appear in both its train and test CSVs)
— this flag is the only thing in the whole pipeline that actually tests cross-generator
generalization *during training*, as opposed to the separate CIFAKE/demo-subset evaluation
afterward. It needs at least 2 distinct groups per label or it raises a clear error - a single
source (e.g. SID_Set alone, or one single WildFake generator) isn't enough to use it.

## Once both sources are materialized: train

```bash
traceguard-train data/merged --output-dir checkpoints/merged/run_001
```

Nothing new here — this is the same local-folder training path that already existed; materializing
just prepares the folder it reads from. Standard naming convention applies:
`checkpoints/<source>/run_XXX`, and the training code will refuse to overwrite an existing run's
results (pick a new run number, or pass `--overwrite` if that's really what you want).

## Why CIFAKE is excluded from this merge

CIFAKE is natively 32×32 pixels; SID_Set/WildFake are full resolution. Mixing them risks the model
learning "blurry/upscaled → CIFAKE's fakes" as a shortcut instead of a genuine forensic signal.
It's more valuable held out entirely as a **cross-generator generalization test**: since a model
trained only on SID_Set/WildFake never saw CIFAKE, evaluating against it afterward
(`traceguard-evaluate` against CIFAKE's test folder with the new checkpoint) is a genuine
"does this generalize to an unseen generator" check — pair this with the `traceguard-evaluate` skill.

## After training the merged model

1. Evaluate it against a genuinely held-out set — CIFAKE (never seen), and ideally the official
   WildFake COCO val2017 / DALL·E-Advanced demonstration subset for the real submission-grade number.
2. Compare against `run_001`/`run_002` using the same `traceguard-status` / `traceguard-evaluate`
   skills already documented in `.claude/skills/`.
3. Update `outputs/evaluation/` with the new comparison and commit it (small text files, not
   gitignored) — follow the pattern in `outputs/evaluation/run_001_vs_run_002.md`.
4. Remember: `checkpoints/` and `outputs/` are both tracked in this repo (checkpoints via Git LFS,
   outputs as plain text) — commit whatever's worth the team seeing, push so everyone can pull it.

## Reference: all `.claude/skills/` available for this project

`traceguard-train`, `traceguard-predict`, `traceguard-evaluate`, `traceguard-demo`,
`traceguard-status`, `traceguard-materialize` — each documents its own command's flags, gotchas,
and the reasoning behind key decisions (checkpoint-overwrite protection, TTA behavior, the
Kaggle-validation-set leakage caveat, etc.). Worth reading before diving in blind.
