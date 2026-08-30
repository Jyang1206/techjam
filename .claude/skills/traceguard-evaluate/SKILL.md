---
name: traceguard-evaluate
description: Run TraceGuard's robustness benchmark (`traceguard-evaluate`) to measure clean vs. transformed (JPEG, blur, resize, noise, color jitter, center crop) accuracy and ROC-AUC on a labeled real/fake test folder, producing the robustness table and false-positive/false-negative error analysis the hackathon submission requires. Use whenever the user wants to evaluate a checkpoint, generate the robustness table for Devpost, check false positives or false negatives, measure how a model holds up under the challenge's required image transformations, or asks "how good is my model."
---

# Run the TraceGuard robustness benchmark

Run from the `techjam` repo root with its virtualenv active:

```bash
source .venv/bin/activate
traceguard-evaluate path/to/labeled_test_folder \
  --checkpoint checkpoints/<run-name>/best.pt \
  --output-dir outputs/evaluation
```

## The folder must be labeled

Unlike `traceguard-predict`, this command needs ground truth to compute accuracy/AUC — the target
directory must contain `real/` and `fake/` subfolders (matched case-insensitively), same layout as
local training data.

## What it actually does

It scores every image once per condition: clean, plus every combination in the challenge's required
transform suite —

- JPEG quality 90 / 70 / 50 / 30
- Gaussian blur σ 0.5 / 1.0 / 2.0
- Resize 0.5× / 0.25× (downscale then upscale back)
- Gaussian noise σ 0.02 / 0.05 / 0.10
- Color jitter (brightness/contrast/saturation) at 0.8× and 1.2×
- Center crop to 80%

Use `--transforms jpeg blur` (space-separated, choices are the transform names above) to run only a
subset instead of the full suite — useful for a quick check. Default is `all`.
Use `--transforms clean` for a one-condition screening run before spending time on the full suite,
especially with a large ViT encoder.

Use `--max-samples N --seed 42` for a deterministic label-balanced development screen. A subset is
appropriate for rejecting an obviously weak experiment, but rerun promoted checkpoints on the full
set before adding them to the comparison table.

Scoring is batched (`--batch-size`, default 64) — images are grouped into batches and scored with
one model forward pass per batch rather than one call per image, which matters a lot on GPU/MPS
where per-call overhead dominates at batch size 1. A progress line prints periodically within each
condition (`  jpeg_30       1280/20000`) so a long run stays observable instead of looking stuck.

## CRITICAL — avoid evaluating on data that already influenced the checkpoint

If the checkpoint was trained via the Kaggle/CIFAKE path, its "validation" set during training was
CIFAKE's own `test/` folder, and that data already influenced which epoch got saved as `best.pt` and
what decision threshold was chosen. Running `traceguard-evaluate` against that same folder again and
reporting the result as your final number is a form of leakage — the number will look better than
the model's true generalization. Before running this, check what data the checkpoint's training run
actually used (its `history.json`'s implied source, or the `training_source` field inside the `.pt`
file itself: `torch.load(path, map_location="cpu", weights_only=True)["training_source"]`) and point
this evaluation at a genuinely different, untouched dataset instead — e.g. a different source
entirely, a held-out slice you set aside before training, or (best, for a submission-grade number)
the challenge's own COCO val2017 / DALL·E-Advanced demonstration subset, which the code's WildFake
loader already protects from ever being used in training by default.

## What gets produced in `--output-dir`

- **`metrics.json` / `metrics.csv`** — one row per condition (clean + every transform×severity),
  with accuracy, balanced accuracy, ROC-AUC, false-positive rate, false-negative rate.
- **`robustness_table.md`** — the same data as a ready-to-paste Markdown table for the Devpost
  submission (this is exactly what deliverable #4, "Robustness Evaluation Summary," asks for).
- **`error_analysis.json`** — the top false positives and false negatives from the *clean* condition
  (ranked by confidence), for deliverable #5, "Error Analysis Note." Default 12 examples each side,
  change with `--error-examples N`.
- **`clean_predictions.csv`** (only with `--save-clean-predictions`) — every clean label/score,
  used to select deployment operating points such as a threshold capped at 1% false-positive rate.

## Reading the results

- **Primary metric is ROC-AUC**, not raw accuracy — it's threshold-free and robust to class
  imbalance, which is why the webinar/brief both call it out as the metric judges care about.
- Compare the **worst transformed condition's AUC to clean AUC** — a big drop tells you exactly
  which transformation the model is least robust to, and is exactly what the "trade-offs" narrative
  the brief asks for should discuss.
- **False positives are the costlier error direction** for this problem (an authentic creator wrongly
  flagged as AI) — when reading `error_analysis.json`, give the false-positive list real scrutiny,
  not just a passing mention, since that's what the brief's error-analysis deliverable is really
  probing for.
