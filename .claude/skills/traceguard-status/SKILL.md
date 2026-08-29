---
name: traceguard-status
description: Inspect a TraceGuard checkpoint's training history and validation metrics — precision, recall, F1, ROC-AUC, balanced accuracy — and diagnose overfitting or underfitting from the train/validation loss curves. Use whenever the user asks how their training run went, wants precision/recall/F1 numbers, asks about overfitting or underfitting, wants to check a checkpoint's stats, or asks whether more epochs would help.
---

# Check a TraceGuard training run's status

Every training run writes `<output-dir>/history.json` (one entry per epoch) and embeds the best
epoch's metrics inside `<output-dir>/best.pt` itself. Use the bundled script to read both without
writing throwaway inspection code each time:

Run from the `techjam` repo root with its virtualenv active:

```bash
source .venv/bin/activate
python .claude/skills/traceguard-status/scripts/inspect_checkpoint.py checkpoints/<run-name>
```

This prints a per-epoch table (train/val loss, precision, recall, F1, ROC-AUC, balanced accuracy)
and an automatic overfit/underfit read based on the loss curve shape. Pass `--json` to get the raw
numbers instead of the formatted table if the user wants to chart them elsewhere.

## How to interpret what comes back

**Overfitting vs. underfitting** — read `train_loss` against `validation_loss` across epochs:

| Pattern | Diagnosis |
|---|---|
| Both drop together, then plateau together | Healthy fit |
| `train_loss` keeps dropping, `validation_loss` flattens or turns upward | Overfitting — the model is starting to memorize training data rather than generalize |
| Both stay high, validation ROC-AUC stays near 0.5 (chance level) | Underfitting — not enough epochs, learning rate too low, or the signal genuinely isn't being picked up |

A run of only ~5 epochs (the CLI's small-run defaults) often isn't long enough to actually see the
overfitting inflection point — both curves may still be improving when it stops. If the diagnosis
looks inconclusive, that's worth saying explicitly rather than guessing; suggest rerunning with more
epochs (e.g. `--epochs 20`) to see where validation loss actually bottoms out.

**Precision / recall / F1** — don't just report F1 as a single summary number; read precision and
recall apart, because for this task the two error directions have different real-world costs:

- **Precision** = of everything flagged fake, how much really was. Low precision means real
  creators get wrongly accused of using AI — the more costly failure mode per the project's own
  stated limitations.
- **Recall** = of everything actually fake, how much got caught. Low recall means fakes slipping
  through.
- If precision looks notably worse than recall (or vice versa), say so plainly rather than folding
  it into one F1 number — this maps directly onto the "false positives vs false negatives" trade-off
  discussion the hackathon's error-analysis deliverable expects.

**Important caveat to always mention**: these numbers come from whatever "validation" split the
training run used. If the checkpoint was trained via the Kaggle/CIFAKE path, that's CIFAKE's own
`test/` folder — already used to pick the best epoch and threshold, not a clean measure of
generalization to truly unseen data. For a trustworthy read of how the model performs on data it has
never seen, point to the `traceguard-evaluate` skill against a separate, untouched dataset instead of
treating these training-time numbers as the final word.
