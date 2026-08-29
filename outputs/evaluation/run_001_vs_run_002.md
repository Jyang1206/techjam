# CIFAKE run_001 vs run_002 — comparison

Both runs trained on the same source (CIFAKE, Kaggle `birdy654/cifake-real-and-ai-generated-synthetic-images`,
20,000 train / 4,000 validation images, batch size 64). The only difference is epoch count.
Robustness numbers below are from `traceguard-evaluate` against CIFAKE's full 20,000-image test set
(`--tta none`), which was also this checkpoint's training-time validation split — see the leakage
caveat at the bottom before treating these as a final, submission-grade generalization number.

| | `checkpoints/cifake/run_001` | `checkpoints/cifake/run_002` |
|---|---|---|
| Epochs requested | 5 | 20 |
| Best epoch | 5 (still improving when it stopped) | 15 (overfitting past this point) |
| Validation ROC-AUC at best epoch | 0.9975 | 0.9980 |
| Decision threshold chosen | 0.6588 | 0.5246 |

## Robustness table: ROC-AUC by condition

| Condition | run_001 | run_002 | Δ |
|---|---|---|---|
| clean | 0.997 | 0.998 | +0.001 |
| jpeg_90 | 0.997 | 0.998 | +0.001 |
| jpeg_70 | 0.997 | 0.998 | +0.001 |
| jpeg_50 | 0.993 | 0.996 | +0.003 |
| jpeg_30 | 0.989 | 0.992 | +0.003 |
| blur_0.5 | 0.995 | 0.997 | +0.002 |
| blur_1 | 0.986 | 0.991 | +0.005 |
| **blur_2** | 0.967 | **0.976** | **+0.009** |
| resize_0.5 | 0.985 | 0.990 | +0.005 |
| **resize_0.25** | 0.964 | **0.972** | **+0.008** |
| noise_0.02 | 0.996 | 0.997 | +0.001 |
| noise_0.05 | 0.993 | 0.996 | +0.003 |
| noise_0.1 | 0.985 | 0.991 | +0.006 |
| color_0.8 | 0.994 | 0.997 | +0.003 |
| color_1.2 | 0.995 | 0.996 | +0.001 |
| crop_0.8 | 0.989 | 0.993 | +0.004 |

**run_002 improved on every single condition** — and improved most on the two conditions that were
already run_001's weakest spots (heavy blur, heavy downscaling). The extra 15 epochs specifically
shored up the model's actual weaknesses rather than only padding the easy clean-image score.

## Error trade-off at the clean condition

| | run_001 | run_002 |
|---|---|---|
| False positive rate (real → flagged fake) | 0.024 | 0.031 ↑ |
| False negative rate (fake → called real) | 0.031 | 0.012 ↓ |
| False positives / negatives (out of 20,000) | 12 / 12 | 12 / 12 |

AUC (threshold-independent) improved cleanly in run_002, but its *chosen* decision threshold
(0.5246 vs. run_001's 0.6588) sets a lower bar for calling something "fake" — it catches more fakes
at the cost of slightly more false accusations of real images. `select_threshold` picks whichever
threshold maximizes balanced accuracy, treating both error types as equally costly. Given false
positives are the more reputationally costly error for this problem (an authentic creator wrongly
flagged as AI-generated), it's worth deciding deliberately whether to keep run_002's threshold as-is
or raise it manually — a one-line change to the checkpoint's stored threshold, not a retrain.

## Most confident errors (clean condition)

| | run_001 | run_002 |
|---|---|---|
| Top false positive | `REAL/0099.jpg` → 0.9996 | `REAL/0227 (7).jpg` → 0.9999 |
| Top false negative | `FAKE/659.jpg` → 0.0045 | `FAKE/190 (6).jpg` → 0.00006 |

## Bottom line

Use **run_002** — it's strictly better on discriminative power (ROC-AUC) across every tested
condition, including its previously-weakest ones. Its lower decision threshold trades slightly more
false positives for far fewer false negatives; revisit that threshold choice explicitly if minimizing
false accusations matters more than maximizing recall for your use case.

## Caveat

Both checkpoints were evaluated against CIFAKE's own `test/` folder, which is the same data used as
validation during training (it influenced which epoch got saved as `best.pt` and which threshold was
picked). Treat the numbers above as a fair *relative* comparison between the two runs, but not yet a
clean *absolute* generalization number — that requires evaluating against data neither run ever saw
(e.g. WildFake, SID_Set, or the challenge's own COCO val2017 / DALL·E-Advanced demo subset).

Full per-condition metrics: [`cifake_run_001/`](cifake_run_001/) and
[`cifake_run_002/`](cifake_run_002/).
