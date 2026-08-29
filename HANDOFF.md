# Handoff: merged multi-source training

Status: merged training is unblocked and `run_001` is complete. The raw datasets remain under
`data/` (gitignored); reproducible model and evaluation artifacts are tracked in Git.

## Current checkpoint

Local commit `528d663` contains the first SID_Set + WildFake model, its full training history,
evaluation output, status-diagnosis fix, and path-sanitized error reports. The branch is one commit
ahead of `origin/master`; it has not been pushed because no remote push was requested.

`checkpoints/merged/run_001/best.pt` is an EfficientNet-B0 spatial/frequency model trained for 12
epochs with a generator-disjoint split. The best checkpoint is epoch 4:

| Metric | Value |
|---|---:|
| Validation ROC-AUC | 0.7062 |
| Validation balanced accuracy | 0.6819 |
| Validation precision | 0.8300 |
| Validation recall | 0.4514 |
| Decision threshold | 0.0203 |

Training loss fell from 0.3326 to 0.0141 while validation loss bottomed at epoch 2 and ended at
3.1542. This is strong cross-generator overfitting; extending the same configuration is not a
useful next step.

## Data now available locally

The machine now has sufficient disk space. These sources were downloaded and extracted:

- SID_Set: 5,000 real + 5,000 fake images.
- WildFake ImageNet real archive.
- WildFake CelebA-HQ real archive.
- WildFake DDIM fake archive.
- WildFake Other-based archive (MAE, MAGE, VQGAN, and VQVAE groups).

The current `data/merged` folder has 24,978 verified images: 12,500 real and 12,478 fake. The
generator-disjoint seed-42 split is:

| Split | Real | Fake | Total | Groups |
|---|---:|---:|---:|---|
| Train | 10,000 | 10,110 | 20,110 | SID_Set, ImageNet, DDIM, MAE |
| Validation | 2,500 | 2,368 | 4,868 | CelebA-HQ, MAGE, VQGAN, VQVAE |

No group crosses the split. All materialized files passed Pillow decode/verify. The 22 missing fake
images exposed a materializer bug: different WildFake subdirectories can contain the same basename
inside one architecture, and the old destination naming scheme overwrote them. Fix destination
names with a stable relative-path digest, test it, then rematerialize to a new folder before the
next definitive run.

## Evaluation and comparison

With the same `--tta none` setting used for every baseline, the first merged model reached ROC-AUC
0.5833 on clean CIFAKE and failed most severely under heavy blur, noise, and 0.25x downscaling.
CIFAKE was never used to train or select this checkpoint, so this is a real cross-dataset result.
The older CIFAKE checkpoints score near 0.998 on CIFAKE, but their test
folder was also used for checkpoint selection and threshold selection; those values are useful only
as an in-domain relative comparison, not as unbiased generalization estimates.

The original merged evaluation used robust TTA while the two baseline evaluations used
`--tta none`; the corrected evaluation is in `merged_run_001_cifake_tta_none/`. Robust TTA reduced
clean AUC from 0.5833 to 0.5467 for this model, so it should remain disabled unless a later ablation
shows a benefit.

Full run-001 analysis is in `outputs/evaluation/merged_run_001_summary.md`.

## Recommended next experiment

1. Fix collision-proof WildFake materialization and rebuild a clean merged dataset.
2. Keep generator-disjoint validation and the same seed so changes are comparable.
3. Reduce optimization pressure (lower learning rate, higher weight decay, and early stopping or a
   shorter run). The original run was already overfitting by epoch 3.
4. Preserve the existing stochastic degradation augmentation, but tune it toward the measured
   blur/downscale failure rather than simply adding more epochs.
5. Evaluate every candidate with identical `--tta none` settings on CIFAKE and report ROC-AUC plus
   FPR/FNR. Select by generator-disjoint validation AUC, not the held-out CIFAKE result.
6. For the final submission-grade comparison, evaluate all candidates on WildFake's protected
   COCO/DALL-E-Advanced subset. Never include that subset in training, threshold selection, or
   checkpoint selection.

Use a new output directory for every experiment (`checkpoints/merged/run_002`, etc.). Training
refuses to overwrite an existing run unless `--overwrite` is explicitly passed.

## Reproduction commands

```bash
traceguard-train data/merged \
  --generator-disjoint-split \
  --output-dir checkpoints/merged/run_001 \
  --epochs 12 --batch-size 32 --workers 4 --lr 3e-4 --weight-decay 1e-4

traceguard-evaluate <CIFAKE-test-folder> \
  --checkpoint checkpoints/merged/run_001/best.pt \
  --output-dir outputs/evaluation/merged_run_001_cifake_tta_none \
  --transforms all --batch-size 128 --tta none
```

Project workflow documentation lives under `.claude/skills/`: `traceguard-materialize`,
`traceguard-train`, `traceguard-evaluate`, `traceguard-status`, `traceguard-predict`, and
`traceguard-demo`.
