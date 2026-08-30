# Handoff: run_007 robustness iteration

## Current recommendation

Use `checkpoints/merged/run_007/best.pt` for the WildFake-style unseen-generator task. It is a
self-contained 303,967,745-parameter OpenAI CLIP ViT-L/14 QuickGELU detector. The backbone is
frozen; only the 769-parameter linear authenticity head was adapted.

Run_007 reaches **0.9448 ROC-AUC** on a 4,375-image validation set whose fake generators (MAGE,
VQVAE, VQGAN, MAE) are entirely absent from training. This clears the requested 0.90 clean AUC
target and improves every tested corruption over run_005 without increasing inference cost. It is
not yet universal production quality: clean FPR is 16.2% at the saved balanced threshold, heavy
resizing falls to 0.793 AUC, and CIFAKE cross-dataset AUC is only 0.310.

## What changed

- Fixed WildFake destination collisions with a stable relative-path digest; all 25,000 images in
  `merged_v2` are unique.
- Added exact WildFake group filtering and downloaded/extracted DDPM. `merged_v3` contains 30,000
  images: 12,500 real and 17,500 fake.
- Replaced greedy whole-group selection with the subset closest to `--val-fraction`.
- Added `--fake-generator-disjoint-split`: fake generator families are held out wholesale, while
  authentic sources are image-stratified across both sides to avoid a faces-vs-objects label proxy.
- Added `--balance-groups`, frozen-feature caching, multiple augmented cache views, early stopping,
  checkpoint initialization, and protected epoch-0 evaluation.
- Corrected the CLIP architecture: preserve its learned projection and final normalization, use
  the 768-D projected embedding, QuickGELU, CLIP bicubic preprocessing, crop percentage 1.0, and a
  plain linear head. Earlier CLIP ablations incorrectly stripped the projection/norm.
- Imported the official MIT-licensed UniversalFakeDetect linear head, evaluated it at epoch 0, and
  adapted it at learning rate 1e-5 only when validation AUC improved.
- Evaluation now supports `--transforms clean` and optional complete clean-score export for
  operating-threshold analysis.
- Added a label-symmetric `--robustness-profile low_resolution` that exposes both classes to
  32/56/112-pixel resize and optional JPEG artifacts. Run_007 adapts the existing single-pass head
  on two cached views; it improves all 16 corruptions with no serving-latency increase.
- Added an optional zero-initialized shared-backbone low-resolution residual expert. Run_006 proved
  the safety property and improved clean/resize AUC slightly, but doubled encoder work and was not
  promoted. Robust TTA also regressed clean AUC from 0.9433 to 0.9388.
- Evaluation now supports deterministic label-balanced `--max-samples` screens. Subset outputs are
  development diagnostics only; final comparisons use every image.

## Data and split

Raw data stays under gitignored `data/`; regenerate it rather than committing it.

`data/merged_v3` groups:

| Group | Images |
|---|---:|
| SID_Set (real + fake) | 10,000 |
| WildFake ImageNet real | 5,000 |
| WildFake CelebA-HQ real | 2,500 |
| WildFake DDIM fake | 5,000 |
| WildFake DDPM fake | 5,000 |
| WildFake MAGE fake | 1,445 |
| WildFake VQVAE fake | 732 |
| WildFake VQGAN fake | 213 |
| WildFake MAE fake | 110 |

Seed-42 fake-generator-disjoint split:

- Train: 25,625 images; fake groups SID_Set, DDIM, DDPM.
- Validation: 4,375 images; fake groups MAGE, VQVAE, VQGAN, MAE.
- Authentic SID_Set/ImageNet/CelebA-HQ images are distinct but represented on both sides.
- Group-balanced replacement sampling gives equal total mass to real/fake and equal mass to each
  source/generator within a label.

Protected WildFake COCO real and advanced DALL-E rows remain excluded. Never pass
`--allow-protected-wildfake` for a submission model.

## Results

| Model | Honest evaluation | ROC-AUC | Balanced accuracy | Note |
|---|---|---:|---:|---|
| merged run_001 | old whole-source/generator validation | 0.7062 | 0.6819 | EfficientNet + frequency |
| merged run_004 | corrected fake-generator validation | 0.7334 | 0.6655 | incorrect pre-projection CLIP ablation |
| UniversalFakeDetect epoch 0 | run_005 held generators | 0.9421 | 0.8619 | official projected CLIP head |
| merged run_005 | held generators | 0.9433 | 0.8636 | projected CLIP baseline |
| merged run_006 | held generators | 0.9445 | 0.8650 | 32-pixel residual; 2x encoder work |
| **merged run_007** | held generators | **0.9448** | **0.8651** | selected single-pass checkpoint |
| merged run_005 | CIFAKE clean | 0.3024 | 0.4889 | severe 32x32/domain failure |
| merged run_007 | CIFAKE clean | 0.3098 | 0.4933 | improved, but still a failure |

Run_007 improves ROC-AUC in all 16 matched conditions and stays at or above 0.90 for 11/16, versus
10/16 for run_005. JPEG-50 crosses the target (0.9005). Worst cases remain resize 0.25 (0.7932),
blur 2 (0.8607), and noise 0.1 (0.8891). Full metrics are under
`outputs/evaluation/merged_run_007_held_generators/`.

The stored threshold (0.0323) maximizes balanced accuracy, producing 16.2% FPR and 89.2% recall in
direct evaluation. On this validation set, threshold 0.6833 caps FPR at 0.96% but reduces recall to
53.0%; threshold 0.1907 gives 4.96% FPR and 73.4% recall. Do not call the current threshold
production-safe without a product decision about that trade-off and a separate calibration set.

## Reproduction

```bash
# Import the official 4 KB UniversalFakeDetect head into a self-contained checkpoint.
python -m traceguard.import_universal <fc_weights.pth> \
  checkpoints/universal_fake_detect/official_vitl14/best.pt

# Safe low-rate adaptation; epoch 0 is retained unless fine-tuning improves validation AUC.
traceguard-train data/merged_v3 \
  --fake-generator-disjoint-split --balance-groups \
  --init-checkpoint checkpoints/universal_fake_detect/official_vitl14/best.pt \
  --evaluate-initial --freeze-backbone --cache-frozen-features \
  --feature-cache-views 2 --head-batch-size 2048 \
  --output-dir checkpoints/merged/run_005 \
  --epochs 20 --early-stopping-patience 5 --batch-size 128 --workers 4 \
  --lr 1e-5 --weight-decay 1e-2 --seed 42 --device cuda

# Single-pass low-resolution/JPEG robustness adaptation from run_005.
traceguard-train data/merged_v3 \
  --fake-generator-disjoint-split --balance-groups \
  --init-checkpoint checkpoints/merged/run_005/best.pt --evaluate-initial \
  --robustness-profile low_resolution --freeze-backbone --cache-frozen-features \
  --feature-cache-views 2 --head-batch-size 2048 \
  --output-dir checkpoints/merged/run_007 \
  --epochs 15 --early-stopping-patience 5 --batch-size 128 --workers 4 \
  --lr 3e-5 --weight-decay 1e-2 --seed 42 --device cuda
```

## Next work

1. Evaluate once on the still-protected WildFake COCO/DALL-E-Advanced demonstration subset; do not
   use it for training, checkpoint choice, or threshold selection.
2. Build an independent low-resolution test/calibration set. CIFAKE has now been repeatedly
   inspected and should be treated as a development stress test, not a final blind benchmark.
3. Add genuinely independent low-resolution real/fake generator data. Synthetic multi-scale
   augmentation is now exhausted and produced only small resize gains; generic extra epochs are
   unlikely to help.
4. Calibrate a conservative threshold on a deployment-matched calibration set and report recall at
   fixed FPR (1% and 5%), not AUC alone.
5. Distill the 304M detector after accuracy is stable; the current 1.13 GB checkpoint and ViT-L
   latency are expensive for production serving.

Detailed reasoning and ablations: `outputs/evaluation/run_005_architecture_and_training.md` and
`outputs/evaluation/run_007_robustness_iteration.md`.
