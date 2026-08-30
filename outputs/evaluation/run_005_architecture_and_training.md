# Run_005 architecture and training report

## Outcome

Run_005 improves honest held-generator ROC-AUC from 0.7062 (merged run_001) to **0.9433**, an
absolute gain of **0.2371**. The 0.90 clean target is met on unseen WildFake generators. The model
is not universally production-ready: its saved operating threshold has 15.0% FPR, heavy resize AUC
is 0.790, and clean CIFAKE AUC is 0.302.

## Why the earlier model plateaued

Run_001 fully fine-tuned an EfficientNet-B0 plus explicit radial-frequency statistics. It learned
the training generators quickly: train loss fell to 0.0141, but held-group validation AUC peaked at
0.7062 and validation loss rose sharply. CIFAKE and heavy blur/resize failures showed reliance on
fragile, source-specific forensic traces.

The first CLIP ablations did not faithfully implement CLIP. `timm.create_model(..., num_classes=0,
global_pool="avg")` removed the learned CLIP projection and final normalization, producing
1,024-D pre-projection features instead of the native 768-D embedding. The repeated warning about
unexpected `norm.weight`/`norm.bias` was therefore actionable. Those runs are diagnostic only.

The original whole-group split also held out every real source group. In one fold, validation reals
were all CelebA-HQ faces while validation fakes were mostly ImageNet-style objects. A 300M CLIP
probe separated those semantics almost perfectly in the wrong direction (AUC 0.081), proving the
split had made content/source a proxy for the label.

## Final architecture

- OpenAI CLIP ViT-L/14 QuickGELU visual encoder.
- Native learned CLIP projection retained: 768-D embedding.
- CLIP mean/std, bicubic interpolation, and native crop percentage 1.0.
- Plain linear binary head initialized from the official UniversalFakeDetect checkpoint.
- No explicit frequency branch and no extra LayerNorm in the detector head.
- 303,967,745 total parameters; frozen encoder; 769 trainable head parameters.

This follows the official UniversalFakeDetect recipe, which trains only a linear layer over frozen
CLIP ViT-L/14 features, and later published work reporting strong cross-generator/perturbation
generalization from lightweight CLIP detectors. Sources: [UniversalFakeDetect official repository](https://github.com/WisconsinAIVision/UniversalFakeDetect),
[CVPR 2023 paper](https://arxiv.org/abs/2302.10174), and [CVPRW 2024 CLIP detector paper](https://openaccess.thecvf.com/content/CVPR2024W/WMF/papers/Cozzolino_Raising_the_Bar_of_AI-generated_Image_Detection_with_CLIP_CVPRW_2024_paper.pdf).

## Data and split corrections

- Stable path digests prevent WildFake basename collisions (22 lost samples recovered).
- Added 5,000 DDPM fakes, producing `merged_v3` with 30,000 images and materially broader fake
  training coverage.
- Fake generators are fully disjoint: train uses SID_Set/DDIM/DDPM; validation uses
  MAGE/VQVAE/VQGAN/MAE.
- Authentic images are image-disjoint but source-stratified across both sides. This avoids
  faces-vs-objects confounding while preserving the actual unseen-generator question.
- Weighted sampling assigns 50% mass to each label, then equal mass to every group within a label.

Protected COCO real and advanced DALL-E data were not downloaded into the merged set, trained on,
used for threshold selection, or used for checkpoint selection.

## Optimization method

The official head was evaluated and saved at epoch 0 before adaptation. The frozen encoder was run
only once for validation and twice for independently augmented training views; the cached 768-D
features then allowed cheap head optimization. AdamW used learning rate 1e-5 and weight decay 1e-2,
with up to 20 epochs and patience 5. This guard meant local tuning could not replace the official
initializer unless validation AUC improved.

| Stage | Held-generator ROC-AUC | Delta |
|---|---:|---:|
| run_001 EfficientNet/frequency | 0.7062 | baseline |
| run_004 incorrect pre-projection CLIP | 0.7334 | +0.0272 |
| official projected CLIP head, epoch 0 | 0.9421 | +0.2359 |
| run_005 adapted head, epoch 20 | **0.9433** | **+0.2371** |

The architectural correction and official universal head account for almost all improvement;
local head tuning contributes about +0.0012 AUC.

## Robustness

| Condition | ROC-AUC | Balanced accuracy |
|---|---:|---:|
| clean | 0.943 | 0.864 |
| jpeg 90 | 0.925 | 0.839 |
| jpeg 70 | 0.921 | 0.831 |
| jpeg 50 | 0.899 | 0.808 |
| jpeg 30 | 0.895 | 0.801 |
| blur 0.5 | 0.935 | 0.848 |
| blur 1.0 | 0.902 | 0.801 |
| blur 2.0 | 0.859 | 0.750 |
| resize 0.5 | 0.898 | 0.799 |
| resize 0.25 | 0.790 | 0.701 |
| noise 0.02 | 0.930 | 0.827 |
| noise 0.05 | 0.915 | 0.791 |
| noise 0.10 | 0.889 | 0.714 |
| color 0.8 | 0.933 | 0.851 |
| color 1.2 | 0.912 | 0.790 |
| crop 0.8 | 0.929 | 0.836 |

Ten of sixteen conditions remain above 0.90 AUC. Heavy downscaling is the main unresolved failure.

## Operating threshold and deployment assessment

The saved threshold 0.0257 maximizes validation balanced accuracy:

- FPR 14.99%, false-negative rate 12.24%, recall 87.76%.
- Threshold 0.1464: FPR 4.96%, recall 72.40%.
- Threshold 0.5780: FPR 0.96%, recall 53.16%.

AUC above 0.90 means the ranking is strong; it does not make the default decision policy safe.
Production deployment needs a separate, deployment-matched calibration set and an explicit false
accusation budget.

Run_005 also scores only 0.3024 AUC on CIFAKE clean. CIFAKE is natively 32x32 and now functions as
an inspected development stress test. The failure demonstrates that the detector is strong for the
declared unseen-generator/full-resolution task, not universal across resolutions and datasets.

## Recommended next methods

1. Target resize/downsampling directly with multi-scale feature views or a low-resolution expert;
   validate on a new untouched low-resolution set.
2. Evaluate exactly once on the protected WildFake COCO/DALL-E-Advanced demonstration subset.
3. Calibrate at fixed FPR and report recall at 1%/5% FPR.
4. Distill or quantize the 304M model only after the robustness target is stable.

Full artifacts: [`merged_run_005_held_generators/`](merged_run_005_held_generators/),
[`merged_run_005_held_generators_clean_scores/`](merged_run_005_held_generators_clean_scores/), and
[`merged_run_005_cifake_clean/`](merged_run_005_cifake_clean/).
