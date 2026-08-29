# Merged SID_Set + WildFake run_001

`checkpoints/merged/run_001` was trained on a generator-disjoint split of the materialized
SID_Set/WildFake mix. CIFAKE was excluded from training and used only afterward as a genuinely
unseen cross-dataset test.

## Data and split

The final materialized folder contains 24,978 unique images: 12,500 real and 12,478 fake. The
22-image difference comes from duplicate filenames within the same Other-based architecture; the
materializer safely overwrote those duplicate targets.

| Split | Real | Fake | Total | Entire groups present |
|---|---:|---:|---:|---|
| Train | 10,000 | 10,110 | 20,110 | SID_Set, ImageNet, DDIM, MAE |
| Validation | 2,500 | 2,368 | 4,868 | CelebA-HQ, MAGE, VQGAN, VQVAE |

No generator/source group appears on both sides. All 24,978 materialized files passed a complete
Pillow decode/verify pass before training.

## Training result

The best checkpoint is epoch 4, selected by validation ROC-AUC:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.7062 |
| Balanced accuracy | 0.6819 |
| Precision | 0.8300 |
| Recall | 0.4514 |
| F1 | 0.5848 |
| Decision threshold | 0.0203 |

The run overfits aggressively across generator groups. Training loss falls from 0.3326 to 0.0141,
while validation loss bottoms at epoch 2 (1.4471) and ends at 3.1542. Epoch 4 has the best AUC;
later epochs mostly regress toward chance, so extending this exact run would not help.

## Untouched CIFAKE evaluation

CIFAKE's 20,000-image test folder was never used for this checkpoint's training, validation,
checkpoint selection, or threshold selection. These numbers therefore measure genuine
cross-dataset generalization, unlike the earlier CIFAKE-trained baselines whose CIFAKE test folder
also served as training-time validation.

| Condition | ROC-AUC | Balanced accuracy | FPR | FNR |
|---|---:|---:|---:|---:|
| Clean | 0.5833 | 0.5415 | 0.2103 | 0.7067 |
| Best: color 1.2x | 0.5985 | 0.5473 | 0.2330 | 0.6725 |
| Worst: noise 0.1 | 0.4295 | 0.5000 | 0.0018 | 0.9982 |
| Blur 2.0 | 0.4708 | 0.4999 | 0.9998 | 0.0005 |
| Resize 0.25x | 0.4770 | 0.4998 | 0.9997 | 0.0007 |

On clean CIFAKE, the saved threshold produces 2,103 false accusations among 10,000 real images and
misses 7,067 of 10,000 fakes. This checkpoint is therefore not suitable for deployment as-is.
Heavy blur, noise, and downscaling are the clearest robustness failures, consistent with the model
relying too strongly on fragile high-frequency forensic traces.

These corrected numbers use `--tta none`, matching both CIFAKE baselines. The earlier robust-TTA
evaluation remains available for ablation purposes, but its clean AUC was lower (0.5467), so it is
not mixed into the comparison.

## Interpretation and next experiment

The generator-disjoint validation score (0.7062) is meaningfully above chance, but it does not
transfer to CIFAKE (0.5833). CIFAKE's native 32x32 resolution is intentionally unlike the
full-resolution training sources, so this result exposes the expected resolution/domain gap rather
than leakage.

The next training experiment should keep generator-disjoint validation, stop around epochs 4-7,
and add stronger blur/downscale/JPEG augmentation or consistency training. The most relevant final
submission check remains WildFake's protected COCO/DALL-E-Advanced demonstration subset, which was
excluded from every training and materialization step.

Fair-comparison artifacts: [`merged_run_001_cifake_tta_none/`](merged_run_001_cifake_tta_none/).
Robust-TTA ablation: [`merged_run_001_cifake/`](merged_run_001_cifake/).
