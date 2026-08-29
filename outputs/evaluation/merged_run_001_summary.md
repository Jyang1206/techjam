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
| Clean | 0.5467 | 0.5154 | 0.6174 | 0.3519 |
| Best: color 1.2x | 0.5621 | 0.5408 | 0.5117 | 0.4067 |
| Worst: blur 2.0 | 0.4705 | 0.5000 | 1.0000 | 0.0000 |
| Resize 0.25x | 0.4748 | 0.5000 | 0.9999 | 0.0002 |

On clean CIFAKE, the saved threshold produces 6,174 false accusations among 10,000 real images and
misses 3,519 of 10,000 fakes. This checkpoint is therefore not suitable for deployment as-is.
Heavy blur and downscaling are the clearest robustness failures, consistent with the model relying
too strongly on high-frequency forensic traces.

## Interpretation and next experiment

The generator-disjoint validation score (0.7062) is meaningfully above chance, but it does not
transfer to CIFAKE (0.5467). CIFAKE's native 32x32 resolution is intentionally unlike the
full-resolution training sources, so this result exposes the expected resolution/domain gap rather
than leakage.

The next training experiment should keep generator-disjoint validation, stop around epochs 4-7,
and add stronger blur/downscale/JPEG augmentation or consistency training. The most relevant final
submission check remains WildFake's protected COCO/DALL-E-Advanced demonstration subset, which was
excluded from every training and materialization step.

Full artifacts: [`merged_run_001_cifake/`](merged_run_001_cifake/).
