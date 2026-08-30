# Final model comparison

Scores are separated by evaluation protocol; numbers from different protocols are not treated as
direct substitutes.

## Held fake generators

| Model | Architecture | ROC-AUC | Balanced accuracy | FPR | FNR |
|---|---|---:|---:|---:|---:|
| merged run_001 | EfficientNet-B0 + frequency | 0.7062 | 0.6819 | 0.0876 | 0.5486 |
| merged run_004 | ViT-L/14 pre-projection probe | 0.7334 | 0.6655 | — | — |
| UniversalFakeDetect initializer | projected CLIP ViT-L/14 | 0.9421 | 0.8619 | — | — |
| **merged run_005** | projected CLIP ViT-L/14 | **0.9433** | **0.8639** | **0.1499** | **0.1224** |

Run_001 used the older, stricter whole-source/group fold, while run_004/run_005 use the corrected
fake-generator-disjoint protocol. Run_005 validation fakes are MAGE/VQVAE/VQGAN/MAE and never
appear in training. The gain is therefore meaningful, but run_001 and run_005 are not a perfect
same-fold ablation; the architectural report explains the split correction.

## CIFAKE clean, no TTA

| Model | Relationship to CIFAKE | ROC-AUC | Balanced accuracy | FPR | FNR |
|---|---|---:|---:|---:|---:|
| CIFAKE run_001 | test used for training-time validation | 0.9967 | 0.9727 | 0.0236 | 0.0310 |
| CIFAKE run_002 | test used for training-time validation | 0.9979 | 0.9783 | 0.0312 | 0.0122 |
| merged run_001 | CIFAKE unseen | 0.5833 | 0.5415 | 0.2103 | 0.7067 |
| official UniversalFakeDetect | CIFAKE unseen | 0.3003 | 0.3606 | 0.7514 | 0.5274 |
| merged run_005 | CIFAKE unseen | 0.3024 | 0.4889 | 0.9887 | 0.0335 |

The CIFAKE-trained models are in-domain references with validation leakage, not honest production
generalization. Run_005's failure shows that its >0.94 WildFake held-generator result does not
transfer to CIFAKE's native 32x32 domain.

## Decision

Use run_005 for the declared full-resolution unseen-generator benchmark. Do not deploy one model
universally across arbitrary resolutions yet. A low-resolution expert/new independent benchmark,
fixed-FPR calibration, and protected WildFake final evaluation are still required.
