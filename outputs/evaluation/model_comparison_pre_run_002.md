# Model comparison before merged run_002

All numbers in this table use the full 20,000-image CIFAKE test folder with `--tta none`.

| Model | Training data | Clean ROC-AUC | Balanced accuracy | FPR | FNR |
|---|---|---:|---:|---:|---:|
| CIFAKE run_001 | CIFAKE | 0.9967 | 0.9727 | 0.0236 | 0.0310 |
| CIFAKE run_002 | CIFAKE | 0.9979 | 0.9783 | 0.0312 | 0.0122 |
| Merged run_001 | SID_Set + WildFake | 0.5833 | 0.5415 | 0.2103 | 0.7067 |

The CIFAKE checkpoints are strong in-domain references, but the same CIFAKE test folder was used
as their training-time validation set. It influenced checkpoint and threshold selection, so their
near-0.998 values are optimistic and cannot be called production generalization. Merged run_001
never saw CIFAKE, making its much lower score the honest cross-dataset baseline.

Merged run_001 also obtained 0.7062 ROC-AUC on generator-disjoint WildFake/SID_Set validation. Its
gap to 0.5833 on CIFAKE, plus severe degradation under noise/blur/downscaling, shows that the current
EfficientNet/frequency representation learns source-specific forensic traces. The next ablation
therefore replaces the fully fine-tuned ImageNet CNN with a frozen CLIP visual representation,
uses CLIP's native normalization, fixes materialization collisions, and reduces trainable capacity.
This follows published evidence that CLIP feature probes improve cross-generator generalization and
robustness; the result must still be verified on this repository's held-out data.
