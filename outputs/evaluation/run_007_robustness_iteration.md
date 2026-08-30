# Run_007 robustness iteration

## Outcome

Run_007 is the promoted full-resolution unseen-generator model. It improves held-generator clean
ROC-AUC from 0.94329 to 0.94482 and improves every one of the 16 matched robustness conditions over
run_005. Conditions at or above 0.90 AUC increase from 10/16 to 11/16. The architecture and serving
cost are unchanged: OpenAI CLIP ViT-L/14 QuickGELU, 303,967,745 total parameters, a frozen visual
encoder, and 769 trainable linear-head parameters during adaptation.

This is not a universal production result. The saved balanced threshold still has a 16.2% clean
false-positive rate, resize-0.25 reaches only 0.7932 AUC, and the full 20,000-image CIFAKE stress
test reaches only 0.3098 AUC.

## Method

Run_007 starts from `checkpoints/merged/run_005/best.pt` and evaluates that initializer at epoch 0,
so training cannot silently replace it unless clean generator-disjoint AUC improves. The fake
generator split is identical to run_005: SID_Set/DDIM/DDPM train; MAGE/VQVAE/VQGAN/MAE validation.
Authentic sources remain image-stratified and training sampling remains label/group-balanced.

The only training-distribution change is `--robustness-profile low_resolution`:

- 80% of training views are downsampled to a randomly selected 32, 56, or 112 pixels and restored;
- half of degraded views also receive JPEG quality 50, 70, or 90;
- the exact same transform policy is applied to authentic and fake images, preventing resolution or
  compression from becoming a label shortcut;
- two independent frozen-feature views produce 51,250 cached training examples;
- the original single 768-D CLIP embedding and linear head remain the inference path.

The 769-weight head was adapted for 15 epochs at learning rate 3e-5 and weight decay 1e-2. Full
precision selection chose epoch 15 at checkpoint validation AUC 0.94481.

## Matched robustness comparison

All rows use the same 4,375 held images, no TTA, and the checkpoint's saved threshold.

| Condition | Run_005 AUC | Run_007 AUC | Delta |
|---|---:|---:|---:|
| clean | 0.943287 | 0.944819 | +0.001532 |
| jpeg_90 | 0.925458 | 0.926799 | +0.001341 |
| jpeg_70 | 0.921298 | 0.923294 | +0.001996 |
| jpeg_50 | 0.899155 | 0.900518 | +0.001363 |
| jpeg_30 | 0.894689 | 0.897295 | +0.002607 |
| blur_0.5 | 0.934835 | 0.935810 | +0.000975 |
| blur_1 | 0.902038 | 0.902283 | +0.000246 |
| blur_2 | 0.859154 | 0.860652 | +0.001498 |
| resize_0.5 | 0.897546 | 0.899209 | +0.001663 |
| resize_0.25 | 0.790483 | 0.793191 | +0.002708 |
| noise_0.02 | 0.929681 | 0.930859 | +0.001178 |
| noise_0.05 | 0.915240 | 0.916059 | +0.000819 |
| noise_0.1 | 0.888556 | 0.889137 | +0.000582 |
| color_0.8 | 0.932863 | 0.934602 | +0.001740 |
| color_1.2 | 0.912080 | 0.912878 | +0.000798 |
| crop_0.8 | 0.928538 | 0.929799 | +0.001261 |

The positive deltas are consistent but small. Synthetic low-resolution augmentation is therefore
useful as a safe incremental improvement, not a solution to severe resolution/domain shift.

## Negative ablations

- Robust TTA averaged clean/JPEG-70/resize-0.5/crop-0.8 views. It reduced held-generator clean AUC
  from 0.9433 to 0.9388 and balanced accuracy from 0.8636 to 0.8427, so it was rejected.
- Run_006 added a zero-initialized 32-pixel residual classifier over a second pass through the same
  CLIP backbone. It safely reproduced run_005 at epoch 0, then reached 0.94455 clean, 0.89960
  resize-0.5, and 0.79388 resize-0.25 AUC. These are marginally above run_007's resize results but
  require roughly twice the encoder inference, so run_006 was not promoted.
- Run_007 improves full CIFAKE AUC from 0.30238 to 0.30983, but FPR is 99.2% at its saved threshold.
  CIFAKE has already been inspected and is a development stress set, not a new blind benchmark.

## Operating points

The saved threshold 0.03231 prioritizes balanced accuracy: direct clean evaluation gives 16.21%
FPR and 89.24% recall. On this held set:

| Target | Threshold | Actual FPR | Recall | Balanced accuracy |
|---|---:|---:|---:|---:|
| 5% FPR | 0.190742 | 4.96% | 73.40% | 0.8422 |
| 1% FPR | 0.683309 | 0.96% | 53.00% | 0.7602 |

These thresholds are analytical operating points, not production calibration. Select and validate
the final threshold on a deployment-matched calibration set.

## Reproduction

```bash
traceguard-train data/merged_v3 \
  --fake-generator-disjoint-split --balance-groups \
  --init-checkpoint checkpoints/merged/run_005/best.pt --evaluate-initial \
  --robustness-profile low_resolution --freeze-backbone --cache-frozen-features \
  --feature-cache-views 2 --head-batch-size 2048 \
  --output-dir checkpoints/merged/run_007 \
  --epochs 15 --early-stopping-patience 5 --batch-size 128 --workers 4 \
  --lr 3e-5 --weight-decay 1e-2 --seed 42 --device cuda

traceguard-evaluate data/eval_merged_v3_fake_disjoint_seed42 \
  --checkpoint checkpoints/merged/run_007/best.pt \
  --output-dir outputs/evaluation/merged_run_007_held_generators \
  --transforms all --tta none --save-clean-predictions --batch-size 128 --device cuda
```

## Next experiment

Do not spend another run on the same synthetic resize policy. The next high-leverage input is a new,
independent low-resolution real/fake generator dataset with its own untouched test split. Evaluate
once on the still-protected WildFake COCO/DALL-E-Advanced subset, calibrate a fixed-FPR threshold on
deployment-matched data, and only then consider distillation of the 304M encoder.
