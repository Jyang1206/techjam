# Run_008 scaled SID_Set experiment

## Question

Does increasing SID_Set from 10,000 binary-balanced images to 75,000 subtype-balanced images improve
the promoted run_007 detector without changing its WildFake held-generator validation task?

## Outcome

Run_008 completed all 15 head-training epochs locally. The selected epoch-15 checkpoint reached
0.945880 ROC-AUC and 0.869667 balanced accuracy on the exact 4,375-image run_007 holdout, improving
run_007's 0.944819 AUC by 0.001061 without changing the validation task.

The improvement did not resolve CIFAKE transfer. On the full 20,000-image CIFAKE test set with no
TTA, clean AUC increased from run_007's 0.309833 to 0.320434, while balanced accuracy remained
0.4935 and false-positive rate remained 0.9905. On the deterministic 2,000-image seed-42 robustness
subset, mean AUC across clean plus 15 transformations was 0.330814. CIFAKE remains an inspected
development stress test rather than a checkpoint-selection dataset.

## Data

- SID_Set label 0 (authentic): 25,000.
- SID_Set label 1 (fully synthetic): 25,000.
- SID_Set label 2 (tampered): 25,000.
- The exact 20,000 WildFake files already present in `data/merged_v3`: 7,500 authentic and 12,500
  fake.
- Total materialized pool before holdout exclusion: 95,000.
- Exact validation folder: `data/eval_merged_v3_fake_disjoint_seed42`, containing the same 1,875
  authentic images and 2,500 MAGE/VQVAE/VQGAN/MAE fakes used to report run_007.
- Expected training count after exclusion: 90,625.

The original 10,000 SID_Set files are reused from the seed-42 materialization. Additional SID_Set
samples use streaming seed 43 so the Hub begins from different shuffled shards instead of
retransferring the already materialized prefix. Training and validation splitting remain seed 42.

The subtype-aware SID filenames create separate `hf__SID_Set__label_0`, `label_1`, and `label_2`
groups. Group-balanced replacement sampling therefore gives equal label mass, then equal mass to
each source/subtype within a label. `--positive-weight 1.0` avoids applying a second class correction
after the sampler has already balanced real and fake examples.

Protected WildFake COCO-real and DALL-E Advanced images remain absent because the workflow copies
only the previously approved `merged_v3` WildFake selection.

## Training

Run_008 initializes the self-contained run_007 CLIP ViT-L/14 checkpoint and evaluates it at epoch 0.
The encoder remains frozen; two independent low-resolution/JPEG augmented feature views are cached,
and only the 769-parameter linear head is adapted. A lower learning rate (`1e-5`) makes this a
conservative data-scale adaptation: the initializer remains selected unless validation AUC improves.

The cloud entry point is `scripts/hf_run_scaled_sid.sh`. It expects five mounted directories, as
documented at the top of the script, and writes `best.pt` plus `history.json` to the writable output
volume.

## Acceptance criteria

- Primary: clean ROC-AUC above run_007's 0.944819 on the exact 4,375-image holdout.
- Guardrail: no regression in clean false-positive rate at a fixed threshold chosen independently
  of this validation set.
- Robustness: compare all 16 transformations with `--tta none`.
- External stress test: evaluate CIFAKE for diagnosis, but do not use it to select the checkpoint or
  claim a new blind result.
