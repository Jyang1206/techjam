#!/usr/bin/env bash
set -euo pipefail

# Expected Hugging Face Job mounts:
#   /code       -> this repository's src/ directory (read-only)
#   /base       -> data/merged_v3 (read-only)
#   /validation -> data/eval_merged_v3_fake_disjoint_seed42 (read-only)
#   /init       -> checkpoints/merged/run_007 (read-only)
#   /outputs    -> an empty writable Job volume

python -m pip install --quiet \
  "datasets>=3.0,<5" \
  "numpy>=1.26,<3" \
  "Pillow>=10.0,<12" \
  "timm>=1.0,<2"

export PYTHONPATH=/code
experiment_root="$(mktemp -d)"
merged_root="${experiment_root}/merged_scaled_sid"
mkdir -p "${merged_root}/real" "${merged_root}/fake"

# Reuse the exact WildFake selection from merged_v3, without carrying over its old 10k SID sample.
find /base/real -maxdepth 1 -type f -name 'wildfake__*' \
  -exec cp -t "${merged_root}/real" {} +
find /base/fake -maxdepth 1 -type f -name 'wildfake__*' \
  -exec cp -t "${merged_root}/fake" {} +

# SID_Set labels: 0=real, 1=fully synthetic, 2=tampered. Keep all three as separate groups.
python -m traceguard.materialize \
  --hf-dataset saberzl/SID_Set \
  --hf-samples-per-label 0=25000 1=25000 2=25000 \
  --hf-group-by-label \
  --hf-shuffle-buffer 200 \
  --seed 42 \
  --output-dir "${merged_root}"

# Use the exact run-7 validation folder. Matching images are removed from the larger pool before
# training, including old-vs-new SID filenames whose subtype-aware group prefix has changed.
python -m traceguard.train "${merged_root}" \
  --validation-dir /validation \
  --balance-groups \
  --init-checkpoint /init/best.pt \
  --evaluate-initial \
  --robustness-profile low_resolution \
  --freeze-backbone \
  --cache-frozen-features \
  --feature-cache-views 2 \
  --head-batch-size 2048 \
  --positive-weight 1.0 \
  --output-dir /outputs/run_008_scaled_sid \
  --epochs 15 \
  --early-stopping-patience 5 \
  --batch-size 128 \
  --workers 4 \
  --lr 1e-5 \
  --weight-decay 1e-2 \
  --seed 42 \
  --device cuda
