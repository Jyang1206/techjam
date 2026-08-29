# TraceGuard

TraceGuard is a hackathon-scale detector for AI-generated images that is designed around the
failure mode that matters in practice: images get compressed, resized, cropped, filtered, and
reposted. It combines a compact visual backbone with explicit frequency-domain evidence, trains
with realistic degradations, and can average predictions across redistributed views at inference.

The default EfficientNet-B0 model has roughly 4.0 million trainable parameters, comfortably below the
2-billion-parameter limit. TraceGuard outputs a calibrated probability rather than treating a
forensic model as ground truth.

## What is included

- A two-branch PyTorch detector with an EfficientNet spatial branch and radial FFT statistics.
- Robust training augmentation covering JPEG, blur, resize, noise, color changes, and center crop.
- A required directory-to-JSON inference command with `image_path` and `pred` fields.
- A reproducible robustness benchmark with clean/transformed metrics and error exemplars.
- A Gradio demo for individual inspection, stability probes, and batch JSON export.
- Focused tests for metrics, transformations, dataset discovery, and output behavior.

## Quickstart: run everything end-to-end

This is the fastest path from a clean checkout to a working model, predictions, a robustness
report, and a live demo. Each step links to the fuller section below if you need more detail or
want a different dataset.

```bash
# 1. Install (see "Setup" below for details)
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -e ".[dev]"

# 2. Train a model. This example downloads CIFAKE via Kaggle and trains in one step
#    (requires a Kaggle account/API token the first time). See "Data layout" and the
#    dataset-specific sections below for SID_Set, WildFake, or your own local folders.
traceguard-train --kaggle-dataset birdy654/cifake-real-and-ai-generated-synthetic-images \
  --epochs 5 --batch-size 64 --workers 2 \
  --output-dir checkpoints/cifake

# 3. Score a directory of images (this is the required challenge deliverable script,
#    see "Required inference output" below)
traceguard-predict path/to/image_directory \
  --checkpoint checkpoints/cifake/best.pt \
  --output predictions.json

# 4. Run the robustness benchmark against a labeled real/ + fake/ test folder
#    (see "Robustness evaluation" below). This writes the table for Devpost.
traceguard-evaluate data/test \
  --checkpoint checkpoints/cifake/best.pt \
  --output-dir outputs/evaluation

# 5. Launch the interactive demo (see "Demo" below) — good for the submission video
traceguard-demo --checkpoint checkpoints/cifake/best.pt
# then open http://127.0.0.1:7860
```

Notes:

- Step 2's checkpoint path (`checkpoints/cifake/best.pt`) is whatever you passed to
  `--output-dir` plus `best.pt` — reuse that exact path in steps 3-5.
- Checkpoints worth sharing are committed to this repository under `checkpoints/<source>/run_XXX/`,
  with `.pt` files tracked via **Git LFS** rather than plain git so they don't bloat repo history.
  Install Git LFS once before cloning/pulling (`brew install git-lfs && git lfs install`, or your
  OS's equivalent) — without it you'll only get small pointer files instead of real model weights.
  Not every checkpoint is pushed; if the one you need isn't there, train it yourself locally.
- Step 3's target directory just needs images (any mix of real/fake, unlabeled) — it produces a
  score per image. Step 4 needs a **labeled** folder with `real/` and `fake/` subfolders because it
  needs ground truth to compute accuracy/AUC.
- Run `pytest` at any point to check the test suite still passes: `pytest -q`.

## Setup

Python 3.10-3.12 is recommended. Create an environment and install the project:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

On macOS or Linux, activate with `source .venv/bin/activate`. The first training run downloads
ImageNet backbone weights. Pass `--no-pretrained` for an offline run.

## Data layout

Export public or properly licensed source data into class folders. Keep generator families and
near-duplicates in only one split when preparing a serious experiment; random image-level splits
can leak visual families and inflate accuracy.

```text
data/train/
  real/
    source_a/example_001.jpg
  fake/
    generator_a/example_002.png

data/test/
  real/
  fake/
```

Suitable starting points from the challenge are SID_Set, CIFAKE, and the training portion of
WildFake. The held-out COCO val2017 and DALL-E Advanced validation subset described in the brief
must not be used for training or model selection.

### Stream SID_Set

SID_Set is approximately 140 GB, so TraceGuard streams it from Hugging Face instead of requiring a
complete local download. Its labels are mapped as `0 -> authentic` and `1/2 -> AIGC` (fully
synthetic or tampered). This command uses 20,000 training and 4,000 validation images by default:

```bash
traceguard-train --hf-dataset saberzl/SID_Set \
  --epochs 5 --batch-size 24 --workers 2 \
  --output-dir checkpoints/sid
```

Increase the sample limits with `--max-train-samples` and `--max-validation-samples`; pass `0` to
stream an entire split. The default positive-class weight is `0.5` because SID_Set contains two
positive categories for one authentic category. Override it with `--positive-weight` when using a
different dataset or sampling policy.

Launch the demo with the SID_Set checkpoint using:

```bash
traceguard-demo --checkpoint checkpoints/sid/best.pt
```

SID_Set is published under CC BY 4.0. Attribute the dataset and cite the SIDA paper in the public
submission; the canonical license, author list, and BibTeX are maintained on the
[SID_Set dataset card](https://huggingface.co/datasets/saberzl/SID_Set).

### Download and train on CIFAKE

TraceGuard can download CIFAKE through the official Kaggle client and use its provided `train` and
`test` split instead of making a new random split. The default caps select a balanced 20,000-image
training subset and 4,000-image test subset:

```bash
traceguard-train \
  --kaggle-dataset birdy654/cifake-real-and-ai-generated-synthetic-images \
  --epochs 5 --batch-size 64 --workers 2 \
  --output-dir checkpoints/cifake
```

Pass `--max-train-samples 0 --max-validation-samples 0` to use all 100,000 training and 20,000 test
images. CIFAKE is balanced, so its default positive-class weight is `1.0`. The dataset is MIT
licensed and requires citation of both CIFAR-10 and Bird and Lotfi's CIFAKE paper; see the
[canonical Kaggle dataset card](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images).

### Train on a bounded WildFake subset

The ModelScope WildFake repository is approximately 1.29 TB with 3,694,313 manifest rows. Do not
download it indiscriminately on a hackathon machine. Install the ModelScope CLI separately, then
download the official manifests and only the image archives you intend to use:

```bash
pip install "modelscope>=1.34,<2"
modelscope download --dataset hy2628982280/WildFake \
  --local_dir data/WildFake \
  --include "split_train_test/csv_file/total_split/*" \
            "Images/Real/<chosen-source>/*" \
            "Images/Diffusion_based/<chosen-source>/*"
```

After extracting the selected image archives so their paths match the manifest, train with:

```bash
traceguard-train --wildfake-root data/WildFake \
  --max-train-samples 20000 --max-validation-samples 4000 \
  --epochs 5 --batch-size 24 --workers 2 \
  --output-dir checkpoints/wildfake
```

TraceGuard uses the official `train_metadata.csv` and `test_metadata.csv`, creates a balanced
reservoir sample from the archives that are actually present, and records ModelScope provenance in
the checkpoint. It excludes all real COCO rows and advanced DALL-E rows by default to keep the
challenge's protected COCO val2017 / DALL-E Advanced demonstration set out of training and threshold
selection. Do not use `--allow-protected-wildfake` for the challenge submission.

WildFake is marked Apache 2.0 on its
[ModelScope card](https://modelscope.cn/datasets/hy2628982280/WildFake/summary). Cite the
[WildFake AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/32363) in the submission.

## Train

```bash
traceguard-train data/train --epochs 12 --batch-size 32 --output-dir checkpoints
```

Training uses a stratified 85/15 split, class-weighted binary cross entropy, AdamW, cosine learning
rate decay, and a validation-selected decision threshold. The best checkpoint and epoch history are
written to `checkpoints/`. For a stronger result, create source-disjoint train/validation folders and
adapt the loader so the validation set never shares generator or capture pipelines with training.

## Required inference output

```bash
traceguard-predict path/to/image_directory \
  --checkpoint checkpoints/best.pt \
  --output predictions.json
```

The directory is scanned recursively. `pred` is the probability that an image is AIGC-generated:

```json
[
  {"image_path": "path/to/image_directory/photo.jpg", "pred": 0.0831},
  {"image_path": "path/to/image_directory/render.png", "pred": 0.9472}
]
```

Use `--tta none` to disable robust consensus and reduce inference from four views to one.

## Robustness evaluation

```bash
traceguard-evaluate data/test \
  --checkpoint checkpoints/best.pt \
  --output-dir outputs/evaluation
```

This evaluates clean images and every challenge transformation/severity. It produces:

- `metrics.json` and `metrics.csv` for analysis.
- `robustness_table.md` for the Devpost submission.
- `error_analysis.json` with the highest-confidence false positives and false negatives.

Report ROC-AUC, balanced accuracy, false-positive rate, and false-negative rate. Also compare the
worst transformed condition to clean performance. Accuracy alone can hide class imbalance and a
high false-positive cost.

## Demo

```bash
traceguard-demo --checkpoint checkpoints/best.pt
```

Open <http://127.0.0.1:7860>. The first tab displays probability and score stability under JPEG,
blur, resize, and crop probes. The batch tab creates the same JSON contract as the CLI.

## Technical approach

The spatial branch learns semantic and texture cues with EfficientNet-B0. The frequency branch
computes radial mean and variance across the log FFT magnitude plus color statistics. This explicit
signal gives the classifier a compact view of resampling, synthesis, and spectral artifacts while
the visual backbone handles content-dependent evidence. The features are fused only at the final
head so either branch can remain useful when redistribution weakens the other.

During training, one realistic degradation is sampled before the usual crop and flip. At inference,
robust consensus averages logits from clean, JPEG-70, half-resolution, and 80% crop views. The
stability spread shown in the demo is an uncertainty cue: large changes under benign transforms are
a reason to avoid an automatic decision.

## Limitations and responsible use

TraceGuard estimates image provenance; it does not prove it. It can fail on unseen generators,
heavy edits, screenshots, illustrations, computational photography, and images whose real camera
pipeline resembles synthetic artifacts. Dataset shortcuts and generator leakage are major risks.
False positives can unfairly discredit authentic creators, so low-confidence or unstable results
should go to human review and should never be the sole basis for punitive action.

With more time, we would add generator-disjoint cross-validation, probability calibration on a
deployment-matched set, provenance metadata such as C2PA as a separate signal, Grad-CAM validation,
and continuous tests against newly released generators and editing pipelines.

## Team contributions

Add names and concrete responsibilities here before submission. For a solo entry, state that all
modeling, engineering, evaluation, and presentation work was completed by the named participant.
