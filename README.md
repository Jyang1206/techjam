# TraceGuard

TraceGuard is a robustness-first detector for AI-generated and manipulated images. It is designed
for the conditions in which images are actually encountered: after JPEG compression, resizing,
cropping, filtering, screenshots, and reposting.

The submitted detector combines frozen **DINOv2-L** visual embeddings with explicit radial FFT
frequency statistics. A lightweight router sends dominant-face crops to a face specialist and every
full image to a scene specialist; the final prediction is the larger of the applicable route scores.
The output is a probability from 0 to 1, where a larger value means the image is more likely to be
AI-generated or manipulated.

> TraceGuard is a screening tool, not proof of provenance. Its scores should support human review,
> not serve as the sole basis for punitive or reputational decisions.

## Project overview

### Submitted DINOv2 pipeline

1. Re-encode the input in memory as JPEG quality 95 to reduce file-format shortcuts.
2. Detect whether a face occupies at least 15% of the frame using an OpenCV Haar cascade.
3. Extract a 1,024-dimensional frozen DINOv2-L embedding from the full image and, when applicable,
   the dominant face crop.
4. Concatenate the embedding with 70 radial FFT frequency features.
5. Score the full image with the scene head and the face crop with the face head.
6. Return the maximum route score as the final prediction.

The repository also contains an earlier EfficientNet-B0 + FFT implementation under
`src/traceguard/`. It follows the same degradation-aware design but is retained mainly as a
comparison branch. The commands in this README use the submitted DINOv2 pipeline unless a section
explicitly says otherwise.

### Included deliverables

- Recursive directory-to-JSON inference with `image_path` and `pred` fields.
- Shipped face and scene heads in `checkpoints/dinov2/`.
- A Gradio interface for live inference, robustness summaries, and error galleries.
- Deterministic data curation, feature extraction, head training, and evaluation modules.
- Generator/method-disjoint evaluation and a corruption grid covering JPEG, blur, resize, noise,
  color changes, and cropping.
- Checked-in evaluation tables, ablations, and representative false-positive/false-negative cases.

### Submission criteria mapping

| Criterion | Where it is addressed |
|---|---|
| Project overview | Architecture, routing, model outputs, and deliverables are described above. |
| Setup and installation | The environment, dependency, CUDA/CPU, and verification commands are documented below. |
| Reproducible results | The complete curation-to-evaluation workflow is provided under “Steps to reproduce the results.” |
| Limitations and reflection | Known failure modes and prioritized improvements are documented under “Limitations and future improvements.” |
| Team contributions | Responsibilities for Nazim, Li Heng, Zavier, Tobias, and Jie Yang are listed below. |
| Well-structured, commented code | The solution is separated into curation, feature extraction, training, evaluation, inference, and UI modules. Each main module contains a module-level description, focused functions/classes, and comments around non-obvious pipeline decisions. |
| Required inference script | [`traceguard_data/predict_v2.py`](traceguard_data/predict_v2.py) recursively accepts an image directory and writes one AIGC confidence score per image to JSON using the required `image_path` and `pred` fields. |

The implementation is organized by responsibility:

| Component | Main code |
|---|---|
| Dataset configuration and deterministic curation | `traceguard_data/config.py`, `run.py`, `df40.py`, `scenes.py`, `finalize.py` |
| DINOv2 and FFT feature extraction | `traceguard_data/extract.py` |
| Head training and ablations | `traceguard_data/heads.py` |
| Robustness evaluation and error analysis | `traceguard_data/eval.py` |
| Required batch inference | `traceguard_data/predict_v2.py` |
| Interactive demo | `traceguard_data/ui.py` |
| Automated checks | `tests/` |

### Results at a glance

| Evaluation | ROC-AUC | Notes |
|---|---:|---|
| Scene validation | 0.993 | Includes generator-disjoint validation |
| Scene, unseen Flux/SD3 generators | 0.972 | GenImage++ clean slice |
| Face validation | 0.832 | Video-disjoint split |
| Face, six unseen manipulation methods | 0.726 | Clean/canonical view |
| Scene, local tampering | 0.650 | Primary failure mode |

Across the checked-in corruption evaluation, the unseen-method face route ranges from 0.641 to
0.735 AUC and the combined scene evaluation ranges from 0.781 to 0.953 AUC. The full route-specific
tables are in [`outputs/evaluation/dinov2/`](outputs/evaluation/dinov2/).

### Repository layout

| Path | Purpose |
|---|---|
| `traceguard_data/predict_v2.py` | Submitted DINOv2 inference CLI |
| `traceguard_data/ui.py` | DINOv2 Gradio application |
| `traceguard_data/` | Curation, extraction, training, and evaluation pipeline |
| `checkpoints/dinov2/` | Shipped lightweight face and scene heads |
| `outputs/evaluation/dinov2/` | Metrics, ablations, audit artifacts, and error galleries |
| `src/traceguard/` | First-generation EfficientNet-B0 + FFT implementation |
| `tests/` | Unit and integration tests |

## Setup and installation

### Requirements

- Python 3.10-3.12
- About 8 GB of memory for inference; more is recommended for batch processing
- An NVIDIA CUDA GPU is strongly recommended, although CPU inference is supported
- Internet access on first use so `timm` can download the DINOv2-L backbone weights

The small trained heads are included in the repository. The much larger DINOv2-L backbone is
downloaded and cached automatically on the first inference run.

### Create an environment

From the repository root:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the project and DINOv2 pipeline dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install -r traceguard_data/requirements.txt
```

For CUDA, install the PyTorch build appropriate for the machine before running the final command;
otherwise pip may install a CPU-only build. Confirm the environment with:

```bash
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -m pytest -q
```

### Run inference

The required submission script is `traceguard_data/predict_v2.py`. It recursively scores every
supported image under the input directory and writes an AIGC confidence score between 0 and 1 for
each image:

```bash
python -m traceguard_data.predict_v2 path/to/images \
  --heads-dir checkpoints/dinov2 \
  --device cuda \
  --batch-size 8 \
  --output predictions.json
```

Use `--device cpu` when CUDA is unavailable. CPU inference works but is substantially slower.
The output follows this contract:

```json
[
  {"image_path": "path/to/images/photo.jpg", "pred": 0.0831},
  {"image_path": "path/to/images/render.png", "pred": 0.9472}
]
```

Unreadable images remain in the output with `"pred": null` and an `error` field.

The directory can also be passed with the named `--input_dir` option:

```bash
python -m traceguard_data.predict_v2 \
  --input_dir path/to/images \
  --heads-dir checkpoints/dinov2 \
  --device cpu \
  --output predictions.json
```

### Launch the demo

```bash
python -m traceguard_data.ui \
  --device cuda \
  --heads-dir checkpoints/dinov2 \
  --results-dir outputs/evaluation/dinov2
```

Open <http://127.0.0.1:7860>. Replace `cuda` with `cpu` if needed. Add `--share` only when a
temporary public Gradio URL is required.

## Steps to reproduce the results

There are two levels of reproduction: running the released detector with the shipped heads, and
rebuilding the dataset, embeddings, heads, and evaluation from raw sources.

### 1. Reproduce released inference

1. Complete the installation above.
2. Place evaluation images in a directory without changing their contents.
3. Run `traceguard_data.predict_v2` with `--heads-dir checkpoints/dinov2`.
4. Use the same input directory, `--input-size 224`, and device precision when comparing outputs.

The pipeline uses deterministic preprocessing. The included
`outputs/evaluation/dinov2/smoke_test_predictions.json` demonstrates the expected JSON structure.

### 2. Rebuild the reported experiment

The curated images (~41 GB) and cached embedding shards (~4 GB) are intentionally not committed.
Obtain the source datasets under their respective licenses and arrange them according to
`traceguard_data/config.py`. The experiment uses DF40, WildFake, SID-Set,
CommunityForensics-Small, and GenImage++; protected challenge evaluation data must never be used for
training or threshold selection.

The default configuration reads raw data from `~/data/` and fixes all sampling decisions with seed
42. From the repository root, run:

```bash
# A. Curate, verify, deduplicate, split, and write ~/data/curated/manifest.csv
python -m traceguard_data.run --stage all

# B. Score photorealism, inspect the generated sheet, then apply the selected threshold
python -m traceguard_data.realism score
python -m traceguard_data.realism sheet --threshold 0.5
python -m traceguard_data.realism apply --threshold 0.5

# C. Extract the complete clean and corruption-view feature sets
python -m traceguard_data.extract --category face --view-plan full \
  --out ~/data/embeddings/face --device cuda
python -m traceguard_data.extract --category scene --view-plan full \
  --out ~/data/embeddings/scene --device cuda

# D. Evaluate the shipped heads against the rebuilt embeddings
python -m traceguard_data.eval --embeddings ~/data/embeddings/face \
  --head checkpoints/dinov2/face_head.pkl --category face \
  --out outputs/evaluation/dinov2
python -m traceguard_data.eval --embeddings ~/data/embeddings/scene \
  --head checkpoints/dinov2/scene_head.pkl --category scene \
  --out outputs/evaluation/dinov2
python -m traceguard_data.eval --combine --out outputs/evaluation/dinov2

# E. Retrain all candidate heads and reproduce the ablation comparisons
python -m traceguard_data.heads --embeddings ~/data/embeddings/face \
  --category face --device cuda --out ~/data/rebuilt_heads
python -m traceguard_data.heads --embeddings ~/data/embeddings/scene \
  --category scene --device cuda --out ~/data/rebuilt_heads
```

The rebuilt heads are written outside the repository so they do not overwrite the released model.
Feature extraction is resume-safe and skips completed `(image hash, view)` pairs.

The shipped face head uses the degradation-trained `all_views` variant because it performed better
on every unseen-method robustness view, despite scoring about one point lower on clean validation
than the automatically selected canonical variant. Consequently, a fresh `heads.py` run reproduces
the candidate comparison but selects the canonical face head by clean validation AUC; use the
shipped `face_head.pkl` to reproduce the reported robustness table. This selection decision and the
complete ablations are documented in
[`outputs/evaluation/dinov2/ablation_face.md`](outputs/evaluation/dinov2/ablation_face.md) and
[`outputs/evaluation/dinov2/ablation_scene.md`](outputs/evaluation/dinov2/ablation_scene.md).

For the older EfficientNet branch, see the installed command help:

```bash
traceguard-train --help
traceguard-predict --help
traceguard-evaluate --help
traceguard-demo --help
```

## Limitations and future improvements

- **Local tampering is the largest blind spot.** The scene route reaches only about 0.65 AUC and a
  91% false-negative rate on locally tampered images because a global CLS embedding can overlook a
  small edited region. With more time, we would add patch-level tokens, localization supervision,
  and a tamper-specific head.
- **The face route has real-domain shift.** Celeb-DF authentic faces produce a 25-28% false-positive
  rate. We would expand the authentic-face domains, camera pipelines, demographics, and compression
  histories used during training.
- **The routing heuristic is simple.** Studio portraits with a face just below the 15% area
  threshold can be sent only to the scene head. A learned router or overlapping multi-crop inference
  would reduce this discontinuity.
- **Robustness remains uneven.** Heavy noise is the worst tested corruption, and the scene head was
  not trained on the complete degradation set. We would finish degradation-matched scene training
  and validate on screenshots and social-media recompression chains.
- **Scores are not universally calibrated.** A threshold fitted on one domain may not transfer to a
  different platform or content type. We would add deployment-specific calibration, abstention for
  uncertain cases, and drift monitoring.
- **Dataset shortcuts remain a risk.** JPEG normalization, source-aware splits, and held-out
  generators reduce shortcut learning but cannot eliminate it. Broader generator-disjoint and
  cross-dataset evaluation should be continuous as new generators appear.
- **No provenance metadata is used.** Future versions should combine forensic scores with C2PA or
  other signed provenance signals while keeping the two sources of evidence independently visible.

## Team member contributions

The team collaborated across the project, with the following primary areas of responsibility:

| Team member | Contributions |
|---|---|
| Nazim | Project coordination, solution architecture, experiment design, and DINOv2 integration |
| Li Heng | Dataset acquisition and curation, source-aware splitting, deduplication, and provenance tracking |
| Zavier | FFT feature engineering, head training, model ablations, and performance optimization |
| Tobias | Directory-to-JSON inference pipeline, Gradio interface, integration testing, and deployment support |
| Jie Yang | Robustness benchmarking, error analysis, result validation, documentation, and presentation |

## Responsible use

False positives can unfairly discredit authentic creators, while false negatives can create false
confidence. Treat TraceGuard as one signal among visual review, source context, metadata, and signed
provenance. Do not use a single model score as definitive evidence that an image is real or fake.
