# TraceGuard — DINOv2 branch (`traceguard_data/`)

AI-generated-image detector: frozen **DINOv2-L** embeddings + explicit **FFT
frequency statistics**, with separate **face** and **scene** heads and
face-detection routing at inference. This directory is the second-generation
system in the TraceGuard project (the first, an end-to-end EfficientNet-B0 +
FFT model, lives in `src/traceguard/`); both share the same core hypothesis —
frequency features plus robustness-first, degradation-matched training.

## The judged script

```bash
python3 -m traceguard_data.predict_v2 /path/to/images --output predictions.json [--device cpu|cuda]
```

Recursively scores every image and writes
`[{"image_path": ..., "pred": <0..1>}]` (higher = more likely AI-generated;
unreadable files get `"pred": null` + an `"error"` field). Pipeline per image:
canonical JPEG-q95 re-encode → OpenCV Haar face gate (face ≥15% of frame →
face head on the crop, always scene head on the frame) → `max` of routes.

## Setup

```bash
pip install torch torchvision timm pyarrow scikit-learn "opencv-python-headless<5" pillow numpy
```

Model heads are loaded from `--heads-dir` (default `~/data/results/`):
`face_head.pkl`, `scene_head.pkl` — produced by `heads.py` (see below).
DINOv2-L weights download automatically via timm on first run
(`vit_large_patch14_dinov2.lvd142m`). CPU works; CUDA is ~20× faster.

## Reproducing the results

Everything is deterministic (seed 42) and stage-based. From a machine with the
raw datasets under `~/data/` (see `config.py` for expected layout):

```bash
python3 -m traceguard_data.run --stage all          # curate ~101k images + manifest + audit
python3 -m traceguard_data.realism score            # CLIP photoreal filter (then: sheet, apply)
python3 -m traceguard_data.extract --view-plan phase1 --category face   # embeddings (repeat per category / plan)
python3 -m traceguard_data.heads --embeddings ~/data/embeddings/face --category face --device cuda
python3 -m traceguard_data.eval  --embeddings ~/data/embeddings/face --head ~/data/results/face_head.pkl --category face
python3 -m traceguard_data.eval  --combine          # results_summary.md + robustness_table.md
```

`extract.py` is resume-safe (skips existing `(sha256, view_id)` pairs), so
extraction phases can be run incrementally. A Gradio demo is in `ui.py`
(`python3 -m traceguard_data.ui --share`).

## Data

DF40 (face manipulation methods + FF++/Celeb-DF reals), WildFake (Midjourney),
SID-Set, CommunityForensics-Small, GenImage++ (Flux/SD3; evaluation only).
Curated with: video-disjoint face splits, generator-disjoint scene val,
collision-safe hashed filenames, Pillow verification, sha256 dedupe, a CLIP
photoreal filter (stylized content moved to an `eval_stylized` slice), and a
documented real-diversity patch. Full provenance and all deviations:
`audit_report.md` (published alongside the curated set).

## Headline results

| Test | AUC |
|---|---|
| Scene val (incl. 31 held-out generators) | 0.993 |
| Scene, unseen generator families (Flux/SD3) | 0.972 |
| Scene, local tampering | 0.65 ← weakest |
| Face val (11 seen methods, video-disjoint) | 0.832 |
| Face, 6 unseen methods | 0.726 |
| Robustness across the 19-cell corruption grid | face 0.64–0.73, scene 0.78–0.95 |

Full tables: `outputs/evaluation/dinov2/robustness_table.md` and
`results_summary.md`.

## Limitations / with more time

- **Local tampering is the blind spot** (0.65 AUC, 91% FNR): global CLS
  embeddings can't see small edited regions → patch-level features or a
  tamper-specific head is the top priority.
- **Face real-domain shift**: 25–28% FPR on Celeb-DF reals; needs a face-real
  diversity patch (as done successfully for scene reals — see audit report).
- **Scene head is not degradation-augmented** (extraction time ran out);
  the identical recipe lifted the face category +3–6 AUC points on every
  eval view, so this is a validated next step.
- Studio-portrait false positives when the face is <15% of frame (routed to
  the scene head, which under-trained on studio photography).
- 224px inputs; DINOv2's native 518px and/or ViT-g are untested upside.
- Ensemble with the EfficientNet-B0 branch (complementary failure modes).

## Team contributions

<!-- fill in: who did what across the EfficientNet branch, the DINOv2 branch,
     data curation, evaluation, demo, writeup -->

## What is in this repo (DINOv2 branch)

| Path | Contents |
|---|---|
| `traceguard_data/` | Pipeline modules: curation (`run.py`, `df40.py`, `scenes.py`, `finalize.py`), realism filter (`realism.py`), embedding extraction (`extract.py`), heads (`heads.py`), evaluation (`eval.py`), **judged inference script (`predict_v2.py`)**, demo UI (`ui.py`) |
| `traceguard_data/pipeline_scripts/` | One-off stage scripts used during the build (raw HF pull, meta repair, parquet label probe, real-diversity patch) — kept for provenance |
| `checkpoints/dinov2/` | Trained heads: `face_head.pkl`, `scene_head.pkl` (shipped) plus `face_head_canonical.pkl`, `face_head_allviews.pkl`, `scene_head_prepatch.pkl` (ablation/comparison variants) |
| `outputs/evaluation/dinov2/` | `results_summary.md`, `robustness_table.md`, per-category `results_*.md` and `ablation_*.md/json`, degradation + score plots, `error_gallery/` (top-15 FP/FN per category with CSVs), `smoke_test_predictions.json` (example script output) |
| `outputs/evaluation/dinov2/dataset/` | `audit_report.md` (full dataset provenance, deviations, bias find/fix), `manifest.csv.gz` (every curated image: sha256, source, generator, label, category, split, style), audit visuals |

Not in the repo (size): the curated images (~41 GB) and embedding parquet shards
(~4 GB) live in S3; `manifest.csv.gz` fully specifies which image is which.
