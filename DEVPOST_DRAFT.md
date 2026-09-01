# TraceGuard — robust AI-generated image detection

## How our solution addresses the problem statement

Detectors that ace pristine benchmarks fall apart on real-world images — re-compressed, resized, filtered, or from generators they never saw. TraceGuard treats that robustness gap as the core problem. Our submission is a two-branch system:

- **DINOv2 branch (submitted script)**: every image is normalized (in-memory JPEG-q95 re-encode, neutralizing format shortcuts), routed by a face detector (dominant face → face head on the crop; always a scene head on the full frame), and scored by lightweight heads over frozen **DINOv2-L embeddings concatenated with 70-dim radial FFT frequency statistics** — semantic evidence and generator frequency fingerprints together. Final score = max of routes.
- **EfficientNet-B0 branch** (first-generation, in `src/traceguard/`): an end-to-end model sharing the same FFT + degradation-matched-training philosophy; kept as a comparison point with complementary failure modes.

Deliberate protocol choices back the robustness claim: video-disjoint face splits, generator-disjoint scene validation, eval slices built only from manipulation methods and generator families excluded from training, and a 19-cell corruption grid evaluated per view. Training the face head **on the challenge's own corruption grid** improved every unseen-method eval number by 3–6 AUC points — robustness training measurably works.

The required inference script is `traceguard_data/predict_v2.py`: takes an image directory, outputs JSON with `image_path` and `pred` per image, runs on CPU.

## Development tools
VS Code + Claude Code (agentic pipeline orchestration on AWS), AWS EC2 (t3.large curation box, 2× g4dn.xlarge T4 GPU boxes), S3, tmux; Windows + Ubuntu.

## Models / APIs
- **DINOv2-L** (`vit_large_patch14_dinov2.lvd142m`, frozen, via timm) — 1024-d CLS embeddings
- **CLIP ViT-B/32** (open_clip, laion2b) — zero-shot photoreal-vs-stylized dataset filter
- **OpenCV Haar cascade** — face gating at inference
- Heads: scikit-learn logistic regression (face) and a small PyTorch MLP (scene)

## Libraries & frameworks
PyTorch, timm, open_clip_torch, scikit-learn, OpenCV, Pillow, NumPy, PyArrow/Parquet, Gradio (demo UI), matplotlib.

## Datasets & assets
- **DF40** (11 train + 6 held-out face manipulation methods; FF++ & Celeb-DF real frames)
- **WildFake** (Midjourney), **SID-Set**, **CommunityForensics-Small** (208 community generators, generator-disjoint splits)
- **GenImage++** (Flux-realistic, SD3) — held out entirely for unseen-generator evaluation
- Curated set: ~101k images, deterministic seed-42 sampling, sha256 dedupe, Pillow verification, CLIP realism filtering, full audit trail (`audit_report.md`) including a live-caught-and-fixed dataset bias ("pristine = fake" shortcut, patched with 6k pristine reals and verified on fresh holdouts).

## Robustness evaluation summary (clean vs transformed, AUC)

| Transform | Face (unseen methods) | Scene (unseen generators) |
|---|---|---|
| **Clean / canonical** | **0.726** | **0.877** |
| JPEG q90 → q30 | 0.724 → 0.703 | 0.875 → 0.843 |
| Blur σ0.5 → σ2.0 | 0.735 → 0.686 | 0.889 → 0.953* |
| Resize 0.5× / 0.25× | 0.719 / 0.696 | 0.936 / 0.937* |
| Noise σ.02 → σ.10 | 0.725 → 0.641 | 0.868 → 0.781 |
| Color/brightness/contrast ±20% | 0.719–0.729 | ~0.879 |
| Center crop 80% | 0.68 | ~0.87 |

*Blur/resize suppress the high-frequency content fakes rely on → detection gets easier. Worst single cell: face 0.641, scene 0.781 (heavy noise). Reference: scene val 0.993, genimagepp clean slice 0.972 AUC / 2.2% FPR. Full per-cell table and degradation curves in `outputs/evaluation/dinov2/`.

## Error analysis note

- **False negatives (scene)**: dominated by **locally tampered images** — all top-15 highest-confidence misses are sid_tampered (slice AUC 0.65, FNR 91%). Global embeddings can't see small edited regions; this is our clearest limitation and next-step (patch-level features).
- **False negatives (face)**: concentrated in **reenactment-style methods** (danet, hyperreenact — 7/15 top misses): per-frame they preserve genuine face texture, leaving subtler artifacts than identity-swap seams.
- **False positives (face)**: essentially all from **real-domain shift** — Celeb-DF reals at 25–28% FPR; the head's notion of "real face" is anchored to FF++ processing.
- **False positives (scene)**: pristine studio photography of people whose faces fall under the 15% routing threshold — a niche under-represented in scene real training data.
- **Trade-offs we chose**: degradation-augmented training costs ~1 point of seen-method val AUC but buys +3–6 on every unseen-method view (shipped); the strict scene threshold (0.873) favors low FPR (2.2%) over recall (misses ~18–20% of clean unseen-generator fakes); `max(face, scene)` routing raises sensitivity at some FPR cost on portraits.

## Links
- Demo video: **[YouTube link — to record]**
- Live demo (Gradio): **[link — 72h-limited, see repo to relaunch]**
- Repository: https://github.com/Jyang1206/techjam/tree/naz
