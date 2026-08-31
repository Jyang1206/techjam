# TraceGuard dataset manifest spec (v2 rebuild)

Frozen decisions for the merged multi-source rebuild. All sampling is seeded and
recorded; split unit is generator / method / video — never individual images.
Categories map to the routed experts: face, scene, document.

## Global settings

- Seed: 42
- Backbone: DINOv2-L (frozen); embeddings cached as float16 parquet
- Degradation grid (applied at extraction time, from challenge spec):
  - JPEG quality 90 / 70 / 50 / 30
  - Gaussian blur σ 0.5 / 1.0 / 2.0
  - Resize 0.5× and 0.25× down, then upscale back
  - Gaussian noise σ 0.02 / 0.05 / 0.10
  - Color jitter brightness/contrast/saturation ±20%
  - Center crop 80%
- Comparison protocol: select by generator-disjoint validation AUC; all runs
  evaluated with `--tta none`; demo set (COCO val2017 / DALL·E Advanced) touched
  only once per finalized candidate, never for training/threshold/selection.

## Face category

**Train (ff domain only):**
| Source | Methods | Count |
|---|---|---|
| DF40 fakes | SimSwap, InSwapper, e4s, BlendFace (FS); SD-2.1, DiT, PixArt (EFS); e4e (FE) | ~1,500 frames/method, max 3 frames per video |
| Reals | FF++ real crops (from DF40 links) | ~12,000 |

**Held out (never trained):**
- Methods: FaceDancer, MobileSwap (FS); SiT, MidJourney (EFS)
- Domain: entire cdf domain of all training methods
- Reals: Celeb-DF real crops
- Optional extra eval: DiFF slice, DeepFakeFace slice

## Scene category

**Train:**
| Source | Selection | Count |
|---|---|---|
| SID_Set | label 0 (real) + label 1 (fully synthetic) | 5,000 + 5,000 |
| WildFake | Diffusion_based: Midjourney + Stable Diffusion archives (exact paths from official manifests; protected COCO / DALL·E-Advanced rows excluded) | ~10,000 fakes |
| CommunityForensics-Small | streamed with cap; includes paired reals | ~15,000 |
| WildFake reals | ImageNet archive (already materialized) | existing |

**Eval only:**
- GenImage++ (full; Flux/SD3 unseen-generator axis)
- Chameleon (full; hard human-deceptive set)
- SID_Set label 2 (tampered) + TGIF — patch-mode eval
- CommunityForensics-Eval (CompEval)
- MIML slice — patch-mode eval for manual forgeries (optional)

## Document category — **TBC (phase 2)**

Deferred until face + scene are downloading. Placeholder plan:
- Train: DocTamper (Kaggle copy test → email approval; LMDB→PNG conversion
  needed) with SROIE/CORD/FUNSD as matched clean reals; fallback =
  self-generated SD-inpainting on SROIE/FUNSD, or Text-Sleuth (Pandey) synthetic set
- Eval: AIForge-Doc, T-SROIE
- Status of ETTD (Qu et al.): asked in DocTamper email, unreleased publicly
- Until the head is validated: router detects documents and the demo layer flags
  "document input — outside validated scope"; JSON output still emits a numeric
  `pred` via the fusion fallback

## Deliverable mapping (why these choices)

- Robustness Evaluation Summary ← degradation grid × category eval matrix
- Error Analysis Note ← per-category FP/FN exemplars from held-out slices
- Judging: eval design and threat framing are scored directly (no hidden test
  set); model AUC on the demo set is a sanity check, not a target
