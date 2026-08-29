# TraceGuard: trust signals that survive reposting

## Inspiration

AI-image detectors often look strongest on pristine benchmark files, while the images people
actually encounter have passed through messaging apps, social feeds, thumbnails, crops, and filters.
TraceGuard treats that gap as the core problem rather than an edge case.

## What it does

TraceGuard assigns an AIGC probability to an image and tests how stable that result remains under
common redistribution operations. It supports single-image inspection, batch JSON export, and a
repeatable benchmark spanning JPEG compression, blur, resize, noise, color adjustment, and crop.

## How we built it

We use PyTorch and timm to combine EfficientNet-B0 spatial features with explicit radial frequency
statistics from the image FFT. Training randomly applies challenge-matched degradations. Robust
consensus at inference averages four benign views, while the demo exposes the per-transformation
spread as a practical uncertainty signal. Gradio powers the end-to-end interface.

## What makes it different

The model does not only return a clean-image score. It asks whether its own conclusion survives the
way images move online. That makes robustness measurable for judges and legible to a reviewer. The
same predictor implementation is shared by the CLI, evaluation harness, and demo.

## Challenges

The central challenge is generalisation: a detector can memorize generator, dataset, or compression
signatures instead of learning transferable forensic evidence. We address this with explicit
degradation training, compact complementary features, source-aware evaluation guidance, and honest
error reporting. These measures reduce the risk but do not eliminate it.

## Accomplishments

- A lightweight model far below the 2B parameter cap.
- Exact challenge-compatible directory-to-JSON inference.
- Automated clean-versus-transformed evaluation and ranked error analysis.
- A demo that communicates both confidence and redistribution sensitivity.

## What we learned

Robustness is not one number. Different transformations suppress different evidence, and stable
probabilities are useful context alongside aggregate accuracy. False-positive analysis is especially
important because an authentic image incorrectly labeled synthetic carries a real trust cost.

## Next steps

We would expand generator-disjoint testing, calibrate scores per deployment domain, combine content
signals with signed provenance metadata, and add monitoring for distribution drift.

## Built with

Python, PyTorch, torchvision, timm, NumPy, Pillow, Gradio, pytest, and VS Code.

Before publishing, add the repository URL, evaluation table from
`outputs/evaluation/robustness_table.md`, team contributions, and public YouTube demo link.

