---
name: traceguard-demo
description: Launch TraceGuard's interactive Gradio demo (`traceguard-demo`) — a browser UI for inspecting single images with redistribution-stability probes, and batch-scoring images with JSON export. Use whenever the user wants to visually demo the model, record the hackathon's required demo video, show TraceGuard working end-to-end, or interact with it through a browser instead of the CLI.
---

# Launch the TraceGuard demo

Run from the `techjam` repo root with its virtualenv active:

```bash
source .venv/bin/activate
traceguard-demo --checkpoint checkpoints/<run-name>/best.pt
```

Then open `http://127.0.0.1:7860` in a browser. Requires a trained checkpoint — same caveat as the
`traceguard-predict` skill: no checkpoint ships with the repo, weights are gitignored.

## What's in the UI

**"Inspect image" tab**: upload one image, get a verdict (AI-generated / Authentic), the raw AIGC
probability, and a small table showing how that probability shifts across 5 probes — clean, JPEG-70,
blur, half-resolution, 80% crop. A "Robust consensus" checkbox toggles whether the headline score
itself uses the 4-view TTA average or just the raw image. If the probability swings a lot across the
probes (spread > 15%), the summary text calls it out as "sensitive to redistribution" — a built-in
uncertainty signal, useful to point out on camera since it directly demonstrates the project's core
thesis (robustness under real-world transforms, not just clean-image accuracy).

**"Batch score" tab**: upload multiple images at once, get a table of predictions plus a downloadable
JSON file in the exact same `{"image_path", "pred"}` format the required `traceguard-predict` script
produces — good for showing that the CLI and the demo share the same underlying prediction logic.

## Flags

- `--host` / `--port` (defaults `127.0.0.1` / `7860`)
- `--share` — creates a temporary public Gradio link (`*.gradio.live`). Useful if recording remotely
  or sharing with a teammate, but treat it as a temporary/public URL, not something to leave running.
- `--device` — same auto/cpu/cuda/mps resolution as the other commands.

## Why this matters for the submission

This is the natural thing to screen-record for the challenge's required 2-4 minute public YouTube
demo video — it visually shows end-to-end behavior (upload → prediction → robustness probes) in a
way a terminal running `traceguard-predict` doesn't. Consider showing: one confidently-real image,
one confidently-fake image, and one where the stability probes visibly disagree — that last case is
the most persuasive illustration of why this project's approach (measuring robustness, not just
clean accuracy) matters.
