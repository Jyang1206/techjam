"""TraceGuard demo dashboard — detector, robustness results, known mistakes.

Wraps the exact predict_v2 pipeline (canonical JPEG-q95 -> Haar face gate ->
DINOv2-L + FFT -> face/scene heads -> max). Run on a GPU box:

    /opt/pytorch/bin/python3 -m traceguard_data.ui [--device cuda] [--share]

Tabs:
  1. Detector      — drag-and-drop an image, per-route scores, verdict
  2. Robustness    — clean vs transformed AUC table + grouped bar chart
  3. Known mistakes— representative false positives / negatives with scores

Robustness numbers are the frozen evaluation results (see
outputs/evaluation/dinov2/robustness_table.md); the gallery reads the error
CSVs written by eval.py if they are present next to the heads.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from traceguard_data.extract import _jpeg, build_model, fft_features, preprocess
from traceguard_data.eval import load_head
from traceguard_data.predict_v2 import FACE_DOMINANT_FRAC, FaceGate

DISCLAIMER = (
    "⚠️ **TraceGuard is a screening tool, not proof.** Scores are likelihoods, "
    "not verdicts: the system misses ~9 in 10 locally *tampered* images, flags "
    "roughly 1 in 4 real faces from unfamiliar sources, and was never trained on "
    "every generator in existence. Treat a high score as a reason to look closer "
    "— never as evidence on its own."
)

# frozen eval results (AUC), eval fakes vs the category's real reference
ROBUSTNESS = {
    "face": {
        "clean (original)": 0.7258, "canonical JPEG-q95": 0.7263,
        "JPEG q90": 0.7239, "JPEG q70": 0.7123, "JPEG q50": 0.7120,
        "JPEG q30": 0.7033, "blur σ0.5": 0.7346, "blur σ1.0": 0.7217,
        "blur σ2.0": 0.6861, "resize 0.5×": 0.7185, "resize 0.25×": 0.6962,
        "noise σ0.02": 0.7246, "noise σ0.05": 0.6847, "noise σ0.10": 0.6410,
        "saturation +20%": 0.7194, "saturation −20%": 0.7290,
        "brightness +20%": 0.7197, "brightness −20%": 0.7287,
        "contrast +20%": 0.7186, "contrast −20%": 0.7272, "crop 80%": 0.6852,
    },
    "scene": {
        "clean (original)": 0.8789, "canonical JPEG-q95": 0.8766,
        "JPEG q90": 0.8754, "JPEG q70": 0.8624, "JPEG q50": 0.8435,
        "JPEG q30": 0.8432, "blur σ0.5": 0.8894, "blur σ1.0": 0.9203,
        "blur σ2.0": 0.9526, "resize 0.5×": 0.9360, "resize 0.25×": 0.9367,
        "noise σ0.02": 0.8679, "noise σ0.05": 0.8358, "noise σ0.10": 0.7811,
        "saturation +20%": 0.8789, "saturation −20%": 0.8799,
        "crop 80%": 0.8808,
    },
}
FAMILIES = {
    "clean": ["clean (original)", "canonical JPEG-q95"],
    "JPEG": ["JPEG q90", "JPEG q70", "JPEG q50", "JPEG q30"],
    "blur": ["blur σ0.5", "blur σ1.0", "blur σ2.0"],
    "resize": ["resize 0.5×", "resize 0.25×"],
    "noise": ["noise σ0.02", "noise σ0.05", "noise σ0.10"],
    "colour": ["saturation +20%", "saturation −20%", "brightness +20%",
               "brightness −20%", "contrast +20%", "contrast −20%"],
    "crop": ["crop 80%"],
}
HEADLINE = """
| | Face head | Scene head |
|---|---|---|
| Validation AUC (held-out videos / generators) | 0.832 | 0.993 |
| Unseen manipulation methods / generator families | **0.726** | **0.972** |
| Weakest corruption cell | 0.641 (noise σ0.10) | 0.781 (noise σ0.10) |
| False-positive rate at threshold | 25–28% (shifted-domain real faces) | 2.2% |
"""


def _chart(out_path: Path) -> Path:
    """Grouped bar chart: mean AUC per transform family, per category."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = list(FAMILIES)
    face = [np.mean([ROBUSTNESS["face"][v] for v in FAMILIES[f]
                     if v in ROBUSTNESS["face"]]) for f in fams]
    scene = [np.mean([ROBUSTNESS["scene"][v] for v in FAMILIES[f]
                      if v in ROBUSTNESS["scene"]]) for f in fams]
    x = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, face, 0.4, label="face (unseen methods)", color="#4C78A8")
    ax.bar(x + 0.2, scene, 0.4, label="scene (unseen generators)",
           color="#F58518")
    ax.axhline(face[0], ls="--", lw=1, color="#4C78A8", alpha=0.6)
    ax.axhline(scene[0], ls="--", lw=1, color="#F58518", alpha=0.6)
    ax.axhline(0.5, ls=":", lw=1, color="grey")
    ax.set_xticks(x, fams)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Clean vs transformed (dashed = clean baseline, dotted = chance)")
    ax.legend(fontsize=8, loc="lower left")
    for i, (f, s) in enumerate(zip(face, scene)):
        ax.text(i - 0.2, f + 0.01, f"{f:.2f}", ha="center", fontsize=7)
        ax.text(i + 0.2, s + 0.01, f"{s:.2f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _robustness_markdown() -> str:
    rows = ["| transform | face AUC | scene AUC |", "|---|---|---|"]
    for view in ROBUSTNESS["face"]:
        f = ROBUSTNESS["face"][view]
        s = ROBUSTNESS["scene"].get(view)
        base_f, base_s = ROBUSTNESS["face"]["clean (original)"], \
            ROBUSTNESS["scene"]["clean (original)"]
        fd = "" if view.startswith("clean") else f" ({f - base_f:+.3f})"
        sd = ("" if s is None or view.startswith("clean")
              else f" ({s - base_s:+.3f})")
        rows.append(f"| {view} | {f:.3f}{fd} | "
                    + (f"{s:.3f}{sd} |" if s is not None else "not extracted |"))
    return "\n".join(rows)


_SHA_TO_NAME = None


def _sha_to_name(manifest: Path):
    """{sha256: basename} — gallery files are named by curated hash, not sha256."""
    global _SHA_TO_NAME
    if _SHA_TO_NAME is None:
        _SHA_TO_NAME = {}
        if manifest.exists():
            with open(manifest, newline="") as f:
                for r in csv.DictReader(f):
                    _SHA_TO_NAME[r["sha256"]] = Path(r["file_path"]).name
    return _SHA_TO_NAME


def _gallery(results_dir: Path, manifest: Path, category: str, kind: str):
    """[(image, caption)] for the top FP/FN of a category, if available."""
    gdir = results_dir / "error_gallery" / category
    csv_path = gdir / f"{kind}.csv"
    if not csv_path.exists():
        return []
    lookup = _sha_to_name(manifest)
    items = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            name = lookup.get(r.get("sha256", ""))
            img = (gdir / kind / name) if name else None
            if img is None or not img.exists():
                continue
            gen = r.get("generator") or "—"
            items.append((str(img), f"score {float(r['score']):.3f} · {gen}"))
    return items


def make_app(device: str, heads_dir: Path, results_dir: Path,
             manifest: Path, input_size: int = 224):
    import gradio as gr
    model, mean, std, torch = build_model(device, input_size)
    gate = FaceGate()
    heads = {c: load_head(heads_dir / f"{c}_head.pkl") for c in ("face", "scene")}
    chart = _chart(Path(results_dir) / "_ui_robustness_chart.png"
                   if Path(results_dir).exists() else Path("/tmp/rob.png"))

    def head_score(im: Image.Image, cat: str) -> float:
        p, sc, fn = heads[cat]
        x = preprocess(im, input_size, mean, std)[None].astype(np.float32)
        with torch.no_grad():
            xt = torch.from_numpy(x).to(device)
            if device.startswith("cuda"):
                with torch.autocast("cuda", torch.float16):
                    e = model(xt)
            else:
                e = model(xt)
        f = e.float().cpu().numpy()
        if p["features"] == "dino+fft":
            f = np.concatenate([f, fft_features(im)[None]], axis=1)
        return float(fn(sc.transform(f))[0])

    def analyze(img):
        if img is None:
            return "Upload an image to begin.", None
        im = Image.fromarray(img).convert("RGB")
        canon = _jpeg(im, 95)
        crop = gate.dominant_face(canon)
        scene_s = head_score(canon, "scene")
        scene_thr = heads["scene"][0]["threshold"]
        lines = [f"**scene head:** {scene_s:.4f}  (flags above {scene_thr:.2f})"]
        flagged = scene_s >= scene_thr
        pred = scene_s
        if crop is not None:
            face_s = head_score(crop, "face")
            face_thr = heads["face"][0]["threshold"]
            lines.append(f"**face head:** {face_s:.4f}  (flags above "
                         f"{face_thr:.2f}) — dominant face ≥"
                         f"{FACE_DOMINANT_FRAC:.0%} of frame")
            flagged = flagged or face_s >= face_thr
            pred = max(pred, face_s)
        else:
            lines.append("_no dominant face detected → scene head only_")
        verdict = ("### 🚨 Likely AI-generated / manipulated"
                   if flagged else "### ✅ Likely authentic")
        md = (f"{verdict}\n\nHighest route score: **{pred:.4f}**\n\n"
              + "\n\n".join(lines) + "\n\n---\n\n" + DISCLAIMER)
        return md, (np.array(crop) if crop is not None else None)

    with gr.Blocks(title="TraceGuard") as app:
        gr.Markdown("# TraceGuard — AI image detector\n"
                    "Frozen DINOv2-L + FFT frequency features, per-category "
                    "heads, face-detection routing.")
        with gr.Tab("Detector"):
            with gr.Row():
                with gr.Column():
                    inp = gr.Image(label="Drop an image")
                    btn = gr.Button("Analyze", variant="primary")
                with gr.Column():
                    out_md = gr.Markdown()
                    out_crop = gr.Image(label="Face crop used (if any)")
            btn.click(analyze, inp, [out_md, out_crop])
            inp.change(analyze, inp, [out_md, out_crop])
        with gr.Tab("Robustness results"):
            gr.Markdown("## Headline performance" + HEADLINE)
            gr.Image(str(chart), label="Mean AUC per transform family")
            gr.Markdown("## Per-transform detail\n\nAUC on held-out eval "
                        "slices; parenthesised values are the change from the "
                        "clean baseline.\n\n" + _robustness_markdown()
                        + "\n\n_Scene brightness/contrast views were not "
                          "extracted (time); saturation stands in for the "
                          "colour family there. Blur and downscaling *raise* "
                          "scene AUC — they suppress the high-frequency detail "
                          "that makes real photos look real._")
        with gr.Tab("Known mistakes"):
            gr.Markdown("## Where TraceGuard gets it wrong\n\n" + DISCLAIMER)
            for cat, title, note in [
                ("scene", "Scene — missed fakes (false negatives)",
                 "Dominated by **locally tampered** images: a real photo with a "
                 "small edited region. Global embeddings barely register the "
                 "edit (slice AUC 0.65, ~91% missed)."),
                ("scene", "Scene — false alarms (false positives)",
                 "Real photographs scored as fake; mostly unusual-for-training "
                 "real content."),
                ("face", "Face — missed fakes (false negatives)",
                 "Concentrated in **reenactment-style** methods (danet, "
                 "hyperreenact, SiT): the face texture is genuine, so the "
                 "artifacts are subtler than identity-swap seams."),
                ("face", "Face — false alarms (false positives)",
                 "Real faces from Celeb-DF — a different capture and "
                 "compression pipeline than the training reals (FPR 25–28%)."),
            ]:
                kind = "fn" if "negatives" in title else "fp"
                items = _gallery(Path(results_dir), manifest, cat, kind)
                gr.Markdown(f"### {title}\n\n{note}")
                if items:
                    gr.Gallery(value=items, columns=5, height=240,
                               show_label=False)
                else:
                    gr.Markdown("_(gallery files not found on this host — see "
                                "`outputs/evaluation/dinov2/error_gallery/`)_")
        gr.Markdown("---\n" + DISCLAIMER)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--heads-dir", default=str(Path.home() / "data/results"))
    ap.add_argument("--results-dir", default=str(Path.home() / "data/results"))
    ap.add_argument("--manifest",
                    default=str(Path.home() / "data/curated/manifest.csv"))
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    app = make_app(args.device, Path(args.heads_dir), Path(args.results_dir),
                   Path(args.manifest))
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
