from __future__ import annotations

import argparse
import json
import tempfile
from functools import lru_cache
from pathlib import Path

import gradio as gr
from PIL import Image

from .inference import Predictor, score_spread

CSS = """
.gradio-container { max-width: 1180px !important; }
#title-row { border-bottom: 1px solid #d1d5db; padding-bottom: 12px; margin-bottom: 16px; }
#title-row h1 { font-size: 26px; line-height: 1.2; letter-spacing: 0; margin: 0; }
#title-row p { color: #4b5563; margin: 5px 0 0; }
.primary-panel { min-height: 390px; }
.result-panel { min-height: 390px; }
footer { display: none !important; }
"""
THEME = gr.themes.Base(primary_hue="teal", neutral_hue="gray", radius_size="sm")


@lru_cache(maxsize=4)
def load_predictor(checkpoint: str, device: str) -> Predictor:
    return Predictor.from_checkpoint(checkpoint, device=device)


def build_demo(checkpoint: str = "checkpoints/best.pt", device: str = "auto") -> gr.Blocks:
    def analyze(image: Image.Image | None, robust_tta: bool):
        if image is None:
            raise gr.Error("Upload an image to analyze.")
        try:
            predictor = load_predictor(checkpoint, device)
        except (FileNotFoundError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc
        tta = "robust" if robust_tta else "none"
        score = predictor.score_image(image, tta=tta)
        profile = predictor.stability_profile(image)
        spread = score_spread(profile)
        verdict = "AI-generated" if score >= predictor.threshold else "Authentic"
        stability = "stable" if spread <= 0.15 else "sensitive to redistribution"
        summary = (
            f"### {verdict}\n"
            f"AIGC probability **{score:.1%}** at a **{predictor.threshold:.1%}** decision threshold. "
            f"The result is **{stability}** across the standard probes (spread {spread:.1%})."
        )
        table = [[row["condition"], round(float(row["aigc_probability"]), 4)] for row in profile]
        return {"AI-generated": score, "Authentic": 1 - score}, summary, table

    def analyze_batch(files, robust_tta: bool):
        if not files:
            raise gr.Error("Upload at least one image.")
        try:
            predictor = load_predictor(checkpoint, device)
        except (FileNotFoundError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc
        tta = "robust" if robust_tta else "none"
        results = []
        for item in files:
            path = Path(item)
            try:
                with Image.open(path) as source:
                    score = predictor.score_image(source.convert("RGB"), tta=tta)
                results.append({"image_path": str(path), "pred": score})
            except (OSError, ValueError) as exc:
                results.append({"image_path": str(path), "pred": None, "error": str(exc)})
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="traceguard_", delete=False, encoding="utf-8"
        ) as handle:
            json.dump(results, handle, indent=2)
            output_path = handle.name
        rows = [[row["image_path"], row["pred"]] for row in results]
        return rows, output_path

    with gr.Blocks(title="TraceGuard") as demo:
        gr.Markdown(
            "# TraceGuard\nRobust image provenance screening for compressed, cropped, and reposted media.",
            elem_id="title-row",
        )
        with gr.Tabs():
            with gr.Tab("Inspect image"):
                with gr.Row(equal_height=True):
                    with gr.Column(elem_classes="primary-panel"):
                        image = gr.Image(
                            type="pil",
                            image_mode="RGB",
                            sources=["upload", "clipboard"],
                            label="Image",
                            height=340,
                        )
                        robust = gr.Checkbox(
                            value=True,
                            label="Robust consensus",
                            info="Average clean, JPEG, resized, and cropped views.",
                        )
                        analyze_button = gr.Button("Analyze", variant="primary")
                    with gr.Column(elem_classes="result-panel"):
                        confidence = gr.Label(label="Confidence", num_top_classes=2)
                        summary = gr.Markdown("Upload an image to begin.")
                        stability = gr.Dataframe(
                            headers=["Condition", "AIGC probability"],
                            datatype=["str", "number"],
                            interactive=False,
                            label="Redistribution stability",
                        )
                analyze_button.click(
                    analyze,
                    inputs=[image, robust],
                    outputs=[confidence, summary, stability],
                    api_name="analyze",
                )
            with gr.Tab("Batch score"):
                batch_files = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                    label="Images",
                )
                batch_robust = gr.Checkbox(value=True, label="Robust consensus")
                batch_button = gr.Button("Score batch", variant="primary")
                batch_table = gr.Dataframe(
                    headers=["Image path", "AIGC probability"],
                    datatype=["str", "number"],
                    interactive=False,
                    label="Predictions",
                )
                download = gr.File(label="JSON output", interactive=False)
                batch_button.click(
                    analyze_batch,
                    inputs=[batch_files, batch_robust],
                    outputs=[batch_table, download],
                    api_name="score_batch",
                )
    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the TraceGuard demo.")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_demo(args.checkpoint, args.device).queue(default_concurrency_limit=2).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=THEME,
        css=CSS,
    )


if __name__ == "__main__":
    main()
