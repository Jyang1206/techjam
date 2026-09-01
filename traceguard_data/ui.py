"""TraceGuard demo UI — drag-and-drop an image, get a real/fake verdict.

Wraps the exact predict_v2 pipeline (canonical JPEG-q95 -> Haar face gate ->
DINOv2-L + FFT -> face/scene heads -> max). Run on a GPU box:

    /opt/pytorch/bin/python3 -m traceguard_data.ui [--device cuda] [--share]
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from traceguard_data.extract import _jpeg, build_model, fft_features, preprocess
from traceguard_data.eval import load_head
from traceguard_data.predict_v2 import FaceGate, FACE_DOMINANT_FRAC


def make_app(device: str, heads_dir: Path, input_size: int = 224):
    import gradio as gr
    model, mean, std, torch = build_model(device, input_size)
    gate = FaceGate()
    heads = {c: load_head(heads_dir / f"{c}_head.pkl") for c in ("face", "scene")}

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
            return "upload an image", None
        im = Image.fromarray(img).convert("RGB")
        canon = _jpeg(im, 95)
        crop = gate.dominant_face(canon)
        scene_s = head_score(canon, "scene")
        rows = [f"scene head: **{scene_s:.4f}** (threshold "
                f"{heads['scene'][0]['threshold']:.2f})"]
        pred = scene_s
        if crop is not None:
            face_s = head_score(crop, "face")
            rows.append(f"face head (dominant face ≥{FACE_DOMINANT_FRAC:.0%} "
                        f"of frame): **{face_s:.4f}** (threshold "
                        f"{heads['face'][0]['threshold']:.2f})")
            pred = max(pred, face_s)
        else:
            rows.append("no dominant face detected → scene head only")
        flagged = (pred >= heads["face"][0]["threshold"] if crop is not None
                   and pred != scene_s else pred >= heads["scene"][0]["threshold"])
        verdict = ("🚨 **LIKELY AI-GENERATED / MANIPULATED**" if flagged
                   else "✅ **LIKELY REAL**")
        md = (f"# {verdict}\n\nfinal score (max of routes): **{pred:.4f}**\n\n"
              + "\n\n".join(rows)
              + "\n\n_TraceGuard: frozen DINOv2-L + FFT features, "
                "per-category heads. Scores are head sigmoids; higher = more "
                "likely fake._")
        return md, np.array(crop) if crop is not None else None

    return gr.Interface(
        fn=analyze,
        inputs=gr.Image(label="drop an image"),
        outputs=[gr.Markdown(label="verdict"),
                 gr.Image(label="face crop used (if any)")],
        title="TraceGuard — AI image detector",
        description="Canonical JPEG-q95 → face gate → DINOv2-L+FFT → "
                    "face/scene heads → max. Demo of the TechJam submission.",
        flagging_mode="never",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--heads-dir", default=str(Path.home() / "data/results"))
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    app = make_app(args.device, Path(args.heads_dir))
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
