"""predict_v2.py — TraceGuard inference script (challenge deliverable).

    python3 -m traceguard_data.predict_v2 \
        --input_dir /path/to/images --output predictions.json \
        [--device cpu|cuda] [--batch-size 32] [--heads-dir ~/data/results]

Emits: [{"image_path": "<path>", "pred": <float 0..1>}] for every image found
recursively under --input_dir (pred = probability the image is AI-generated
or manipulated).

Pipeline per image
------------------
1. Load, convert to RGB, re-encode as canonical JPEG q95 in memory — the same
   normalization the heads were trained with, so PNG/JPEG format signal is
   neutralized at inference exactly as in training.
2. Face gate: OpenCV Haar frontal-face detector (ships inside opencv-python;
   no external model downloads). A face is "dominant" when its bbox covers
   >= 15% of the frame area.
3. Dominant face  -> face crop (20% margin) -> DINOv2-L CLS + FFT features
   -> FACE head; the full frame is ALSO scored with the SCENE head and the
   final pred is the max of the two (both are plausible routes).
   No dominant face -> full frame -> SCENE head only.
4. Heads are the winners saved by heads.py (<category>_head.pkl: model,
   scaler, threshold metadata). Scores are head sigmoid outputs.

Dependencies: numpy, Pillow, torch, timm, opencv-python-headless,
scikit-learn (to unpickle a logistic-regression head). Runs on CPU
(slow but fine) or CUDA.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

if __package__ in (None, ""):  # allow `python3 traceguard_data/predict_v2.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from traceguard_data.extract import (_jpeg, build_model, fft_features,
                                     preprocess)
from traceguard_data.eval import load_head

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}
FACE_DOMINANT_FRAC = 0.15
CROP_MARGIN = 0.20


def find_images(root: Path):
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)


class FaceGate:
    def __init__(self):
        import cv2
        self.cv2 = cv2
        self.det = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def dominant_face(self, im: Image.Image):
        """Largest detected face bbox, or None if none covers >=15% of frame."""
        g = self.cv2.cvtColor(np.asarray(im), self.cv2.COLOR_RGB2GRAY)
        faces = self.det.detectMultiScale(g, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        if (w * h) / (im.width * im.height) < FACE_DOMINANT_FRAC:
            return None
        mx, my = int(w * CROP_MARGIN), int(h * CROP_MARGIN)
        return im.crop((max(0, x - mx), max(0, y - my),
                        min(im.width, x + w + mx), min(im.height, y + h + my)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input_dir_pos", nargs="?", default=None,
                    help="image directory (positional, team predict.py style)")
    ap.add_argument("--input_dir", default=None)
    ap.add_argument("--output", default="predictions.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--heads-dir", default=str(Path.home() / "data/results"))
    args = ap.parse_args()

    input_dir = args.input_dir or args.input_dir_pos
    if not input_dir:
        ap.error("provide an image directory (positional or --input_dir)")
    args.input_dir = input_dir
    heads_dir = Path(args.heads_dir)
    heads = {}
    for cat in ("face", "scene"):
        pkl = heads_dir / f"{cat}_head.pkl"
        if not pkl.exists():
            raise SystemExit(f"missing head: {pkl} (train with heads.py first)")
        heads[cat] = load_head(pkl)

    paths = find_images(Path(args.input_dir))
    print(f"{len(paths)} images found under {args.input_dir}", file=sys.stderr)
    model, mean, std, torch = build_model(args.device, args.input_size)
    gate = FaceGate()

    # jobs: one entry per (image, route); an image with a dominant face gets
    # a face-crop job AND a full-frame scene job; pred = max of its jobs.
    jobs = []  # (path_index, category, PIL image)
    skipped = []
    for pi, p in enumerate(paths):
        try:
            im = Image.open(p)
            im.load()
            canon = _jpeg(im, 95)
        except Exception as e:  # noqa: BLE001
            skipped.append((p, repr(e)[:80]))
            continue
        crop = gate.dominant_face(canon)
        if crop is not None:
            jobs.append((pi, "face", crop))
        jobs.append((pi, "scene", canon))

    preds = {}
    for lo in range(0, len(jobs), args.batch_size):
        chunk = jobs[lo:lo + args.batch_size]
        x = np.stack([preprocess(im, args.input_size, mean, std)
                      for _, _, im in chunk]).astype(np.float32)
        xt = torch.from_numpy(x).to(args.device)
        with torch.no_grad():
            if args.device.startswith("cuda"):
                with torch.autocast("cuda", torch.float16):
                    emb = model(xt)
            else:
                emb = model(xt)
        emb = emb.float().cpu().numpy()
        for (pi, cat, im), e in zip(chunk, emb):
            payload, sc, score_fn = heads[cat]
            feat = e[None, :]
            if payload["features"] == "dino+fft":
                feat = np.concatenate([feat, fft_features(im)[None, :]], axis=1)
            s = float(score_fn(sc.transform(feat))[0])
            preds[pi] = max(preds.get(pi, 0.0), s)
        done = min(lo + args.batch_size, len(jobs))
        if done % (args.batch_size * 8) < args.batch_size:
            print(f"  {done}/{len(jobs)} views scored", file=sys.stderr)

    errors = {str(p): e for p, e in skipped}
    out = []
    for i, p in enumerate(paths):
        if str(p) in errors:  # team predict.py convention: null + error field
            out.append({"image_path": str(p), "pred": None,
                        "error": errors[str(p)]})
        else:
            out.append({"image_path": str(p), "pred": round(preds[i], 6)})
    with open(args.output, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {len(out)} predictions -> {args.output} "
          f"({len(skipped)} unreadable, pred=null)", file=sys.stderr)
    for p, e in skipped[:10]:
        print(f"  skipped {p}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
