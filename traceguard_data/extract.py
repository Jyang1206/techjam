"""DINOv2 embedding + FFT feature extraction over the curated manifest.

Outputs parquet shards keyed by (sha256, view_id):
    sha256, view_id, view_name, backbone,
    embedding  (1024-d float16, DINOv2-L CLS token),
    fft_features (70-d float32: 32 radial log-|FFT| means + 32 variances
                  on grayscale 256x256, + RGB channel means/stds),
    label, category, split, generator_or_method, source_dataset

Views:
  train/val:  0 = canonical (decode -> JPEG q95 re-encode in memory),
              1..6 = seeded sample (seed 42 + manifest row index) of 6
              challenge-grid transforms applied to the canonical pixels.
  eval:       0 = original as-is, 1 = canonical JPEG q95,
              2..N = the full challenge grid applied to the ORIGINAL pixels.

NOTE: the spec's challenge grid enumerates 15 transform-severity combos
(jpeg 90/70/50/30, blur 0.5/1/2, resize 0.5/0.25, noise 0.02/0.05/0.10,
color +-20%, crop 80%), so eval views run 2..16. view_name records the
transform, so a 16th combo can be appended later without renumbering.

Resume-safe: (sha256, view_id) pairs already present in existing shards are
skipped. Shards rotate every ~2000 images.

    python3 -m traceguard_data.extract --manifest ~/data/curated/manifest.csv \
        --out ~/data/embeddings --device cpu --batch-size 8 --limit 50
    python3 -m traceguard_data.extract --self-test
"""
import argparse
import csv
import hashlib
import io
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

BACKBONE = "dinov2_vitl14"
TIMM_MODEL = "vit_large_patch14_dinov2.lvd142m"
EMB_DIM = 1024
FFT_DIM = 70
SHARD_IMAGES = 2000
CANON_JPEG_Q = 95

# --- challenge grid ---------------------------------------------------------


def _jpeg(im, q):
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _blur(im, sigma):
    return im.convert("RGB").filter(ImageFilter.GaussianBlur(sigma))


def _resize(im, factor):
    w, h = im.size
    dw, dh = max(1, round(w * factor)), max(1, round(h * factor))
    return (im.convert("RGB")
            .resize((dw, dh), Image.BILINEAR)
            .resize((w, h), Image.BILINEAR))


def _noise(im, sigma, seed):
    rng = np.random.default_rng(seed)
    a = np.asarray(im.convert("RGB"), np.float32) / 255.0
    a = np.clip(a + rng.normal(0.0, sigma, a.shape), 0.0, 1.0)
    return Image.fromarray((a * 255).round().astype(np.uint8))


def _color(im, factor):
    return ImageEnhance.Color(im.convert("RGB")).enhance(factor)


def _enhance(im, kind, factor):
    return getattr(ImageEnhance, kind)(im.convert("RGB")).enhance(factor)


def _crop(im, frac):
    w, h = im.size
    cw, ch = max(1, round(w * frac)), max(1, round(h * frac))
    x, y = (w - cw) // 2, (h - ch) // 2
    return im.convert("RGB").crop((x, y, x + cw, y + ch))


# name -> fn(im, seed). Order is the canonical eval view order.
CHALLENGE_GRID = [
    ("jpeg90", lambda im, s: _jpeg(im, 90)),
    ("jpeg70", lambda im, s: _jpeg(im, 70)),
    ("jpeg50", lambda im, s: _jpeg(im, 50)),
    ("jpeg30", lambda im, s: _jpeg(im, 30)),
    ("blur0.5", lambda im, s: _blur(im, 0.5)),
    ("blur1.0", lambda im, s: _blur(im, 1.0)),
    ("blur2.0", lambda im, s: _blur(im, 2.0)),
    ("resize0.5", lambda im, s: _resize(im, 0.5)),
    ("resize0.25", lambda im, s: _resize(im, 0.25)),
    ("noise0.02", lambda im, s: _noise(im, 0.02, s)),
    ("noise0.05", lambda im, s: _noise(im, 0.05, s)),
    ("noise0.10", lambda im, s: _noise(im, 0.10, s)),
    ("color+20", lambda im, s: _color(im, 1.2)),
    ("color-20", lambda im, s: _color(im, 0.8)),
    ("crop80", lambda im, s: _crop(im, 0.8)),
    # appended (challenge color jitter = brightness/contrast/saturation ±20%;
    # keep at END so existing view_ids stay stable for resume)
    ("bright+20", lambda im, s: _enhance(im, "Brightness", 1.2)),
    ("bright-20", lambda im, s: _enhance(im, "Brightness", 0.8)),
    ("contrast+20", lambda im, s: _enhance(im, "Contrast", 1.2)),
    ("contrast-20", lambda im, s: _enhance(im, "Contrast", 0.8)),
]
GRID_NAMES = [n for n, _ in CHALLENGE_GRID]
GRID_FNS = dict(CHALLENGE_GRID)
TRAIN_VIEWS_PER_IMAGE = 6

EVAL_SPLITS_PREFIX = "eval_"


def _noise_seed(sha256: str, view_name: str) -> int:
    return int(hashlib.sha1(f"{sha256}:{view_name}".encode()).hexdigest()[:8], 16)


def views_for_row(im: Image.Image, row: dict, row_index: int, plan: str = "full"):
    """Yield (view_id, view_name, PIL image) for one manifest row.

    plan="phase1" is the deadline-critical subset: canonical only for
    train/val, orig+canonical for eval. Running plan="full" later adds the
    remaining views incrementally (resume skips existing (sha256, view_id)).
    """
    is_eval = row["split"].startswith(EVAL_SPLITS_PREFIX)
    canon = _jpeg(im, CANON_JPEG_Q)
    if is_eval:
        orig = im.convert("RGB")
        yield 0, "orig", orig
        yield 1, "canon_jpeg95", canon
        if plan == "phase1":
            return
        for i, name in enumerate(GRID_NAMES):
            yield 2 + i, name, GRID_FNS[name](orig, _noise_seed(row["sha256"], name))
    else:
        yield 0, "canon_jpeg95", canon
        if plan == "phase1":
            return
        rng = random.Random(42 + row_index)
        for i, name in enumerate(rng.sample(GRID_NAMES, TRAIN_VIEWS_PER_IMAGE)):
            yield 1 + i, name, GRID_FNS[name](canon, _noise_seed(row["sha256"], name))


# --- FFT features -----------------------------------------------------------

_RADIAL_BINS = None


def _radial_bins():
    global _RADIAL_BINS
    if _RADIAL_BINS is None:
        yy, xx = np.mgrid[0:256, 0:256]
        r = np.hypot(yy - 128.0, xx - 128.0)
        _RADIAL_BINS = np.minimum((r / (r.max() / 32.0)).astype(np.int32), 31).ravel()
    return _RADIAL_BINS


def fft_features(im: Image.Image) -> np.ndarray:
    """32 radial means + 32 radial variances of log-|FFT| + RGB means/stds."""
    g = np.asarray(im.convert("L").resize((256, 256), Image.BILINEAR),
                   np.float32) / 255.0
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g)))).ravel()
    bins = _radial_bins()
    counts = np.bincount(bins, minlength=32).astype(np.float32)
    mean = np.bincount(bins, weights=mag, minlength=32) / counts
    sq = np.bincount(bins, weights=mag * mag, minlength=32) / counts
    var = np.maximum(sq - mean * mean, 0.0)
    rgb = np.asarray(im.convert("RGB").resize((256, 256), Image.BILINEAR),
                     np.float32) / 255.0
    out = np.concatenate([
        mean, var, rgb.mean(axis=(0, 1)), rgb.std(axis=(0, 1)),
    ]).astype(np.float32)
    assert out.shape == (FFT_DIM,)
    return out


# --- backbone ---------------------------------------------------------------


def build_model(device: str, input_size: int):
    import timm
    import torch
    model = timm.create_model(TIMM_MODEL, pretrained=True, num_classes=0,
                              img_size=input_size)
    model.eval().to(device)
    cfg = timm.data.resolve_model_data_config(model)
    mean = np.array(cfg["mean"], np.float32).reshape(3, 1, 1)
    std = np.array(cfg["std"], np.float32).reshape(3, 1, 1)
    return model, mean, std, torch


def preprocess(im: Image.Image, input_size: int, mean, std) -> np.ndarray:
    a = np.asarray(im.convert("RGB").resize((input_size, input_size),
                                            Image.BILINEAR), np.float32) / 255.0
    return (a.transpose(2, 0, 1) - mean) / std


def embed_batch(model, torch, device: str, chw: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(chw).to(device)
    with torch.no_grad():
        if device.startswith("cuda"):
            with torch.autocast("cuda", torch.float16):
                out = model(x)
        else:
            out = model(x)
    emb = out.float().cpu().numpy().astype(np.float16)
    assert emb.shape[1] == EMB_DIM
    return emb


# --- parquet io -------------------------------------------------------------


def existing_pairs(out_dir: Path):
    import pyarrow.parquet as pq
    pairs = set()
    for f in sorted(out_dir.glob("shard_*.parquet")):
        t = pq.read_table(f, columns=["sha256", "view_id"])
        pairs.update(zip(t["sha256"].to_pylist(), t["view_id"].to_pylist()))
    return pairs


def next_shard_index(out_dir: Path) -> int:
    idxs = [int(f.stem.split("_")[1]) for f in out_dir.glob("shard_*.parquet")]
    return max(idxs) + 1 if idxs else 0


def write_shard(out_dir: Path, index: int, rows: list) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq
    emb_flat = np.concatenate([r["embedding"] for r in rows])
    fft_flat = np.concatenate([r["fft_features"] for r in rows])
    table = pa.table({
        "sha256": [r["sha256"] for r in rows],
        "view_id": pa.array([r["view_id"] for r in rows], pa.int16()),
        "view_name": [r["view_name"] for r in rows],
        "backbone": [BACKBONE] * len(rows),
        "embedding": pa.FixedSizeListArray.from_arrays(
            pa.array(emb_flat, pa.float16()), EMB_DIM),
        "fft_features": pa.FixedSizeListArray.from_arrays(
            pa.array(fft_flat, pa.float32()), FFT_DIM),
        "label": pa.array([int(r["label"]) for r in rows], pa.int8()),
        "category": [r["category"] for r in rows],
        "split": [r["split"] for r in rows],
        "generator_or_method": [r["generator_or_method"] for r in rows],
        "source_dataset": [r["source_dataset"] for r in rows],
    })
    path = out_dir / f"shard_{index:05d}.parquet"
    pq.write_table(table, path)
    return path


# --- self-test --------------------------------------------------------------


def self_test(manifest: Path, audit_dir: Path) -> None:
    rows = list(csv.DictReader(open(manifest)))
    src = Image.open(manifest.parent / rows[0]["file_path"]).convert("RGB")
    w, h = src.size
    strip_ims, failures = [], []
    for name in ["canon_jpeg95"] + GRID_NAMES:
        fn = ((lambda im, s: _jpeg(im, CANON_JPEG_Q)) if name == "canon_jpeg95"
              else GRID_FNS[name])
        out = fn(src, 12345)
        try:
            assert isinstance(out, Image.Image), "not a PIL image"
            expect = ((round(w * 0.8), round(h * 0.8)) if name == "crop80"
                      else (w, h))
            assert out.size == expect, f"size {out.size} != {expect}"
            assert np.asarray(out.convert("RGB")).dtype == np.uint8
            f = fft_features(out)
            assert f.shape == (FFT_DIM,) and f.dtype == np.float32
        except AssertionError as e:
            failures.append(f"{name}: {e}")
        thumb = out.convert("RGB").copy()
        thumb.thumbnail((128, 128))
        strip_ims.append((name, thumb))
    strip = Image.new("RGB", (130 * len(strip_ims), 148), (20, 20, 20))
    for i, (name, thumb) in enumerate(strip_ims):
        strip.paste(thumb, (130 * i + (130 - thumb.width) // 2, 4))
    audit_dir.mkdir(parents=True, exist_ok=True)
    strip.save(audit_dir / "transform_strip.png")
    print(f"self-test: {len(strip_ims)} transforms on {rows[0]['file_path']} "
          f"({w}x{h}); strip -> {audit_dir / 'transform_strip.png'}")
    if failures:
        for f in failures:
            print("  FAIL", f)
        raise SystemExit(1)
    print("self-test: ALL TRANSFORMS PASS")


# --- main -------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(Path.home() / "data/curated/manifest.csv"))
    ap.add_argument("--out", default=str(Path.home() / "data/embeddings"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--input-size", type=int, default=224,
                    help="model input side; multiple of 14 (224 default, 518 native)")
    ap.add_argument("--category", choices=["face", "scene"])
    ap.add_argument("--view-plan", choices=["full", "phase1"], default="full",
                    help="phase1 = canonical-only train/val + orig/canon eval")
    ap.add_argument("--exclude-splits", default="",
                    help="comma-separated split names to skip entirely")
    ap.add_argument("--limit", type=int,
                    help="dry-run: N manifest rows, seeded shuffle across categories/splits")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    curated_root = manifest.parent
    if args.self_test:
        self_test(manifest, curated_root / "audit")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(manifest)))
    indexed = list(enumerate(rows))  # manifest row index drives train-view rng
    if args.category:
        indexed = [(i, r) for i, r in indexed if r["category"] == args.category]
    excl = {s for s in args.exclude_splits.split(",") if s}
    if excl:
        indexed = [(i, r) for i, r in indexed if r["split"] not in excl]
    if args.limit:
        random.Random(42).shuffle(indexed)
        indexed = indexed[:args.limit]

    done = existing_pairs(out_dir)
    print(f"{len(indexed)} images to process; {len(done)} (sha256,view) pairs "
          f"already in {out_dir}", flush=True)
    model, mean, std, torch = build_model(args.device, args.input_size)

    shard_idx = next_shard_index(out_dir)
    shard_rows, shard_img_count = [], 0
    pend_meta, pend_px = [], []
    t0 = time.time()
    n_views = n_imgs = 0

    def flush_batch():
        nonlocal pend_meta, pend_px, n_views
        for lo in range(0, len(pend_px), args.batch_size):
            chunk = pend_px[lo:lo + args.batch_size]
            emb = embed_batch(model, torch, args.device,
                              np.stack(chunk).astype(np.float32))
            for m, e in zip(pend_meta[lo:lo + args.batch_size], emb):
                m["embedding"] = e
                shard_rows.append(m)
                n_views += 1
        pend_meta, pend_px = [], []

    def flush_shard():
        nonlocal shard_rows, shard_idx, shard_img_count
        if shard_rows:
            p = write_shard(out_dir, shard_idx, shard_rows)
            print(f"wrote {p.name}: {len(shard_rows)} rows", flush=True)
            shard_idx += 1
            shard_rows, shard_img_count = [], 0

    for row_index, r in indexed:
        try:
            im = Image.open(curated_root / r["file_path"])
            im.load()
        except Exception as e:  # noqa: BLE001
            print(f"SKIP unreadable {r['file_path']}: {e!r}", flush=True)
            continue
        emitted = False
        for view_id, view_name, view_im in views_for_row(im, r, row_index,
                                                         args.view_plan):
            if (r["sha256"], view_id) in done:
                continue
            pend_meta.append({
                "sha256": r["sha256"], "view_id": view_id,
                "view_name": view_name,
                "fft_features": fft_features(view_im),
                "label": r["label"], "category": r["category"],
                "split": r["split"],
                "generator_or_method": r["generator_or_method"],
                "source_dataset": r["source_dataset"],
            })
            pend_px.append(preprocess(view_im, args.input_size, mean, std))
            emitted = True
        if len(pend_px) >= args.batch_size:
            flush_batch()
        if emitted:
            n_imgs += 1
            shard_img_count += 1
        if shard_img_count >= SHARD_IMAGES:
            flush_batch()
            flush_shard()
        if n_imgs and n_imgs % 200 == 0:
            print(f"  {n_imgs} images / {n_views} views, "
                  f"{(time.time()-t0)/n_imgs:.2f}s/img", flush=True)
    flush_batch()
    flush_shard()

    dt = time.time() - t0
    print(f"DONE: {n_imgs} images, {n_views} views in {dt:.1f}s "
          f"({dt/max(n_imgs,1):.2f}s/img, {dt/max(n_views,1):.3f}s/view)",
          flush=True)

    # full-job estimate from this run's measured throughput
    trainval = sum(1 for r in rows if not r["split"].startswith(EVAL_SPLITS_PREFIX))
    ev = len(rows) - trainval
    total_views = trainval * (1 + TRAIN_VIEWS_PER_IMAGE) + ev * (2 + len(GRID_NAMES))
    if n_views:
        s_per_view_cpu = dt / n_views
        bytes_per_row = sum(
            f.stat().st_size for f in out_dir.glob("shard_*.parquet")
        ) / max(len(existing_pairs(out_dir)), 1)
        print(f"ESTIMATE: full job = {total_views} views "
              f"({trainval} train/val x7 + {ev} eval x{2+len(GRID_NAMES)})")
        for speedup in (15, 25):
            print(f"ESTIMATE: A10G batch128 at {speedup}x CPU: "
                  f"{total_views*s_per_view_cpu/speedup/3600:.1f} GPU-hours")
        print(f"ESTIMATE: parquet size ~{total_views*bytes_per_row/1e9:.1f} GB "
              f"({bytes_per_row:.0f} B/row measured)")


if __name__ == "__main__":
    main()
