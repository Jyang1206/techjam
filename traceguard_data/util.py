"""Hashing, image verification, deterministic sampling, and disk guards."""
import hashlib
import io
import random
import shutil

from PIL import Image

from . import config


def rng_for(context: str) -> random.Random:
    """A Random seeded from SEED plus a stable string context.

    CPython seeds str via SHA-512, so this is deterministic across runs and
    machines.
    """
    return random.Random(f"{config.SEED}:{context}")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def dest_name(source_key: str, orig_relpath: str) -> str:
    """Collision-safe destination filename.

    First 16 hex chars of sha1(source_zip_or_dataset + "/" + original relative
    path) + the original extension. Never a bare basename.
    """
    h = hashlib.sha1(f"{source_key}/{orig_relpath}".encode()).hexdigest()[:16]
    ext = orig_relpath.rsplit(".", 1)[-1].lower()
    ext = "jpg" if ext == "jpeg" else ext
    return f"{h}.{ext}"


def check_disk(stage: str) -> None:
    free_gb = shutil.disk_usage(config.DATA).free / 1e9
    if free_gb < config.MIN_FREE_GB:
        raise RuntimeError(
            f"disk guard: {free_gb:.1f}GB free < {config.MIN_FREE_GB}GB before stage {stage}"
        )


def verify_image(raw: bytes):
    """Pillow open+verify. Returns (ok, fmt, width, height, error)."""
    try:
        im = Image.open(io.BytesIO(raw))
        im.verify()
        im2 = Image.open(io.BytesIO(raw))  # verify() invalidates the object
        fmt = (im2.format or "").lower()
        w, h = im2.size
        return True, fmt, w, h, ""
    except Exception as e:  # noqa: BLE001 - any failure means "drop and log"
        return False, "", 0, 0, repr(e)[:200]


def sample_video_frames(vid2frames: dict, per_video: int, cap: int, context: str) -> dict:
    """Deterministically sample up to per_video frames per video, then keep
    whole videos (in seeded shuffled order) until ~cap frames are collected.

    Returns {video_id: [frame, ...]} preserving video grouping.
    """
    rng = rng_for(context)
    picked = {}
    for v in sorted(vid2frames):
        frames = sorted(vid2frames[v])
        if len(frames) > per_video:
            frames = sorted(rng.sample(frames, per_video))
        picked[v] = frames
    order = sorted(picked)
    rng.shuffle(order)
    out, total = {}, 0
    for v in order:
        if total >= cap:
            break
        out[v] = picked[v]
        total += len(picked[v])
    return out


def split_videos(video_ids, val_frac: float, context: str):
    """Video-disjoint split. Returns the set of val video ids."""
    vids = sorted(video_ids)
    rng = rng_for(context)
    n_val = round(len(vids) * val_frac)
    if len(vids) >= 2:
        n_val = max(1, n_val)
    return set(rng.sample(vids, n_val))
