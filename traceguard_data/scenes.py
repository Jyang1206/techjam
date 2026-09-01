"""Scene pipeline: WildFake Midjourney, SID-Set, CommunityForensics, GenImage++."""
import csv
import zipfile

from . import config, util
from .manifest import PartWriter


def _split_by_file(items, val_frac: float, context: str):
    """Deterministic per-file split. Returns set of items assigned to val."""
    items = sorted(items)
    rng = util.rng_for(context)
    rng.shuffle(items)
    n_val = round(len(items) * val_frac)
    return set(items[:n_val])


def stage_wildfake() -> None:
    pw = PartWriter("wildfake")
    with zipfile.ZipFile(config.WILDFAKE_ZIP) as zf:
        members = sorted(
            n for n in zf.namelist()
            if not n.endswith("/") and n.lower().endswith(config.IMG_EXTS))
        rng = util.rng_for("wildfake:sample")
        chosen = sorted(rng.sample(members, min(config.WILDFAKE_SAMPLE, len(members))))
        val = _split_by_file(chosen, config.SCENE_VAL_FRAC, "wildfake:val")
        for m in chosen:
            pw.ingest(
                zf.read(m),
                source_key="wildfake/part_1.zip", orig_relpath=m,
                source_dataset="wildfake", generator="midjourney",
                domain="wildfake", label=1, category="scene",
                split="val" if m in val else "train",
            )
    pw.close()
    print(f"[wildfake] written={pw.written} dropped={pw.dropped}", flush=True)


def stage_sid_set() -> None:
    pw = PartWriter("sid_set")
    plan = [
        ("0", 0, "sid_real", None),
        ("1", 1, "sid_fake", None),
        ("2", 1, "sid_tampered", "eval_tampered"),
    ]
    for sub, label, generator, split_fixed in plan:
        d = config.SID_SET / sub
        files = sorted(p.name for p in d.iterdir()
                       if p.suffix.lower() in config.IMG_EXTS)
        val = (set() if split_fixed else
               _split_by_file(files, config.SCENE_VAL_FRAC, f"sid_set:{sub}:val"))
        for name in files:
            split = split_fixed or ("val" if name in val else "train")
            pw.ingest(
                (d / name).read_bytes(),
                source_key="sid_set", orig_relpath=f"{sub}/{name}",
                source_dataset="sid_set", generator=generator,
                domain="sid_set", label=label, category="scene", split=split,
            )
        print(f"[sid_set] class {sub}: {len(files)} files", flush=True)
    pw.close()
    print(f"[sid_set] written={pw.written} dropped={pw.dropped}", flush=True)


def stage_commfor() -> None:
    pw = PartWriter("commfor")
    with open(config.COMMFOR / "meta.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    models = sorted({r["model_name"] for r in rows if r["label"] == "1"})
    rng = util.rng_for("commfor:val_models")
    n_val = max(1, round(len(models) * config.COMMFOR_VAL_MODEL_FRAC))
    val_models = set(rng.sample(models, n_val))
    real_files = [r["file"] for r in rows if r["label"] == "0"]
    val_reals = _split_by_file(real_files, config.SCENE_VAL_FRAC, "commfor:val_reals")
    print(f"[commfor] holding out {len(val_models)}/{len(models)} model_names for val", flush=True)
    for r in rows:
        rel = r["file"]
        if r["label"] == "1":
            generator = r["model_name"]
            split = "val" if r["model_name"] in val_models else "train"
        else:
            generator = "commfor_real"
            split = "val" if rel in val_reals else "train"
        pw.ingest(
            (config.COMMFOR / rel).read_bytes(),
            source_key="commfor_small", orig_relpath=rel,
            source_dataset="commfor_small", generator=generator,
            domain="commfor", label=int(r["label"]), category="scene",
            split=split,
        )
    pw.close()
    print(f"[commfor] written={pw.written} dropped={pw.dropped}", flush=True)


def stage_genimagepp() -> None:
    pw = PartWriter("genimagepp")
    for folder, generator in [("flux_realistic", "flux_realistic"),
                              ("stable_diffusion_v_3_0", "sd3")]:
        root = config.GENIMAGEPP / folder
        files = sorted(p for p in root.rglob("*")
                       if p.is_file() and p.suffix.lower() in config.IMG_EXTS)
        for p in files:
            rel = str(p.relative_to(config.GENIMAGEPP))
            pw.ingest(
                p.read_bytes(),
                source_key="genimagepp", orig_relpath=rel,
                source_dataset="genimagepp", generator=generator,
                domain="genimagepp", label=1, category="scene",
                split="eval_genimagepp",
            )
        print(f"[genimagepp] {generator}: {len(files)} files", flush=True)
    pw.close()
    print(f"[genimagepp] written={pw.written} dropped={pw.dropped}", flush=True)
