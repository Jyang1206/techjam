"""Photoreal-vs-stylized filter for scene fakes (commfor_small + wildfake).

Mechanism: open_clip ViT-B-32 (laion2b_s34b_b79k) zero-shot. Each image is
scored softmax(100 * [sim(photo prompts), sim(stylized prompts)])[photo],
so realism_score in [0,1]; higher = more photographic.

Policy (user-specified): TAG, don't delete. Rows scoring below the threshold
get style=stylized and, if they sat in train/val, move to split=eval_stylized.
Nothing is removed from disk. Counts are then backfilled to the original
train/val targets from each source's not-yet-sampled remainder (same seed
discipline; commfor keeps model_name-disjoint val, new model_names are
assigned to val with probability 0.15 via a seeded per-model draw).

Stages:
    python3 -m traceguard_data.realism score            # score + histogram
    python3 -m traceguard_data.realism sheet --threshold 0.5
    python3 -m traceguard_data.realism apply --threshold 0.5
"""
import argparse
import csv
import io
import random
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from . import config, util

CURATED = config.CURATED
WORK = config.WORK
SCORES_CSV = WORK / "realism_scores.csv"

PHOTO_PROMPTS = [
    "a photograph",
    "a photo taken with a camera",
    "a realistic photograph of a scene",
]
STYLIZED_PROMPTS = [
    "an anime illustration",
    "a cartoon drawing",
    "a digital painting",
    "concept art",
    "a 3d render",
    "an illustration",
]

# original train/val targets that must survive filtering
TARGETS = {
    ("wildfake", "train"): 8500, ("wildfake", "val"): 1500,
    ("commfor_small", "train"): 6771, ("commfor_small", "val"): 1229,
}
# unused label-1 parquet files (fake side is files <= ~110; 0/1/10 consumed)
COMMFOR_BACKFILL_FILES = ["11", "12", "13", "14"]


def _load_manifest():
    with open(CURATED / "manifest.csv", newline="") as f:
        r = csv.DictReader(f)
        return list(r), r.fieldnames


def _scorer(batch_size=32):
    import open_clip
    import torch
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    model.eval()
    tok = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        t = model.encode_text(tok(PHOTO_PROMPTS + STYLIZED_PROMPTS))
        t = t / t.norm(dim=-1, keepdim=True)
        text = torch.stack([t[:len(PHOTO_PROMPTS)].mean(0),
                            t[len(PHOTO_PROMPTS):].mean(0)])
        text = text / text.norm(dim=-1, keepdim=True)

    def score(images):
        out = []
        for lo in range(0, len(images), batch_size):
            x = torch.stack([preprocess(im.convert("RGB"))
                             for im in images[lo:lo + batch_size]])
            with torch.no_grad():
                v = model.encode_image(x)
                v = v / v.norm(dim=-1, keepdim=True)
                p = (100.0 * v @ text.T).softmax(dim=-1)[:, 0]
            out.extend(float(s) for s in p)
        return out
    return score


def targets_rows(man):
    return [r for r in man
            if (r["source_dataset"] == "wildfake")
            or (r["source_dataset"] == "commfor_small" and r["label"] == "1")]


def stage_score(batch_size: int) -> None:
    man, _ = _load_manifest()
    rows = targets_rows(man)
    score = _scorer(batch_size)
    WORK.mkdir(parents=True, exist_ok=True)
    with open(SCORES_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sha256", "file_path", "source_dataset", "split", "realism_score"])
        buf_rows, buf_ims = [], []

        def flush():
            for r, s in zip(buf_rows, score(buf_ims)):
                w.writerow([r["sha256"], r["file_path"], r["source_dataset"],
                            r["split"], f"{s:.4f}"])
            buf_rows.clear(); buf_ims.clear()

        for i, r in enumerate(rows):
            buf_rows.append(r)
            buf_ims.append(Image.open(CURATED / r["file_path"]))
            if len(buf_rows) >= 256:
                flush()
            if i and i % 2000 == 0:
                print(f"scored {i}/{len(rows)}", flush=True)
        flush()
    print(f"SCORED {len(rows)} images -> {SCORES_CSV}", flush=True)
    hist(np.array([float(x["realism_score"]) for x in
                   csv.DictReader(open(SCORES_CSV))]))


def hist(scores: np.ndarray) -> None:
    edges = np.linspace(0, 1, 11)
    counts, _ = np.histogram(scores, edges)
    for lo, hi, n in zip(edges, edges[1:], counts):
        print(f"  {lo:.1f}-{hi:.1f}: {n}", flush=True)


def stage_sheet(threshold: float) -> None:
    scores = list(csv.DictReader(open(SCORES_CSV)))
    for s in scores:
        s["v"] = float(s["realism_score"])
    below = sorted((s for s in scores if s["v"] < threshold),
                   key=lambda s: -s["v"])[:50]
    above = sorted((s for s in scores if s["v"] >= threshold),
                   key=lambda s: s["v"])[:50]
    cell, cols = 100, 10
    sheet = Image.new("RGB", (cell * cols, cell * 11 + 30), (25, 25, 25))
    for block, items in ((0, above), (1, below)):
        y0 = block * (cell * 5 + 30) + 30
        for i, s in enumerate(items):
            try:
                im = Image.open(CURATED / s["file_path"]).convert("RGB")
                im.thumbnail((cell, cell))
                sheet.paste(im, ((i % cols) * cell, y0 + (i // cols) * cell))
            except Exception as e:  # noqa: BLE001
                print("sheet error", s["file_path"], repr(e)[:60])
    out = CURATED / "audit" / "realism_borderline.png"
    sheet.save(out)
    print(f"SHEET {out} — top 5 rows: kept (just above {threshold}), "
          f"bottom 5 rows: tagged stylized (just below)", flush=True)


def _assign_backfill_split(deficits: dict, source: str, forced=None):
    for split in ("val", "train"):
        if forced and split != forced:
            continue
        if deficits.get((source, split), 0) > 0:
            return split
    return None


def stage_apply(threshold: float, batch_size: int) -> None:
    from .manifest import PartWriter  # reuse verified ingest machinery
    scores = {s["sha256"]: float(s["realism_score"])
              for s in csv.DictReader(open(SCORES_CSV))}
    man, fields = _load_manifest()
    for r in man:
        r.setdefault("style", "")
        r.setdefault("realism_score", "")

    meta = {m["file"]: m for m in
            csv.DictReader(open(config.COMMFOR / "meta.csv"))}
    tagged = {"wildfake": 0, "commfor_small": 0}
    deficits = dict.fromkeys(TARGETS, 0)
    for r in targets_rows(man):
        v = scores.get(r["sha256"])
        if v is None:
            continue
        r["realism_score"] = f"{v:.4f}"
        if v < threshold:
            r["style"] = "stylized"
            if r["split"] in ("train", "val"):
                key = (r["source_dataset"], r["split"])
                deficits[key] = deficits.get(key, 0) + 1
                r["split"] = "eval_stylized"
                tagged[r["source_dataset"]] += 1
        else:
            r["style"] = "realistic"
    print("tagged stylized:", tagged, "| deficits:", deficits, flush=True)

    existing_sha = {r["sha256"] for r in man}
    used_wf = {r["orig_relpath"] for r in man
               if r["source_dataset"] == "wildfake"}
    new_rows = []
    # Recover rows ingested by a previously interrupted apply run (their image
    # files are on disk and their rows sit in the uncommitted part tmp file).
    part_tmp = config.PARTS / "backfill.csv.tmp"
    if part_tmp.exists():
        with open(part_tmp, newline="") as f:
            for pr in csv.DictReader(f):
                key = (pr["source_dataset"], pr["split"])
                if pr["sha256"] in existing_sha or deficits.get(key, 0) <= 0:
                    continue
                new_rows.append({
                    "file_path": pr["file_path"], "sha256": pr["sha256"],
                    "source_dataset": pr["source_dataset"],
                    "generator_or_method": pr["generator_or_method"],
                    "domain": pr["domain"], "label": pr["label"],
                    "category": pr["category"], "split": pr["split"],
                    "orig_relpath": pr["orig_relpath"],
                    "style": "realistic", "realism_score": "",
                })
                existing_sha.add(pr["sha256"])
                if pr["source_dataset"] == "wildfake":
                    used_wf.add(pr["orig_relpath"])
                deficits[key] -= 1
        print(f"recovered {len(new_rows)} backfill rows from interrupted run; "
              f"deficits now: {deficits}", flush=True)
    score_fn = _scorer(batch_size)
    pw = PartWriter("backfill")

    def ingest(raw, source_key, relpath, source_dataset, generator, domain,
               split, sc):
        if not pw.ingest(raw, source_key=source_key, orig_relpath=relpath,
                         source_dataset=source_dataset, generator=generator,
                         domain=domain, label=1, category="scene", split=split):
            return False
        name = util.dest_name(source_key, relpath)
        sha = util.sha256_bytes(raw)
        new_rows.append({
            "file_path": f"images/{source_dataset}/{name}", "sha256": sha,
            "source_dataset": source_dataset, "generator_or_method": generator,
            "domain": domain, "label": "1", "category": "scene",
            "split": split, "orig_relpath": relpath,
            "style": "realistic", "realism_score": f"{sc:.4f}",
        })
        existing_sha.add(sha)
        deficits[(source_dataset, split)] -= 1
        return True

    # --- wildfake backfill from the zip remainder ---------------------------
    if any(deficits[("wildfake", s)] > 0 for s in ("train", "val")):
        with zipfile.ZipFile(config.WILDFAKE_ZIP) as zf:
            cands = sorted(n for n in zf.namelist()
                           if not n.endswith("/")
                           and n.lower().endswith(config.IMG_EXTS)
                           and n not in used_wf)
            util.rng_for("wildfake:backfill").shuffle(cands)
            batch = []
            for name in cands:
                if not any(deficits[("wildfake", s)] > 0 for s in ("train", "val")):
                    break
                raw = zf.read(name)
                try:
                    im = Image.open(io.BytesIO(raw)); im.load()
                except Exception:  # noqa: BLE001
                    continue
                batch.append((name, raw, im))
                if len(batch) < 64:
                    continue
                for (nm, rw, _), sc in zip(batch, score_fn([b[2] for b in batch])):
                    split = _assign_backfill_split(deficits, "wildfake")
                    if split and sc >= threshold and \
                            util.sha256_bytes(rw) not in existing_sha:
                        ingest(rw, "wildfake/part_1.zip", nm, "wildfake",
                               "midjourney", "wildfake", split, sc)
                batch = []
            for (nm, rw, _), sc in zip(batch, score_fn([b[2] for b in batch]) if batch else []):
                split = _assign_backfill_split(deficits, "wildfake")
                if split and sc >= threshold and \
                        util.sha256_bytes(rw) not in existing_sha:
                    ingest(rw, "wildfake/part_1.zip", nm, "wildfake",
                           "midjourney", "wildfake", split, sc)
    print("after wildfake backfill, deficits:", deficits, flush=True)

    # --- commfor backfill from unused label-1 parquet files -----------------
    if any(deficits[("commfor_small", s)] > 0 for s in ("train", "val")):
        val_models = {r["generator_or_method"] for r in man
                      if r["source_dataset"] == "commfor_small"
                      and r["label"] == "1" and r["split"] == "val"}
        train_models = {r["generator_or_method"] for r in man
                        if r["source_dataset"] == "commfor_small"
                        and r["label"] == "1" and r["split"] == "train"}

        def model_split(model):
            if model in val_models:
                return "val"
            if model in train_models:
                return "train"
            s = ("val" if util.rng_for(f"commfor:model:{model}").random() < 0.15
                 else "train")
            (val_models if s == "val" else train_models).add(model)
            return s

        import signal
        import time as _time

        from huggingface_hub import HfFileSystem
        import pyarrow.parquet as pq

        class _Stall(Exception):
            pass

        def _on_alarm(signum, frame):
            raise _Stall("row-group read stalled")

        signal.signal(signal.SIGALRM, _on_alarm)

        def _need():
            return any(deficits[("commfor_small", s)] > 0
                       for s in ("train", "val"))

        def _process_rg(t, fid, rg):
            batch = []
            for i in range(t.num_rows):
                if t["nsfw_flag"][i].as_py() or \
                        str(t["label"][i].as_py()) != "1":
                    continue
                raw = t["image_data"][i].as_py()
                try:
                    im = Image.open(io.BytesIO(raw)); im.load()
                except Exception:  # noqa: BLE001
                    continue
                batch.append((f"{fid}/rg{rg}/{i}", raw, im,
                              t["model_name"][i].as_py() or ""))
            for (rel, rw, _, model), sc in zip(
                    batch, score_fn([b[2] for b in batch])):
                if sc < threshold or util.sha256_bytes(rw) in existing_sha:
                    continue
                split = _assign_backfill_split(
                    deficits, "commfor_small", forced=model_split(model))
                if split:
                    ingest(rw, "commfor_backfill", rel, "commfor_small",
                           model, "commfor", split, sc)

        for fid in COMMFOR_BACKFILL_FILES:
            if not _need():
                break
            path = (f"datasets/OwensLab/CommunityForensics-Small/data/"
                    f"HFCF_small_{fid}.parquet")
            for attempt in (1, 2, 3):
                try:
                    local = (config.DATA / ".commfor_raw" / "data" /
                             f"HFCF_small_{fid}.parquet")
                    if local.exists():
                        print(f"commfor backfill: local {local} attempt {attempt}",
                              flush=True)
                        src = open(local, "rb")
                    else:
                        print(f"commfor backfill: reading {path} attempt {attempt}",
                              flush=True)
                        fs = HfFileSystem()  # fresh connection per attempt
                        src = fs.open(path, "rb")
                    with src as f:
                        pf = pq.ParquetFile(f)
                        for rg in range(pf.metadata.num_row_groups):
                            if not _need():
                                break
                            signal.alarm(600)  # dead-connection watchdog
                            try:
                                t = pf.read_row_group(rg, columns=[
                                    "image_data", "model_name", "nsfw_flag",
                                    "label"])
                            finally:
                                signal.alarm(0)
                            # ingest dedupes by sha256, so a retried row group
                            # never double-adds
                            _process_rg(t, fid, rg)
                    break
                except (_Stall, OSError) as e:
                    print(f"commfor backfill: attempt {attempt} on {fid} "
                          f"failed: {repr(e)[:100]}", flush=True)
                    _time.sleep(30 * attempt)
    pw.close()
    print("after commfor backfill, deficits:", deficits,
          f"| backfilled rows: {len(new_rows)}", flush=True)

    # --- write updated manifest + audit -------------------------------------
    out_fields = fields + [c for c in ("style", "realism_score")
                           if c not in fields]
    man.extend(new_rows)
    with open(CURATED / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(man)

    notes = f"""
## Addendum: realism filter (2026-08-31)

- **Mechanism**: open_clip ViT-B-32 (laion2b_s34b_b79k) zero-shot,
  realism_score = softmax(100*[sim(photo prompts), sim(stylized prompts)])[photo].
  Photo prompts: {PHOTO_PROMPTS}. Stylized prompts: {STYLIZED_PROMPTS}.
- **Threshold**: {threshold}. Applied to commfor_small fakes and wildfake only.
- **Policy**: tag, don't delete. New manifest columns style
  (realistic/stylized, blank = not scored) and realism_score. Stylized rows
  formerly in train/val moved to split=eval_stylized ({tagged['wildfake']}
  wildfake + {tagged['commfor_small']} commfor rows). Files remain on disk.
- **Backfill**: {len(new_rows)} replacement fakes ingested from unsampled
  remainders (wildfake part_1.zip; CommunityForensics-Small parquet files
  {COMMFOR_BACKFILL_FILES}, provenance recorded as file/rowgroup/row in
  orig_relpath) to restore the original train/val targets. commfor backfill
  keeps the model_name-disjoint val (new models: seeded 15% draw).
  Remaining deficits if any: { {k: v for k, v in deficits.items() if v > 0} }.
- **Sanity sheet**: audit/realism_borderline.png (kept-just-above vs
  tagged-just-below the threshold).
- commfor reals were NOT filtered (they are camera photos by construction);
  sid_set/genimagepp/df40 untouched per scope decision.
"""
    with open(CURATED / "audit_report.md", "a") as f:
        f.write(notes)
    print("APPLY_DONE", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["score", "sheet", "apply"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    if args.stage == "score":
        stage_score(args.batch_size)
    elif args.stage == "sheet":
        stage_sheet(args.threshold)
    else:
        stage_apply(args.threshold, args.batch_size)


if __name__ == "__main__":
    main()
