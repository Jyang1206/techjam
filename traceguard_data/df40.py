"""DF40 face pipeline: train/val fakes, held-out eval fakes, and reals.

Zip layouts (verified against the on-disk archives):
  swap zips:  <ziptop>/frames/<pair>/<frame>.png   (uniface has no <ziptop>/)
  gen  zips:  <ziptop>/<video>/<frame>.png
JSON train entries reference the same members under a
deepfakes_detection_datasets/DF40_train/ (swap) or .../DF40/<m>/ff/ (gen)
prefix; both are mapped back to zip member names and validated against the
namelist. Methods without a JSON are sampled from the namelist, keeping only
videos whose first id token is in the FF++ train id set.
"""
import json
import zipfile

from . import config, util
from .manifest import PartWriter


def ffpp_train_ids() -> set:
    d = json.load(open(config.DF40_JSON / config.FFPP_SPLIT_REFERENCE_JSON))
    root = d[next(iter(d))]
    real = next(v for k, v in root.items() if k.endswith("_Real"))
    return set(real["train"].keys())


def _zip_image_members(zf: zipfile.ZipFile):
    return [n for n in zf.namelist()
            if not n.endswith("/") and n.lower().endswith(config.IMG_EXTS)]


def group_namelist(zf: zipfile.ZipFile, kind: str) -> dict:
    """{video_id: [member, ...]} from the zip's own structure."""
    vids = {}
    for m in _zip_image_members(zf):
        parts = m.split("/")
        if kind == "swap":
            if "frames" not in parts:
                continue  # skip landmarks/ etc.
            i = parts.index("frames")
            if len(parts) != i + 3:
                continue
            vid = parts[i + 1]
        else:  # gen: <top>/<video>/<frame>
            if len(parts) != 3:
                continue
            vid = parts[1]
        vids.setdefault(vid, []).append(m)
    return vids


def json_train_frames(method: str, cfg: dict, member_set: set) -> tuple:
    """{video: [member]} from dataset_json train entries, plus miss count."""
    d = json.load(open(config.DF40_JSON / cfg["json"]))
    root = d[next(iter(d))]
    fake = next(v for k, v in root.items() if k.endswith("_Fake"))
    vids, misses = {}, 0
    for vid, entry in fake["train"].items():
        members = []
        for p in entry["frames"]:
            p = p.lstrip("./")
            if cfg["kind"] == "swap":
                marker = "DF40_train/"
                member = p.split(marker, 1)[1] if marker in p else None
                if member is not None and not cfg["ziptop"]:
                    member = member.split("/", 1)[1]
            else:
                marker = "/ff/"
                tail = p.split(marker, 1)[1] if marker in p else None
                member = f"{cfg['ziptop']}/{tail}" if tail else None
            if member and member in member_set:
                members.append(member)
            else:
                misses += 1
        if members:
            vids[vid] = members
    return vids, misses


def _emit(pw: PartWriter, zf: zipfile.ZipFile, zip_name: str, sampled: dict,
          val_vids: set, generator: str, label: int, split_fixed=None):
    for vid in sorted(sampled):
        split = split_fixed or ("val" if vid in val_vids else "train")
        for member in sampled[vid]:
            pw.ingest(
                zf.read(member),
                source_key=f"df40/{zip_name}", orig_relpath=member,
                source_dataset="df40", generator=generator, domain="ff",
                label=label, category="face", split=split,
            )


def stage_df40_fakes() -> None:
    pw = PartWriter("df40_fakes")
    train_ids = ffpp_train_ids()
    for method, cfg in config.TRAIN_METHODS.items():
        zpath = config.DF40_TRAIN_ZIPS / cfg["zip"]
        with zipfile.ZipFile(zpath) as zf:
            if cfg["json"]:
                vid2frames, misses = json_train_frames(
                    method, cfg, set(zf.namelist()))
                if misses:
                    print(f"[df40_fakes] {method}: {misses} JSON frames not in zip", flush=True)
            else:
                allvids = group_namelist(zf, cfg["kind"])
                vid2frames = {v: f for v, f in allvids.items()
                              if v.split("_")[0] in train_ids}
            sampled = util.sample_video_frames(
                vid2frames, config.TRAIN_FRAMES_PER_VIDEO,
                config.TRAIN_CAP_PER_METHOD, f"df40:{method}")
            val_vids = util.split_videos(
                sampled, config.FACE_VAL_FRAC, f"df40:{method}:val")
            _emit(pw, zf, cfg["zip"], sampled, val_vids, method, 1)
            n = sum(len(v) for v in sampled.values())
            print(f"[df40_fakes] {method}: {n} frames from {len(sampled)} videos "
                  f"({len(val_vids)} val videos)", flush=True)
    pw.close()
    print(f"[df40_fakes] total written={pw.written} dropped={pw.dropped}", flush=True)


def stage_df40_unseen() -> None:
    pw = PartWriter("df40_unseen")
    for method, cfg in config.UNSEEN_METHODS.items():
        zpath = config.DF40_TRAIN_ZIPS / cfg["zip"]
        with zipfile.ZipFile(zpath) as zf:
            vid2frames = group_namelist(zf, cfg["kind"])
            sampled = util.sample_video_frames(
                vid2frames, config.UNSEEN_FRAMES_PER_VIDEO,
                config.UNSEEN_CAP_PER_METHOD, f"df40:unseen:{method}")
            _emit(pw, zf, cfg["zip"], sampled, set(), method, 1,
                  split_fixed="eval_face_unseen")
            n = sum(len(v) for v in sampled.values())
            print(f"[df40_unseen] {method}: {n} frames from {len(sampled)} videos", flush=True)
    pw.close()
    print(f"[df40_unseen] total written={pw.written} dropped={pw.dropped}", flush=True)


def _group_real_zip(zf: zipfile.ZipFile) -> dict:
    vids = {}
    for m in _zip_image_members(zf):
        parts = m.split("/")
        vids.setdefault("/".join(parts[:-1]), []).append(m)
    return vids


def stage_df40_reals() -> None:
    pw = PartWriter("df40_reals")
    with zipfile.ZipFile(config.FFPP_REAL_ZIP) as zf:
        sampled = util.sample_video_frames(
            _group_real_zip(zf), config.FFPP_FRAMES_PER_VIDEO,
            config.FFPP_TARGET, "df40:ffpp_real")
        val_vids = util.split_videos(
            sampled, config.FACE_VAL_FRAC, "df40:ffpp_real:val")
        _emit(pw, zf, "ffpp_real.zip", sampled, val_vids, "ffpp_real", 0)
        n = sum(len(v) for v in sampled.values())
        print(f"[df40_reals] ffpp: {n} frames from {len(sampled)} videos "
              f"({len(val_vids)} val videos)", flush=True)
    with zipfile.ZipFile(config.CDF_REAL_ZIP) as zf:
        sampled = util.sample_video_frames(
            _group_real_zip(zf), config.CDF_FRAMES_PER_VIDEO,
            config.CDF_TARGET, "df40:cdf_real")
        for vid in sorted(sampled):
            for member in sampled[vid]:
                pw.ingest(
                    zf.read(member),
                    source_key="df40/cdf_real.zip", orig_relpath=member,
                    source_dataset="df40", generator="cdf_real", domain="cdf",
                    label=0, category="face", split="eval_face_cdf",
                )
        n = sum(len(v) for v in sampled.values())
        print(f"[df40_reals] cdf: {n} frames from {len(sampled)} videos", flush=True)
    pw.close()
    print(f"[df40_reals] total written={pw.written} dropped={pw.dropped}", flush=True)
