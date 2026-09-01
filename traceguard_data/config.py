"""Central configuration for the TraceGuard curation pipeline.

All sampling is deterministic: every random decision derives from SEED plus a
stable string context (method name, stage name), so reruns produce identical
output.
"""
from pathlib import Path

SEED = 42

DATA = Path.home() / "data"
DF40 = DATA / "df40"
DF40_TRAIN_ZIPS = DF40 / "DF40_train"
DF40_JSON = DF40 / "dataset_json"
WILDFAKE_ZIP = DATA / "wildfake/Images/Diffusion_based/Midjourney/Typical/part_1.zip"
SID_SET = DATA / "hf/sid_set"
COMMFOR = DATA / "hf/commfor_small"
GENIMAGEPP = DATA / "hf/genimagepp"

CURATED = DATA / "curated"
IMAGES = CURATED / "images"
WORK = DATA / ".curation_work"   # part manifests + drop logs; NOT synced to S3
PARTS = WORK / "parts"

MIN_FREE_GB = 40

# --- Face (DF40) ------------------------------------------------------------
# kind: "swap" zips look like  <top>/frames/<pair>/<frame>.png
#       "gen"  zips look like  <top>/<video>/<frame>.png
# json: name of dataset_json file driving train sampling, or None to sample
#       directly from the zip namelist (restricted to FF++ train video ids).
# ziptop: top-level dir inside the zip ("" when members have no method prefix).
TRAIN_METHODS = {
    "e4s":       dict(zip="e4s.zip",       kind="swap", json="e4s_ff.json",      ziptop="e4s"),
    "inswap":    dict(zip="inswap.zip",    kind="swap", json="inswap_ff.json",   ziptop="inswap"),
    "blendface": dict(zip="blendface.zip", kind="swap", json="blendface_ff.json", ziptop="blendface"),
    "faceswap":  dict(zip="faceswap.zip",  kind="swap", json="faceswap_ff.json", ziptop="faceswap"),
    "fsgan":     dict(zip="fsgan.zip",     kind="swap", json="fsgan_ff.json",    ziptop="fsgan"),
    "simswap":   dict(zip="simswap.zip",   kind="swap", json=None,               ziptop="simswap"),
    "uniface":   dict(zip="uniface.zip",   kind="swap", json=None,               ziptop=""),
    "DiT":       dict(zip="DiT.zip",       kind="gen",  json="DiT_ff.json",      ziptop="DiT"),
    "pixart":    dict(zip="pixart.zip",    kind="gen",  json="pixart_ff.json",   ziptop="pixart"),
    "rddm":      dict(zip="rddm.zip",      kind="gen",  json="rddm_ff.json",     ziptop="RDDM"),
    "sd2.1":     dict(zip="sd2.1.zip",     kind="gen",  json=None,               ziptop="sd2.1"),
}
# e4e was in the original plan but its zip is not part of the download; dropped
# (user-approved 2026-08-31).

UNSEEN_METHODS = {
    "facedancer":   dict(zip="facedancer.zip",   kind="swap", ziptop="facedancer"),
    "mobileswap":   dict(zip="mobileswap.zip",   kind="swap", ziptop="mobileswap"),
    "SiT":          dict(zip="SiT.zip",          kind="gen",  ziptop="SiT"),
    # widened (user-approved) to compensate for the unavailable cdf fake set:
    "danet":        dict(zip="danet.zip",        kind="swap", ziptop="danet"),
    "fomm":         dict(zip="fomm.zip",         kind="swap", ziptop="fomm"),
    "hyperreenact": dict(zip="hyperreenact.zip", kind="swap", ziptop="hyperreenact"),
}

# JSON whose Real/train keys define the canonical FF++ train video id set.
FFPP_SPLIT_REFERENCE_JSON = "DiT_ff.json"

TRAIN_FRAMES_PER_VIDEO = 3
TRAIN_CAP_PER_METHOD = 1500
UNSEEN_FRAMES_PER_VIDEO = 3
UNSEEN_CAP_PER_METHOD = 500
FACE_VAL_FRAC = 0.15

FFPP_REAL_ZIP = DF40 / "ffpp_real.zip"
CDF_REAL_ZIP = DF40 / "cdf_real.zip"
# 999 videos on disk; 12/video reaches the ~12k target (user-approved).
FFPP_FRAMES_PER_VIDEO = 12
FFPP_TARGET = 12000
CDF_FRAMES_PER_VIDEO = 3
CDF_TARGET = 3000

# --- Scene ------------------------------------------------------------------
WILDFAKE_SAMPLE = 10000
SCENE_VAL_FRAC = 0.15
COMMFOR_VAL_MODEL_FRAC = 0.15

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# Stage names in canonical run order (finalize consumes parts in this order,
# which also fixes dedupe keep-first priority).
STAGES = [
    "df40_fakes",
    "df40_unseen",
    "df40_reals",
    "wildfake",
    "sid_set",
    "commfor",
    "genimagepp",
]

DEVIATIONS = [
    "e4e dropped from train methods: e4e.zip absent from the DF40_train download (JSON only). 11 train methods remain.",
    "eval_face_cdf contains cdf reals only: DF40's cdf-domain fake frames belong to the separate test release, which is not on disk.",
    "eval_face_unseen widened with danet, fomm, hyperreenact (~500 each) to compensate for the missing cdf fakes.",
    "sd2.1, simswap, uniface (train) and SiT (eval) sampled from zip namelists (their dataset_json files are absent); train methods restricted to FF++ train video ids via first video-id token.",
    "ffpp reals sampled at 12 frames/video (999 videos on disk; 3/video could only reach ~3k of the ~12k target).",
    "cdf reals: 888 videos x 3 frames/video = 2664 (< nominal 3000 target).",
    "commfor reals split 85/15 at random (seeded): real rows carry real_source='N/A', so the planned model->real_source linkage does not exist in the data.",
]
