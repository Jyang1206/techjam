import sys, os, io, csv
from datasets import load_dataset
from PIL import Image as PILImage

ds_id, split, out_dir, cap_per_label = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
ds = load_dataset(ds_id, split=split, streaming=True)
os.makedirs(out_dir, exist_ok=True)
meta = csv.writer(open(os.path.join(out_dir, "meta.csv"), "w", newline=""))
meta.writerow(["file", "label", "model_name", "architecture", "real_source"])

counts = {}
for i, ex in enumerate(ds):
    if ex.get("nsfw_flag"): continue
    lbl = str(ex.get("label", "unlabeled"))
    if counts.get(lbl, 0) >= cap_per_label:
        if len(counts) >= 2 and all(c >= cap_per_label for c in counts.values()): break
        continue
    raw = ex.get("image_data")
    if raw is None:
        img_obj = ex.get("image")
        if img_obj is None: continue
        buf = io.BytesIO(); img_obj.save(buf, format=img_obj.format or "PNG"); raw = buf.getvalue()
    try:
        img = PILImage.open(io.BytesIO(raw)); img.load()
    except Exception: continue
    d = os.path.join(out_dir, lbl); os.makedirs(d, exist_ok=True)
    ext = (img.format or "png").lower(); ext = "jpg" if ext == "jpeg" else ext
    fname = f"{i:07d}.{ext}"
    with open(os.path.join(d, fname), "wb") as f: f.write(raw)
    meta.writerow([f"{lbl}/{fname}", lbl, ex.get("model_name",""), ex.get("architecture",""), ex.get("real_source","")])
    counts[lbl] = counts.get(lbl, 0) + 1
    if sum(counts.values()) % 500 == 0: print(i, counts, flush=True)
print("DONE", counts)
