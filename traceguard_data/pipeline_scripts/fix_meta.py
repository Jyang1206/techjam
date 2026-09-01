import os, csv
out_dir = os.path.expanduser("~/data/hf/commfor_small")
ondisk = set(os.listdir(os.path.join(out_dir, "1")))
with open(os.path.join(out_dir, "meta.csv")) as f:
    rows = list(csv.reader(f))
header, body = rows[0], rows[1:]
inmeta = {r[0].split("/", 1)[1] for r in body if r[0].startswith("1/")}
missing = sorted(ondisk - inmeta)
print("missing:", len(missing))
if missing:
    bounds = [("0", 0, 2993), ("1", 2993, 5986), ("10", 5986, 8979)]
    need = {}
    for fn in missing:
        i = int(fn.split(".")[0])
        for name, lo, hi in bounds:
            if lo <= i < hi:
                need.setdefault(name, []).append((i, i - lo, fn)); break
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq
    fs = HfFileSystem()
    added = []
    for name, items in need.items():
        p = f"datasets/OwensLab/CommunityForensics-Small/data/HFCF_small_{name}.parquet"
        with fs.open(p, "rb") as f:
            t = pq.read_table(f, columns=["model_name", "architecture", "real_source", "label", "nsfw_flag"])
        for i, off, fn in items:
            assert str(t["label"][off].as_py()) == "1", (i, off)
            assert not t["nsfw_flag"][off].as_py(), (i, off)
            added.append([f"1/{fn}", "1", t["model_name"][off].as_py() or "", t["architecture"][off].as_py() or "", t["real_source"][off].as_py() or ""])
    body.extend(added)
    print("reconstructed:", len(added))
with open(os.path.join(out_dir, "meta_real.csv")) as f:
    real_rows = list(csv.reader(f))
body.extend(real_rows)
with open(os.path.join(out_dir, "meta.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(header); w.writerows(body)
n1 = sum(1 for r in body if r[1] == "1"); n0 = sum(1 for r in body if r[1] == "0")
print("final meta rows:", len(body), "label1:", n1, "label0:", n0)
assert n1 == 8000 and n0 == 8000 and len(set(r[0] for r in body)) == len(body)
os.remove(os.path.join(out_dir, "meta_real.csv"))
print("META_FIXED")
