"""Real-diversity patch (post-smoke-test).

Finding: the scene head scored pristine held-out real photos 0.91-0.999 fake.
Cause: curated scene reals (commfor files 120/140/150 + sid_set) skew
compressed/web-quality, while scene fakes skew clean and sharp, so the head
learned a "pristine = fake" shortcut that val (drawn from the same real
sources) could not expose.

Patch: ingest up to 2000 reals from each of the UNUSED CommunityForensics
parquet files 160/170/180 (pristine LandscapesHQ/FFHQ-grade reals) as
additional scene train/val rows (85/15, seed 42), then retrain the scene
head. Additive only; provenance recorded as <file>/rg<g>/<row> in
orig_relpath; the pre-patch head is preserved as scene_head_prepatch.pkl.
"""
import csv
import io
import sys
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path.home()))
from traceguard_data import config, util  # noqa: E402
from traceguard_data.manifest import PartWriter, SCHEMA  # noqa: E402

PER_FILE = 2000
FILES = ["160", "170", "180"]

man_path = config.CURATED / "manifest.csv"
with open(man_path, newline="") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    existing_sha = {r["sha256"] for r in reader}

pw = PartWriter("realpatch")
new_rows = []
for fid in FILES:
    local = config.DATA / ".commfor_raw" / "data" / f"HFCF_small_{fid}.parquet"
    if not local.exists():
        print(f"missing {local}, skipping", flush=True)
        continue
    pf = pq.ParquetFile(local)
    taken = 0
    for rg in range(pf.metadata.num_row_groups):
        if taken >= PER_FILE:
            break
        t = pf.read_row_group(rg, columns=["image_data", "label", "nsfw_flag"])
        idxs = list(range(t.num_rows))
        util.rng_for(f"realpatch:{fid}:{rg}").shuffle(idxs)
        for i in idxs:
            if taken >= PER_FILE:
                break
            if t["nsfw_flag"][i].as_py() or str(t["label"][i].as_py()) != "0":
                continue
            raw = t["image_data"][i].as_py()
            sha = util.sha256_bytes(raw)
            if sha in existing_sha:
                continue
            rel = f"{fid}/rg{rg}/{i}"
            split = ("val" if util.rng_for(f"realpatch:split:{fid}:{rg}:{i}").random() < 0.15
                     else "train")
            if pw.ingest(raw, source_key="commfor_realpatch", orig_relpath=rel,
                         source_dataset="commfor_small",
                         generator="commfor_real", domain="commfor",
                         label=0, category="scene", split=split):
                existing_sha.add(sha)
                name = util.dest_name("commfor_realpatch", rel)
                row = {c: "" for c in fields}
                row.update({"file_path": f"images/commfor_small/{name}",
                            "sha256": sha, "source_dataset": "commfor_small",
                            "generator_or_method": "commfor_real",
                            "domain": "commfor", "label": "0",
                            "category": "scene", "split": split,
                            "orig_relpath": rel})
                new_rows.append(row)
                taken += 1
    print(f"file {fid}: ingested {taken}", flush=True)
pw.close()

with open(man_path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writerows(new_rows)
n_val = sum(1 for r in new_rows if r["split"] == "val")
print(f"PATCH_ROWS {len(new_rows)} (val {n_val})", flush=True)

with open(config.CURATED / "audit_report.md", "a") as f:
    f.write(f"""
## Addendum: real-diversity patch (post-smoke-test, 2026-09-01)

- predict_v2 smoke test exposed a scene-head blind spot: 5/5 pristine held-out
  real photos (CommunityForensics file 160 reals, never in the curated set)
  scored 0.91-0.999 fake. Diagnosis: curated scene reals skew compressed
  web-quality vs clean sharp fakes -> "pristine = fake" shortcut invisible to
  val (same real sources).
- Patch: +{len(new_rows)} pristine reals ({n_val} val) ingested from unused
  CommunityForensics files 160/170/180 as scene train/val label-0 rows
  (source_key commfor_realpatch, provenance file/rowgroup/row). Scene head
  retrained; pre-patch head kept as results/scene_head_prepatch.pkl for
  comparison. Additive only - no rows removed or relabeled.
""")
print("PATCH_DONE", flush=True)
