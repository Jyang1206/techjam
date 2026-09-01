"""Part-manifest writing. Each stage writes parts/<stage>.csv atomically.

Part files carry the final manifest schema plus fmt/width/height columns used
by the audit; finalize strips the extras. Dropped (verification-failed) images
go to parts/<stage>_dropped.csv.
"""
import csv
import os

from . import config, util

SCHEMA = [
    "file_path", "sha256", "source_dataset", "generator_or_method", "domain",
    "label", "category", "split", "orig_relpath",
]
PART_SCHEMA = SCHEMA + ["fmt", "width", "height"]


class PartWriter:
    def __init__(self, stage: str):
        self.stage = stage
        config.PARTS.mkdir(parents=True, exist_ok=True)
        config.IMAGES.mkdir(parents=True, exist_ok=True)
        self.final_path = config.PARTS / f"{stage}.csv"
        self.tmp_path = config.PARTS / f"{stage}.csv.tmp"
        self.drop_path = config.PARTS / f"{stage}_dropped.csv"
        self._f = open(self.tmp_path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(PART_SCHEMA)
        self._df = open(self.drop_path, "w", newline="")
        self._dw = csv.writer(self._df)
        self._dw.writerow(["source_key", "orig_relpath", "error"])
        self.written = 0
        self.dropped = 0

    def ingest(self, raw: bytes, *, source_key: str, orig_relpath: str,
               source_dataset: str, generator: str, domain: str, label: int,
               category: str, split: str) -> bool:
        """Verify bytes with Pillow, write the image file, append a row.

        Returns False (and logs) when Pillow rejects the image.
        """
        ok, fmt, w, h, err = util.verify_image(raw)
        if not ok:
            self._dw.writerow([source_key, orig_relpath, err])
            self.dropped += 1
            return False
        name = util.dest_name(source_key, orig_relpath)
        rel = f"images/{source_dataset}/{name}"
        dest = config.CURATED / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():  # idempotent on rerun
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            with open(tmp, "wb") as f:
                f.write(raw)
            os.replace(tmp, dest)
        self._w.writerow([
            rel, util.sha256_bytes(raw), source_dataset, generator, domain,
            label, category, split, orig_relpath, fmt, w, h,
        ])
        self.written += 1
        return True

    def close(self) -> None:
        self._f.close()
        self._df.close()
        os.replace(self.tmp_path, self.final_path)


def read_part(stage: str):
    path = config.PARTS / f"{stage}.csv"
    with open(path, newline="") as f:
        yield from csv.DictReader(f)
