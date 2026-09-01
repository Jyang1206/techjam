"""Merge part manifests: dedupe by sha256, write manifest.csv + audit_report.md."""
import csv
from collections import Counter, defaultdict

from . import config
from .manifest import SCHEMA, read_part


def _fmt_table(counter: Counter, headers) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for key, n in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        key = key if isinstance(key, tuple) else (key,)
        lines.append("| " + " | ".join(str(k) for k in key) + f" | {n} |")
    return "\n".join(lines)


def stage_finalize() -> None:
    rows, dups = [], []
    seen = {}
    for stage in config.STAGES:
        for r in read_part(stage):
            first = seen.get(r["sha256"])
            if first is not None:
                dups.append((r["file_path"], r["orig_relpath"], first["file_path"]))
                if r["file_path"] != first["file_path"]:
                    p = config.CURATED / r["file_path"]
                    if p.exists():
                        p.unlink()
                continue
            seen[r["sha256"]] = r
            rows.append(r)

    with open(config.CURATED / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SCHEMA)
        for r in rows:
            w.writerow([r[c] for c in SCHEMA])

    # --- audit ---------------------------------------------------------------
    by_cls = Counter((r["category"], r["label"], r["split"]) for r in rows)
    by_src = Counter((r["source_dataset"], r["label"], r["split"]) for r in rows)
    fmt_by_src_label = defaultdict(Counter)
    res_by_src = defaultdict(Counter)
    for r in rows:
        fmt_by_src_label[(r["source_dataset"], r["label"])][r["fmt"]] += 1
        res_by_src[r["source_dataset"]][f'{r["width"]}x{r["height"]}'] += 1

    warnings = []
    for (src, label), fmts in sorted(fmt_by_src_label.items()):
        total = sum(fmts.values())
        top_fmt, top_n = fmts.most_common(1)[0]
        if total and top_n / total > 0.9:
            warnings.append(
                f"{src} label={label}: {top_n}/{total} ({top_n/total:.0%}) are "
                f"'{top_fmt}' — a detector could key on format instead of content."
            )

    dropped = []
    for stage in config.STAGES:
        with open(config.PARTS / f"{stage}_dropped.csv", newline="") as f:
            for r in csv.DictReader(f):
                dropped.append((stage, r["orig_relpath"], r["error"]))

    L = ["# TraceGuard curated dataset — audit report", ""]
    L += [f"Total curated images: **{len(rows)}** "
          f"(after {len(dups)} sha256 dedupe drops, {len(dropped)} Pillow verification drops)", ""]
    L += ["## Counts by category x label x split", "",
          _fmt_table(by_cls, ["category", "label", "split", "count"]), ""]
    L += ["## Counts by source x label x split", "",
          _fmt_table(by_src, ["source", "label", "split", "count"]), ""]
    L += ["## Format histogram by source x label", "",
          _fmt_table(
              Counter({(s, l, f): n for (s, l), c in fmt_by_src_label.items()
                       for f, n in c.items()}),
              ["source", "label", "format", "count"]), ""]
    L += ["## Resolution histogram by source (top 8 each)", ""]
    for src in sorted(res_by_src):
        top = res_by_src[src].most_common(8)
        rest = sum(res_by_src[src].values()) - sum(n for _, n in top)
        entries = ", ".join(f"{k}: {n}" for k, n in top)
        if rest:
            entries += f", other: {rest}"
        L.append(f"- **{src}**: {entries}")
    L += ["", "## Bias warnings", ""]
    L += [f"- ⚠️ {w}" for w in warnings] or ["- none"]
    L += ["", "## Verification drops", ""]
    L += [f"- [{s}] {p}: {e}" for s, p, e in dropped[:50]] or ["- none"]
    if len(dropped) > 50:
        L.append(f"- … and {len(dropped) - 50} more")
    L += ["", "## sha256 duplicate drops", ""]
    L += [f"- kept {k}, dropped duplicate {d} ({o})" for d, o, k in dups[:50]] or ["- none"]
    if len(dups) > 50:
        L.append(f"- … and {len(dups) - 50} more")
    L += ["", "## Deviations from the original sampling plan (user-approved)", ""]
    L += [f"- {d}" for d in config.DEVIATIONS]
    L.append("")
    (config.CURATED / "audit_report.md").write_text("\n".join(L))

    print(f"[finalize] manifest rows={len(rows)} dups_removed={len(dups)} "
          f"verify_dropped={len(dropped)} warnings={len(warnings)}", flush=True)
    for w in warnings:
        print(f"[finalize] WARNING: {w}", flush=True)
