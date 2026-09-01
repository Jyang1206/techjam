"""Evaluate the winning head per category and emit all deliverables.

Negative sets for AUC (stated in every table):
  face:  eval reals = eval_face_cdf (reals-only slice; also reported as FPR).
  scene: NO eval-split reals exist in the curated set -> val reals are the
         negative reference (noted; slightly optimistic vs a true holdout).
For degraded views, negatives use the same view where extracted, else the
canonical view (phase-1 shards carry only orig+canonical for eval rows).

    python3 -m traceguard_data.eval --embeddings ~/data/embeddings/face \
        --head ~/data/results/face_head.pkl --out ~/data/results \
        --manifest ~/data/curated/manifest.csv --category face
    python3 -m traceguard_data.eval --combine --out ~/data/results
"""
import argparse
import csv
import pickle
import shutil
from pathlib import Path

import numpy as np

from .heads import CANON, Scaler, load_embeddings, roc_auc

GRID_FAMILIES = {
    "jpeg": ["jpeg90", "jpeg70", "jpeg50", "jpeg30"],
    "blur": ["blur0.5", "blur1.0", "blur2.0"],
    "resize": ["resize0.5", "resize0.25"],
    "noise": ["noise0.02", "noise0.05", "noise0.10"],
    "color": ["color+20", "color-20", "bright+20", "bright-20",
              "contrast+20", "contrast-20"],
    "crop": ["crop80"],
}
CDF_NOTE = "eval_face_cdf is reals-only: FPR under domain shift, no AUC."


def load_head(path):
    with open(path, "rb") as f:
        p = pickle.load(f)
    sc = Scaler()
    sc.mean, sc.std = p["scaler_mean"], p["scaler_std"]
    if p["model_type"] == "logreg":
        m = p["sk_model"]
        score = lambda x: m.predict_proba(x)[:, 1]  # noqa: E731
    else:
        import torch
        from torch import nn
        m = nn.Sequential(nn.Linear(p["in_dim"], p["mlp_hidden"]), nn.ReLU(),
                          nn.Dropout(0.3), nn.Linear(p["mlp_hidden"], 1))
        m.load_state_dict({k: torch.from_numpy(v)
                           for k, v in p["mlp_state"].items()})
        m.eval()

        def score(x):
            import torch as _t
            with _t.no_grad():
                return _t.sigmoid(m(_t.from_numpy(x)).squeeze(1)).numpy()
    return p, sc, score


def featurize(d, idx, p, sc):
    x = d["emb"][idx]
    if p["features"] == "dino+fft":
        x = np.concatenate([x, d["fft"][idx]], axis=1)
    return sc.transform(x)


def metrics_at(y, s, thr):
    pred = s >= thr
    out = {}
    pos, neg = (y == 1), (y == 0)
    if pos.any() and neg.any():
        out["auc"] = roc_auc(y, s)
        out["bal_acc"] = ((pred[pos].mean() + (~pred[neg]).mean()) / 2)
    if neg.any():
        out["fpr"] = float(pred[neg].mean())
    if pos.any():
        out["fnr"] = float((~pred[pos]).mean())
        out["tpr"] = float(pred[pos].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings")
    ap.add_argument("--head")
    ap.add_argument("--manifest", default=str(Path.home() / "data/curated/manifest.csv"))
    ap.add_argument("--curated-root", default=str(Path.home() / "data/curated"))
    ap.add_argument("--out", default=str(Path.home() / "data/results"))
    ap.add_argument("--category", choices=["face", "scene"])
    ap.add_argument("--combine", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.combine:
        return combine(out)

    cat = args.category
    p, sc, score_fn = load_head(args.head)
    d = load_embeddings(Path(args.embeddings), cat)
    thr = p["threshold"]
    all_scores = score_fn(featurize(d, np.arange(len(d["label"])), p, sc))

    def sel(**kw):
        m = np.ones(len(d["label"]), bool)
        for k, v in kw.items():
            m &= (np.isin(d[k], v) if isinstance(v, (list, tuple))
                  else (d[k] == v))
        return m

    eval_slices = sorted(s for s in set(d["split"]) if s.startswith("eval_"))
    real_ref_split = "eval_face_cdf" if cat == "face" else "val"
    lines = [f"# Evaluation — {cat}", "",
             f"Winner: {p['model_type']} / {p['features']} / "
             f"{p['train_views']} (val AUC {p['val_auc']:.4f}, "
             f"threshold {thr:.3f})", "",
             f"_Negative (real) reference for AUC: {real_ref_split}"
             + (" — NO scene eval reals exist; val reals used (noted "
                "optimism)_" if cat == "scene" else "_"), "",
             f"_{CDF_NOTE}_", ""]

    # 1. per slice x view metrics ------------------------------------------
    lines += ["## Per-slice metrics (at fitted threshold)", "",
              "| slice | view | n | AUC | bal_acc | FPR | FNR |",
              "|---|---|---|---|---|---|---|"]
    for sl in eval_slices:
        for vn in sorted(set(d["view_name"][sel(split=sl)])):
            m = sel(split=sl, view_name=vn)
            y, s = d["label"][m], all_scores[m]
            if sl != real_ref_split and not (y == 0).any():
                ref = sel(split=real_ref_split, view_name=vn, label=0)
                if not ref.any():
                    ref = sel(split=real_ref_split, view_name=CANON, label=0)
                y = np.concatenate([y, d["label"][ref] * 0])
                s = np.concatenate([s, all_scores[ref]])
            mt = metrics_at(y, s, thr)
            lines.append(
                f"| {sl} | {vn} | {int(m.sum())} | "
                f"{mt.get('auc', float('nan')):.4f} | "
                f"{mt.get('bal_acc', float('nan')):.4f} | "
                f"{mt.get('fpr', float('nan')):.4f} | "
                f"{mt.get('fnr', float('nan')):.4f} |")
    lines.append("")

    # 2./3. robustness table + degradation curves ---------------------------
    fake_slices = [s for s in eval_slices
                   if (d["label"][sel(split=s)] == 1).any()]
    view_rows, curve_data = [], {}
    eval_views = sorted(set(d["view_name"][np.isin(d["split"], eval_slices)]))
    for vn in ["orig", CANON] + [v for f in GRID_FAMILIES.values() for v in f]:
        if vn not in eval_views:
            continue
        fk = sel(split=fake_slices, view_name=vn)
        ref = sel(split=real_ref_split, view_name=vn, label=0)
        if not ref.any():
            ref = sel(split=real_ref_split, view_name=CANON, label=0)
        if not fk.any():
            continue
        y = np.concatenate([np.ones(int(fk.sum())), np.zeros(int(ref.sum()))])
        s = np.concatenate([all_scores[fk], all_scores[ref]])
        auc = roc_auc(y, s)
        tpr = float((all_scores[fk] >= thr).mean())
        view_rows.append((vn, int(fk.sum()), auc, tpr))
        curve_data[vn] = auc
    val_m = sel(split="val", view_name=CANON)
    val_auc = roc_auc(d["label"][val_m], all_scores[val_m])
    lines += ["## Robustness by view (eval fakes vs "
              f"{real_ref_split} reals)", "",
              f"val (canonical) reference AUC: **{val_auc:.4f}**", "",
              "| view | n fakes | AUC | TPR@thr |", "|---|---|---|---|"]
    for vn, n, auc, tpr in view_rows:
        lines.append(f"| {vn} | {n} | {auc:.4f} | {tpr:.4f} |")
    if len(view_rows) <= 2:
        lines += ["", "_Only clean/canonical eval views extracted so far "
                  "(phase 1); grid rows appear after the phase-2 shards._"]
    lines.append("")
    np.save(out / f"robustness_{cat}.npy",
            np.array([(vn, auc) for vn, _, auc, _ in view_rows], dtype=object),
            allow_pickle=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for fam, views in GRID_FAMILIES.items():
            xs = [v for v in views if v in curve_data]
            if not xs:
                continue
            ax.plot(range(len(xs)), [curve_data[v] for v in xs],
                    marker="o", label=fam)
            for i, v in enumerate(xs):
                ax.annotate(v, (i, curve_data[v]), fontsize=6)
        base = curve_data.get(CANON) or curve_data.get("orig")
        if base:
            ax.axhline(base, ls="--", c="gray", label="canonical")
        ax.set_ylabel("AUC")
        ax.set_xlabel("severity rank")
        ax.set_title(f"degradation curves — {cat}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"degradation_{cat}.png", dpi=150)
        plt.close(fig)
        # score distribution
        fk_all = sel(split=fake_slices, view_name=CANON)
        rf_all = sel(split=real_ref_split, view_name=CANON, label=0)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(all_scores[fk_all], bins=50, alpha=0.6,
                label="fake (eval)", density=True)
        ax.hist(all_scores[rf_all], bins=50, alpha=0.6,
                label=f"real ({real_ref_split})", density=True)
        ax.axvline(thr, c="k", ls="--", label=f"thr={thr:.2f}")
        ax.legend(fontsize=8)
        ax.set_title(f"score distribution — {cat}")
        fig.tight_layout()
        fig.savefig(out / f"scores_{cat}.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        lines.append(f"_plots skipped: {e!r}_")

    # 4. FP/FN gallery -------------------------------------------------------
    sha2path = {}
    with open(args.manifest, newline="") as f:
        for r in csv.DictReader(f):
            sha2path.setdefault(r["sha256"], (r["file_path"],
                                              r["generator_or_method"],
                                              r["source_dataset"]))
    gal = out / "error_gallery" / cat
    for kind, mask, order in (
            ("fp", sel(split=real_ref_split, view_name=CANON, label=0), -1),
            ("fn", sel(split=fake_slices, view_name=CANON), +1)):
        idxs = np.where(mask)[0]
        sc_ = all_scores[idxs]
        wrong = idxs[sc_ >= thr] if kind == "fp" else idxs[sc_ < thr]
        wrong = wrong[np.argsort(order * all_scores[wrong])][:15]
        kdir = gal / kind
        kdir.mkdir(parents=True, exist_ok=True)
        with open(gal / f"{kind}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sha256", "score", "generator", "source"])
            for i in wrong:
                sha = d["sha256"][i]
                fp, gen, src = sha2path.get(sha, ("", "", ""))
                w.writerow([sha, f"{all_scores[i]:.4f}", gen, src])
                srcp = Path(args.curated_root) / fp
                if fp and srcp.exists():
                    shutil.copy(srcp, kdir / Path(fp).name)
    lines += ["## Error gallery", "",
              f"error_gallery/{cat}/fp (reals from {real_ref_split} scored "
              "fake) and /fn (eval fakes scored real), top-15 by confidence, "
              "canonical view; CSVs alongside.", ""]

    (out / f"results_{cat}.md").write_text("\n".join(lines))
    print(f"EVAL_{cat}_DONE", flush=True)


def combine(out: Path):
    parts = []
    for cat in ("face", "scene"):
        for name in (f"ablation_{cat}.md", f"results_{cat}.md"):
            f = out / name
            if f.exists():
                parts.append(f.read_text())
    summary = ("# TraceGuard — results summary\n\n"
               f"_{CDF_NOTE}_\n\n" + "\n\n---\n\n".join(parts))
    (out / "results_summary.md").write_text(summary)
    rob = ["# Robustness table (AUC per view)", "", f"_{CDF_NOTE}_", ""]
    for cat in ("face", "scene"):
        f = out / f"robustness_{cat}.npy"
        if f.exists():
            rows = np.load(f, allow_pickle=True)
            rob += [f"## {cat}", "", "| view | AUC |", "|---|---|"]
            rob += [f"| {vn} | {auc:.4f} |" for vn, auc in rows]
            rob.append("")
    (out / "robustness_table.md").write_text("\n".join(rob))
    print("COMBINE_DONE", flush=True)


if __name__ == "__main__":
    main()
