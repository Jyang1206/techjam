"""Train detection heads on precomputed embeddings.

8-run ablation per category: {logreg, mlp} x {dino, dino+fft} x
{canonical-only, all-views} training. Winner picked by val ROC-AUC on the
canonical val view; decision threshold fitted on val (max balanced accuracy).

If the shards contain no degraded train views yet (phase-1 extraction), the
all-views regime is identical to canonical-only and is skipped with a note —
rerun after the phase-2 shards land to fill those rows in.

    python3 -m traceguard_data.heads --embeddings ~/data/embeddings/face \
        --out ~/data/results --category face --device cuda
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SEED = 42
EMB_DIM = 1024
FFT_DIM = 70
CANON = "canon_jpeg95"


def load_embeddings(emb_dir: Path, category=None):
    embs, ffts, meta = [], [], {k: [] for k in
                               ("sha256", "view_id", "view_name", "label",
                                "category", "split", "generator_or_method",
                                "source_dataset")}
    files = sorted(emb_dir.rglob("shard_*.parquet"))
    if not files:
        raise SystemExit(f"no shards under {emb_dir}")
    for f in files:
        t = pq.read_table(f)
        if category is not None:
            mask = np.array(t["category"].to_pylist()) == category
            if not mask.any():
                continue
            t = t.filter(pa.array(mask))
        embs.append(t["embedding"].combine_chunks().values.to_numpy(
            zero_copy_only=False).reshape(-1, EMB_DIM).astype(np.float32))
        ffts.append(t["fft_features"].combine_chunks().values.to_numpy(
            zero_copy_only=False).reshape(-1, FFT_DIM).astype(np.float32))
        for k in meta:
            meta[k].extend(t[k].to_pylist())
    out = {k: np.array(v) for k, v in meta.items()}
    out["emb"] = np.concatenate(embs)
    out["fft"] = np.concatenate(ffts)
    out["label"] = out["label"].astype(int)
    print(f"loaded {len(out['label'])} rows from {len(files)} shards")
    return out


def features(d, idx, feature_set):
    x = d["emb"][idx]
    if feature_set == "dino+fft":
        x = np.concatenate([x, d["fft"][idx]], axis=1)
    return x


class Scaler:
    def fit(self, x):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-6
        return self

    def transform(self, x):
        return (x - self.mean) / self.std


def roc_auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def train_logreg(xtr, ytr):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    m.fit(xtr, ytr)
    return m, lambda x: m.predict_proba(x)[:, 1]


def train_mlp(xtr, ytr, xva, yva, device):
    import torch
    from torch import nn
    torch.manual_seed(SEED)
    model = nn.Sequential(
        nn.Linear(xtr.shape[1], 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    xt = torch.from_numpy(xtr).to(device)
    yt = torch.from_numpy(ytr.astype(np.float32)).to(device)
    xv = torch.from_numpy(xva).to(device)
    best_auc, best_state, patience = -1.0, None, 0
    n = len(xt)
    for epoch in range(100):
        model.train()
        perm = torch.randperm(n, device=device)
        for lo in range(0, n, 512):
            i = perm[lo:lo + 512]
            opt.zero_grad()
            loss = lossf(model(xt[i]).squeeze(1), yt[i])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            sv = torch.sigmoid(model(xv).squeeze(1)).cpu().numpy()
        auc = roc_auc(yva, sv)
        if auc > best_auc + 1e-4:
            best_auc, patience = auc, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 5:
                break
    model.load_state_dict(best_state)
    model.eval()

    def score(x):
        with torch.no_grad():
            xs = torch.from_numpy(x).to(device)
            return torch.sigmoid(model(xs).squeeze(1)).cpu().numpy()
    return model, score


def fit_threshold(y, s):
    """Threshold maximizing balanced accuracy on val."""
    best_t, best_ba = 0.5, -1.0
    for t in np.unique(np.round(s, 3)):
        pred = s >= t
        tpr = (pred & (y == 1)).sum() / max((y == 1).sum(), 1)
        tnr = (~pred & (y == 0)).sum() / max((y == 0).sum(), 1)
        ba = (tpr + tnr) / 2
        if ba > best_ba:
            best_ba, best_t = ba, float(t)
    return best_t, float(best_ba)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--out", default=str(Path.home() / "data/results"))
    ap.add_argument("--category", required=True, choices=["face", "scene"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d = load_embeddings(Path(args.embeddings), args.category)
    is_tr = d["split"] == "train"
    is_va = d["split"] == "val"
    va_canon = is_va & (d["view_name"] == CANON)
    train_views = sorted(set(d["view_name"][is_tr]))
    have_degraded = len(train_views) > 1
    print(f"train views present: {train_views}")

    regimes = [("canonical", is_tr & (d["view_name"] == CANON))]
    if have_degraded:
        regimes.append(("all_views", is_tr))
    results, models = [], {}
    for regime, tr_mask in regimes:
        tr_idx = np.where(tr_mask)[0]
        va_idx = np.where(va_canon)[0]
        for feature_set in ("dino", "dino+fft"):
            xtr_raw = features(d, tr_idx, feature_set)
            sc = Scaler().fit(xtr_raw)
            xtr = sc.transform(xtr_raw)
            xva = sc.transform(features(d, va_idx, feature_set))
            ytr, yva = d["label"][tr_idx], d["label"][va_idx]
            for mtype in ("logreg", "mlp"):
                t0 = time.time()
                if mtype == "logreg":
                    model, score = train_logreg(xtr, ytr)
                else:
                    model, score = train_mlp(xtr, ytr, xva, yva, args.device)
                sva = score(xva)
                auc = roc_auc(yva, sva)
                key = (mtype, feature_set, regime)
                results.append({"model": mtype, "features": feature_set,
                                "train_views": regime, "val_auc": auc,
                                "train_rows": len(tr_idx),
                                "secs": round(time.time() - t0, 1)})
                models[key] = (model, score, sc, sva, yva)
                print(f"  {key}: val AUC {auc:.4f} "
                      f"({len(tr_idx)} rows, {time.time()-t0:.0f}s)", flush=True)

    best = max(results, key=lambda r: r["val_auc"])
    bkey = (best["model"], best["features"], best["train_views"])
    model, score, sc, sva, yva = models[bkey]
    thr, ba = fit_threshold(yva, sva)
    print(f"WINNER {args.category}: {bkey} val_auc={best['val_auc']:.4f} "
          f"threshold={thr:.3f} val_bal_acc={ba:.4f}", flush=True)

    payload = {"category": args.category, "model_type": best["model"],
               "features": best["features"], "train_views": best["train_views"],
               "scaler_mean": sc.mean, "scaler_std": sc.std,
               "threshold": thr, "val_auc": best["val_auc"],
               "val_balanced_acc": ba, "have_degraded_train_views": have_degraded}
    if best["model"] == "logreg":
        payload["sk_model"] = model
    else:
        payload["mlp_state"] = {k: v.detach().cpu().numpy() for k, v in
                                model.state_dict().items()}
        payload["mlp_hidden"] = 256
        payload["in_dim"] = int(sc.mean.shape[0])
    with open(out / f"{args.category}_head.pkl", "wb") as f:
        pickle.dump(payload, f)

    lines = [f"# Ablation — {args.category}", "",
             "| model | features | train views | val ROC-AUC | train rows | secs |",
             "|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: -r["val_auc"]):
        mark = " **<- winner**" if (r["model"], r["features"],
                                    r["train_views"]) == bkey else ""
        lines.append(f"| {r['model']} | {r['features']} | {r['train_views']} | "
                     f"{r['val_auc']:.4f}{mark} | {r['train_rows']} | {r['secs']} |")
    if not have_degraded:
        lines += ["", "_all-views regime skipped: shards contain no degraded "
                  "train views yet (phase-1 extraction); rerun after phase 2._"]
    lines += ["", f"Winner threshold (max balanced acc on val): **{thr:.3f}** "
              f"(val balanced acc {ba:.4f})", ""]
    (out / f"ablation_{args.category}.md").write_text("\n".join(lines))
    with open(out / f"ablation_{args.category}.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"HEADS_{args.category}_DONE", flush=True)


if __name__ == "__main__":
    main()
