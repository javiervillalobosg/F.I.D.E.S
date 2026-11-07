"""Extraccion de rasgos forenses de bajo nivel y evaluacion con validacion cruzada agrupada."""
import argparse
import csv
import json
import os
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage, stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

N_BINS = 12
FEATURE_NAMES = ([f"spec_ring{k:02d}" for k in range(N_BINS)] +
                 ["spec_hf_ratio", "res_logstd", "res_kurtosis", "res_skew",
                  "res_blk_mean", "res_blk_std", "res_blk_p90_p10",
                  "res_corr_rg", "res_corr_gb", "jpeg_blockiness"])
HF_IDX = FEATURE_NAMES.index("spec_hf_ratio")
VARIANTS = ["orig", "jpeg90", "jpeg70", "jpeg50", "scale50"]


def spectrum_features(g):
    h, w = g.shape
    s = min(h, w)
    s -= s % 2
    y0, x0 = (h - s) // 2, (w - s) // 2
    c = g[y0:y0 + s, x0:x0 + s]
    c = (c - c.mean()) * np.outer(np.hanning(s), np.hanning(s))
    power = np.abs(np.fft.fftshift(np.fft.fft2(c))) ** 2
    yy, xx = np.indices(power.shape)
    r = np.hypot(yy - s / 2, xx - s / 2) / (s / 2)
    logp = np.log10(power + 1e-8)
    ring = np.minimum((r * N_BINS).astype(int), N_BINS - 1)
    prof = np.array([logp[(r < 1) & (ring == k)].mean() for k in range(N_BINS)])
    prof = prof - prof.mean()
    hf = power[(r > 0.5) & (r < 1)].sum() / power[(r > 0.05) & (r < 1)].sum()
    return list(prof) + [np.log10(hf + 1e-12)]


def residual_features(img):
    g = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    res = (g - ndimage.gaussian_filter(g, 1.0))[4:-4, 4:-4]
    flat = res.ravel()
    bh, bw = res.shape[0] // 32, res.shape[1] // 32
    blk = (res[:bh * 32, :bw * 32].reshape(bh, 32, bw, 32)
           .swapaxes(1, 2).reshape(bh * bw, -1).std(axis=1))
    lb = np.log(blk + 1e-3)
    chans = [(img[..., i] - ndimage.gaussian_filter(img[..., i], 1.0))[4:-4, 4:-4].ravel()
             for i in range(3)]
    c_rg = np.corrcoef(chans[0], chans[1])[0, 1]
    c_gb = np.corrcoef(chans[1], chans[2])[0, 1]
    return [np.log(flat.std() + 1e-3), stats.kurtosis(flat), stats.skew(flat),
            lb.mean(), lb.std(), np.percentile(lb, 90) - np.percentile(lb, 10),
            c_rg, c_gb]


def blockiness(g):
    d = np.abs(np.diff(g, axis=1))
    idx = np.arange(d.shape[1])
    on, off = d[:, idx % 8 == 7].mean(), d[:, idx % 8 != 7].mean()
    return np.log((on + 1e-6) / (off + 1e-6))


def extract(path):
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    g = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return spectrum_features(g) + residual_features(img) + [blockiness(g)]


def make_model(C=0.1):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def group_folds(pair_ids, k, seed):
    rng = np.random.default_rng(seed)
    p = np.array(sorted(set(pair_ids)))
    rng.shuffle(p)
    return [set(p[i::k]) for i in range(k)]


def platt_from_inner(F, y, pair, train_mask, k_inner=5, seed=0):
    tr_pairs = sorted(set(pair[train_mask]))
    scores, labels = [], []
    for fold in group_folds(tr_pairs, k_inner, seed):
        va = train_mask & np.isin(pair, list(fold))
        tr = train_mask & ~np.isin(pair, list(fold))
        m = make_model().fit(F[tr], y[tr])
        scores.append(m.decision_function(F[va]))
        labels.append(y[va])
    s, l = np.concatenate(scores), np.concatenate(labels)
    return LogisticRegression(C=1e6, max_iter=3000).fit(s.reshape(-1, 1), l)


def run_cv(F, y, pair, variant, train_variants, k=6, repeats=10, calibrate=True):
    n = len(y)
    raw = np.zeros((repeats, n))
    cal = np.zeros((repeats, n))
    for r in range(repeats):
        for fold in group_folds(pair, k, seed=r):
            te = np.isin(pair, list(fold))
            tr = ~te & np.isin(variant, train_variants)
            m = make_model().fit(F[tr], y[tr])
            raw[r, te] = m.predict_proba(F[te])[:, 1]
            if calibrate:
                pl = platt_from_inner(F, y, pair, tr, seed=100 + r)
                cal[r, te] = pl.predict_proba(m.decision_function(F[te]).reshape(-1, 1))[:, 1]
    return raw.mean(axis=0), (cal.mean(axis=0) if calibrate else None)


def ece_score(y, p, bins=5):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return tot


def boot_ci(fn, y, p, pair, mask, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    pairs = np.array(sorted(set(pair[mask])))
    by_pair = {q: np.where(mask & (pair == q))[0] for q in pairs}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(pairs, size=len(pairs), replace=True)
        idx = np.concatenate([by_pair[q] for q in pick])
        vals.append(fn(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarize(y, p, pair, variant, calibrated=False, n_boot=1000):
    out = {}
    for v in VARIANTS:
        m = variant == v
        d = {"auc": float(roc_auc_score(y[m], p[m])),
             "auc_ci": boot_ci(roc_auc_score, y, p, pair, m, n_boot),
             "bal_acc": float(balanced_accuracy_score(y[m], p[m] >= 0.5)),
             "brier": float(brier_score_loss(y[m], p[m])),
             "ece": float(ece_score(y[m], p[m])),
             "ece_ci": boot_ci(ece_score, y, p, pair, m, n_boot)}
        out[v] = d
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="resultados")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(data / "labels_all.csv", encoding="utf-8")))
    root = data.parent

    feats, bpp = [], []
    for i, r in enumerate(rows):
        path = root / r["file"].replace("\\", "/")
        feats.append(extract(path))
        with Image.open(path) as im:
            bpp.append(os.path.getsize(path) * 8 / (im.width * im.height))
        if (i + 1) % 40 == 0:
            print(f"rasgos: {i + 1}/{len(rows)}")
    F = np.array(feats, dtype=np.float64)
    F[~np.isfinite(F)] = 0.0
    y = np.array([int(r["label"]) for r in rows])
    pair = np.array([r["pair_id"] for r in rows])
    variant = np.array([r["variant"] for r in rows])
    bpp = np.array(bpp).reshape(-1, 1)

    with open(out / "features.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "pair_id", "label", "variant"] + FEATURE_NAMES + ["bits_per_pixel"])
        for r, fv, b in zip(rows, F, bpp[:, 0]):
            w.writerow([r["sample_id"], r["pair_id"], r["label"], r["variant"]] +
                       [f"{v:.6g}" for v in fv] + [f"{b:.4f}"])

    res = {"n_pairs": int(len(set(pair))), "n_samples": int(len(y)),
           "repeats": args.repeats, "folds": 6}
    print("evaluando: completo (entrena orig)...")
    raw_a, cal_a = run_cv(F, y, pair, variant, ["orig"], repeats=args.repeats)
    res["A_full_train_orig_raw"] = summarize(y, raw_a, pair, variant, n_boot=args.boot)
    res["A_full_train_orig_calibrated"] = summarize(y, cal_a, pair, variant, n_boot=args.boot)
    print("evaluando: completo (entrena todas las variantes)...")
    raw_b, cal_b = run_cv(F, y, pair, variant, VARIANTS, repeats=args.repeats)
    res["B_full_train_all_raw"] = summarize(y, raw_b, pair, variant, n_boot=args.boot)
    res["B_full_train_all_calibrated"] = summarize(y, cal_b, pair, variant, n_boot=args.boot)
    print("evaluando: controles...")
    hf, _ = run_cv(F[:, [HF_IDX]], y, pair, variant, ["orig"], repeats=args.repeats, calibrate=False)
    res["control_hf_only_train_orig"] = summarize(y, hf, pair, variant, n_boot=args.boot)
    bp, _ = run_cv(bpp, y, pair, variant, ["orig"], repeats=args.repeats, calibrate=False)
    res["control_bpp_only_train_orig"] = summarize(y, bp, pair, variant, n_boot=args.boot)

    with open(out / "resultados.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


    def row(name, block):
        cells = [f"{block[v]['auc']:.2f} [{block[v]['auc_ci'][0]:.2f}-{block[v]['auc_ci'][1]:.2f}]"
                 for v in VARIANTS]
        return f"| {name} | " + " | ".join(cells) + " |"

    md = [f"# Resultados piloto FIDES ({res['n_pairs']} pares, {res['n_samples']} muestras)\n",
          "AUC [IC 95% bootstrap por pares]\n",
          "| Modelo | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS),
          row("Completo, entrena orig (A)", res["A_full_train_orig_raw"]),
          row("Completo, entrena todas (B)", res["B_full_train_all_raw"]),
          row("Solo razon alta frecuencia", res["control_hf_only_train_orig"]),
          row("Control: solo bits/pixel", res["control_bpp_only_train_orig"]),
          "\nCalibracion, modelo A (ECE con 5 intervalos / Brier)\n",
          "| Version | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
    for name, key in (("Sin calibrar", "A_full_train_orig_raw"), ("Platt", "A_full_train_orig_calibrated")):
        b = res[key]
        md.append(f"| {name} | " + " | ".join(f"{b[v]['ece']:.2f} / {b[v]['brier']:.2f}" for v in VARIANTS) + " |")
    (out / "resultados.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(VARIANTS))
    for off, key, lab in ((-0.2, "A_full_train_orig_raw", "Entrena orig"),
                          (0.2, "B_full_train_all_raw", "Entrena todas")):
        a = np.array([res[key][v]["auc"] for v in VARIANTS])
        lo = a - np.array([res[key][v]["auc_ci"][0] for v in VARIANTS])
        hi = np.array([res[key][v]["auc_ci"][1] for v in VARIANTS]) - a
        ax[0].errorbar(x + off, a, yerr=[lo, hi], fmt="o", capsize=3, label=lab)
    ax[0].plot(x, [res["control_bpp_only_train_orig"][v]["auc"] for v in VARIANTS],
               "k^", label="Control bits/pixel")
    ax[0].axhline(0.5, color="gray", ls="--", lw=0.8)
    ax[0].set_xticks(x, VARIANTS)
    ax[0].set_ylabel("AUC")
    ax[0].set_title("Robustez a recompresion y reescalado")
    ax[0].legend(fontsize=8)
    edges = np.linspace(0, 1, 6)
    for name, mask, p in (("orig", variant == "orig", cal_a),
                          ("variantes degradadas", variant != "orig", cal_a)):
        idx = np.minimum((p[mask] * 5).astype(int), 4)
        xs = [p[mask][idx == b].mean() for b in range(5) if (idx == b).any()]
        ys = [y[mask][idx == b].mean() for b in range(5) if (idx == b).any()]
        ax[1].plot(xs, ys, "o-", label=name)
    ax[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[1].set_xlabel("Probabilidad predicha (calibrada)")
    ax[1].set_ylabel("Frecuencia observada")
    ax[1].set_title("Diagrama de fiabilidad (modelo A)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figura_resultados.png", dpi=150)
    print(f"\nListo -> {out}/ (resultados.md, resultados.json, features.csv, figura_resultados.png)")


if __name__ == "__main__":
    main()
