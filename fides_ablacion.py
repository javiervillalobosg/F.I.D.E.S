"""Ablacion por familia de rasgos y tiempo de procesamiento por imagen."""
import argparse
import csv
import glob
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

import fides_eval as fe

SUBSETS = {"Solo espectro (13)": list(range(0, 13)),
           "Solo ruido residual (8)": list(range(13, 21)),
           "Solo rejilla JPEG (1)": [21],
           "Espectro + ruido (21)": list(range(0, 21)),
           "Todos (22)": list(range(0, 22))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="resultados")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    out = Path(a.out)

    rows = list(csv.DictReader(open(out / "features.csv", encoding="utf-8")))
    F = np.array([[float(r[n]) for n in fe.FEATURE_NAMES] for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    pair = np.array([r["pair_id"] for r in rows])
    variant = np.array([r["variant"] for r in rows])

    res = {}
    for name, idx in SUBSETS.items():
        raw, _ = fe.run_cv(F[:, idx], y, pair, variant, ["orig"], repeats=a.repeats, calibrate=False)
        s = fe.summarize(y, raw, pair, variant, n_boot=a.boot)
        res[name] = {v: (round(s[v]["auc"], 2), round(s[v]["auc_ci"][0], 2), round(s[v]["auc_ci"][1], 2))
                     for v in fe.VARIANTS}
        print(name, {v: res[name][v][0] for v in fe.VARIANTS})
    json.dump(res, open(out / "ablacion.json", "w"), indent=1, ensure_ascii=False)

    files = sorted(glob.glob(str(Path(a.data) / "real" / "*.jpg")))[:12] + \
        sorted(glob.glob(str(Path(a.data) / "edited" / "*.jpg")))[:12]
    fe.extract(files[0])
    t, px = [], []
    for f in files:
        t0 = time.perf_counter()
        fe.extract(f)
        t.append(time.perf_counter() - t0)
        with Image.open(f) as im:
            px.append(im.width * im.height)
    tim = {"ms_media": 1000 * float(np.mean(t)), "ms_mediana": 1000 * float(np.median(t)),
           "ms_max": 1000 * float(np.max(t)), "megapixeles_medios": float(np.mean(px)) / 1e6}
    json.dump(tim, open(out / "tiempos.json", "w"), indent=1)
    print("ms por imagen: media %.0f, max %.0f" % (tim["ms_media"], tim["ms_max"]))


if __name__ == "__main__":
    main()
