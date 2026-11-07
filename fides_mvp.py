"""Prototipo: ingesta con cadena de custodia, analisis calibrado e informe PDF."""
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import pickle
import platform
import shutil
import sqlite3
import stat
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import fides_eval as fe

VERSION = "0.1-piloto"
SEED = 7
N_BOOT = 100
MAX_SIDE, QUALITY = 1024, 92
GENESIS = "0" * 64

LIMITES = [
    "Modelo entrenado con 24 pares de imagenes (fotografia real y edicion de imagen completa con una sola herramienta). No se ha validado con otros generadores ni con edicion localizada.",
    "El desempeno cae con el reescalado (AUC piloto cercano a 0,63 al reducir la imagen al 50 %) y es moderado en general (AUC piloto cercano a 0,80). Un resultado bajo no prueba autenticidad.",
    "No se evaluo la cadena real de plataformas de mensajeria (WhatsApp, Instagram), ni intentos de evasion.",
    "La calibracion no mostro mejora demostrable con esta muestra: la probabilidad es una estimacion orientativa, no una probabilidad validada.",
    "El intervalo mostrado refleja la variabilidad del entrenamiento por remuestreo de pares; no cuantifica todos los errores posibles.",
    "No analiza capturas de mensajeria ni audio. No localiza zonas editadas.",
    "El resultado es un apoyo al juicio de un perito y no tiene validez judicial por si solo.",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def event_hash(prev, evidence_id, op, ts, detail):
    return hashlib.sha256("|".join([prev, str(evidence_id), op, ts, detail]).encode("utf-8")).hexdigest()


class Custody:
    def __init__(self, db_path="fides_custodia.db", store_dir="evidencia"):
        self.db_path, self.store = Path(db_path), Path(store_dir)
        self.store.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.db_path)
        self.con.executescript("""
        CREATE TABLE IF NOT EXISTS evidencia(
            id INTEGER PRIMARY KEY, nombre TEXT NOT NULL, sha256 TEXT NOT NULL,
            ingreso TEXT NOT NULL, copia TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS evento(
            id INTEGER PRIMARY KEY, evidencia_id INTEGER NOT NULL, operacion TEXT NOT NULL,
            ts TEXT NOT NULL, detalle TEXT NOT NULL, hash_previo TEXT NOT NULL, hash TEXT NOT NULL);
        CREATE TRIGGER IF NOT EXISTS evento_no_update BEFORE UPDATE ON evento
            BEGIN SELECT RAISE(ABORT, 'registro de custodia inmutable'); END;
        CREATE TRIGGER IF NOT EXISTS evento_no_delete BEFORE DELETE ON evento
            BEGIN SELECT RAISE(ABORT, 'registro de custodia inmutable'); END;
        CREATE TRIGGER IF NOT EXISTS evid_no_update BEFORE UPDATE ON evidencia
            BEGIN SELECT RAISE(ABORT, 'registro de custodia inmutable'); END;
        CREATE TRIGGER IF NOT EXISTS evid_no_delete BEFORE DELETE ON evidencia
            BEGIN SELECT RAISE(ABORT, 'registro de custodia inmutable'); END;
        """)
        self.con.commit()

    def _last_hash(self):
        r = self.con.execute("SELECT hash FROM evento ORDER BY id DESC LIMIT 1").fetchone()
        return r[0] if r else GENESIS

    def add_event(self, evidence_id, op, detail=""):
        ts, prev = now_iso(), self._last_hash()
        h = event_hash(prev, evidence_id, op, ts, detail)
        self.con.execute("INSERT INTO evento(evidencia_id,operacion,ts,detalle,hash_previo,hash) VALUES(?,?,?,?,?,?)",
                         (evidence_id, op, ts, detail, prev, h))
        self.con.commit()
        return h

    def ingest(self, path):
        path = Path(path)
        digest = sha256_file(path)
        dest = self.store / f"{digest[:16]}_{path.name}"
        if not dest.exists():
            shutil.copy2(path, dest)
            os.chmod(dest, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        cur = self.con.execute("INSERT INTO evidencia(nombre,sha256,ingreso,copia) VALUES(?,?,?,?)",
                               (path.name, digest, now_iso(), str(dest)))
        self.con.commit()
        eid = cur.lastrowid
        self.add_event(eid, "INGRESO", f"archivo={path.name}; sha256={digest}; bytes={path.stat().st_size}")
        return eid, digest, dest

    def events(self, evidence_id):
        return self.con.execute("SELECT id,operacion,ts,detalle,hash FROM evento WHERE evidencia_id=? ORDER BY id",
                                (evidence_id,)).fetchall()

    def verify(self):
        prev = GENESIS
        for (i, eid, op, ts, det, hp, h) in self.con.execute(
                "SELECT id,evidencia_id,operacion,ts,detalle,hash_previo,hash FROM evento ORDER BY id"):
            if hp != prev:
                return False, f"cadena rota en el evento {i}: hash previo no coincide"
            if event_hash(hp, eid, op, ts, det) != h:
                return False, f"evento {i} alterado: su hash no coincide"
            prev = h
        for (i, name, digest, copia) in self.con.execute("SELECT id,nombre,sha256,copia FROM evidencia"):
            if not Path(copia).exists():
                return False, f"falta la copia de la evidencia {i} ({name})"
            if sha256_file(copia) != digest:
                return False, f"la copia de la evidencia {i} ({name}) fue modificada"
        n = self.con.execute("SELECT COUNT(*) FROM evento").fetchone()[0]
        return True, f"cadena integra ({n} eventos)"


def load_features(csv_path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    F = np.array([[float(r[n]) for n in fe.FEATURE_NAMES] for r in rows])
    return (F, np.array([int(r["label"]) for r in rows]),
            np.array([r["pair_id"] for r in rows]), np.array([r["variant"] for r in rows]))


def train(features_csv, model_path="fides_modelo.pkl", metrics_json=None):
    F, y, pair, variant = load_features(features_csv)
    model = fe.make_model().fit(F, y)
    all_mask = np.ones(len(y), dtype=bool)
    platt = fe.platt_from_inner(F, y, pair, all_mask, seed=SEED)
    rng = np.random.default_rng(SEED)
    pairs = np.array(sorted(set(pair)))
    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(pairs, size=len(pairs), replace=True)
        idx = np.concatenate([np.where(pair == q)[0] for q in pick])
        boots.append(fe.make_model().fit(F[idx], y[idx]))
    pilot = {}
    if metrics_json and Path(metrics_json).exists():
        r = json.load(open(metrics_json, encoding="utf-8"))
        b = r["B_full_train_all_raw"]
        pilot = {v: round(b[v]["auc"], 2) for v in b}
    import sklearn
    meta = {"version": VERSION, "n_pairs": int(len(pairs)), "n_samples": int(len(y)), "seed": SEED,
            "n_boot": N_BOOT, "pilot_auc_B": pilot, "features": fe.FEATURE_NAMES,
            "entrenado": now_iso(), "sklearn": sklearn.__version__, "numpy": np.__version__,
            "python": platform.python_version()}
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "platt": platt, "boots": boots, "meta": meta}, f)
    print(f"Modelo guardado en {model_path} ({meta['n_pairs']} pares, {meta['n_samples']} muestras)")
    return meta


def normalize_for_analysis(path):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    orig_size = img.size
    scale = MAX_SIDE / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    fd, tmp = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    img.save(tmp, "JPEG", quality=QUALITY, subsampling="4:2:0")
    return tmp, orig_size, img.size


def score(bundle, feats):
    x = np.array(feats, dtype=np.float64).reshape(1, -1)
    x[~np.isfinite(x)] = 0.0
    s = bundle["model"].decision_function(x)
    p = float(bundle["platt"].predict_proba(s.reshape(-1, 1))[0, 1])
    ps = [float(bundle["platt"].predict_proba(m.decision_function(x).reshape(-1, 1))[0, 1])
          for m in bundle["boots"]]
    return p, float(np.percentile(ps, 2.5)), float(np.percentile(ps, 97.5)), float(s[0])


def interpretacion(p, lo, hi):
    if p >= 0.8:
        nivel = "indicio fuerte de edicion generativa"
    elif p >= 0.6:
        nivel = "indicio moderado de edicion generativa"
    elif p > 0.4:
        nivel = "resultado no concluyente"
    elif p > 0.2:
        nivel = "indicio moderado de imagen no editada"
    else:
        nivel = "indicio debil de edicion generativa"
    if hi - lo > 0.4 and p >= 0.6:
        nivel += " (con alta variabilidad: trate el resultado con cautela)"
    return nivel


def build_report(pdf_path, info):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=12.5)
    small = ParagraphStyle("s", parent=body, fontSize=8, leading=10)
    mono = ParagraphStyle("m", parent=small, fontName="Courier", fontSize=7)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=17, textColor=colors.HexColor("#1F3864"))
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, textColor=colors.HexColor("#2F5496"))

    def tbl(rows, widths):
        t = Table(rows, colWidths=widths)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BFBFBF")),
                               ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#D9E2F3")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"), ("FONTSIZE", (0, 0), (-1, -1), 8.5)]))
        return t

    P = lambda t, s=body: Paragraph(t, s)
    m = info["meta"]
    story = [P("F.I.D.E.S - Informe de analisis de imagen", h1),
             P("Documento orientativo generado por un prototipo de investigacion. No constituye peritaje.", small),
             Spacer(1, 8), P("1. Evidencia", h2),
             tbl([[P("Archivo", small), P(info["nombre"], small)],
                  [P("SHA-256", small), P(info["sha256"], mono)],
                  [P("Ingreso (UTC)", small), P(info["ingreso"], small)],
                  [P("Dimensiones", small), P(f"{info['size_orig'][0]}x{info['size_orig'][1]} px (analizada a {info['size_an'][0]}x{info['size_an'][1]} px)", small)],
                  [P("ID de evidencia", small), P(str(info["eid"]), small)]], [3.5 * cm, 13.5 * cm]),
             Spacer(1, 6), P("2. Resultado", h2),
             tbl([[P("Probabilidad estimada de edicion generativa", small), P(f"<b>{info['p']*100:.0f} %</b>", body)],
                  [P("Rango por remuestreo del entrenamiento (2,5 % - 97,5 %)", small), P(f"{info['lo']*100:.0f} % - {info['hi']*100:.0f} %", body)],
                  [P("Lectura", small), P(info["lectura"], body)]], [7 * cm, 10 * cm]),
             Spacer(1, 4),
             P("La probabilidad proviene de una regresion logistica sobre 22 rasgos de bajo nivel (espectro de "
               "frecuencias, ruido residual y rejilla JPEG), calibrada con escalado de Platt. Es una estimacion "
               "orientativa: la calibracion no se pudo validar con la muestra disponible.", body),
             Spacer(1, 6), P("3. Limites del analisis", h2)]
    for l in LIMITES:
        story.append(P("- " + l, body))
    story += [Spacer(1, 6), P("4. Parametros y reproducibilidad", h2),
              tbl([[P("Version de F.I.D.E.S", small), P(m["version"], small)],
                   [P("Modelo", small), P(f"entrenado {m['entrenado']} con {m['n_pairs']} pares / {m['n_samples']} muestras", small)],
                   [P("Semilla / remuestreos", small), P(f"{m['seed']} / {m['n_boot']}", small)],
                   [P("AUC piloto por variante (validacion cruzada)", small), P(json.dumps(m["pilot_auc_B"]) if m["pilot_auc_B"] else "no disponible", small)],
                   [P("Preprocesado", small), P(f"orientacion EXIF, RGB, lado mayor {MAX_SIDE} px, JPEG calidad {QUALITY}, sin metadatos", small)],
                   [P("Software", small), P(f"Python {m['python']}, NumPy {m['numpy']}, scikit-learn {m['sklearn']}", small)],
                   [P("Puntaje bruto del modelo", small), P(f"{info['raw']:.4f}", small)]], [5.5 * cm, 11.5 * cm]),
              Spacer(1, 6), P("5. Cadena de custodia", h2)]
    rows = [[P("#", small), P("Operacion", small), P("Fecha (UTC)", small), P("Hash del evento (SHA-256)", small)]]
    for (i, op, ts, det, h) in info["events"]:
        rows.append([P(str(i), small), P(op, small), P(ts, small), P(h, mono)])
    t = Table(rows, colWidths=[0.8 * cm, 2.8 * cm, 4.2 * cm, 9.2 * cm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BFBFBF")),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E2F3")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [t, Spacer(1, 4),
              P(f"Verificacion de la cadena al emitir el informe: {info['verif']}. El SHA-256 de este PDF se "
                "registra en la base de custodia como evento INFORME y se muestra en la terminal.", small)]
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="F.I.D.E.S - Informe", author="F.I.D.E.S")
    doc.build(story)


def analyze(image, model_path="fides_modelo.pkl", out="informes", db="fides_custodia.db", store="evidencia"):
    if not Path(model_path).exists():
        sys.exit(f"Falta {model_path}: corre primero 'python fides_mvp.py train ...'")
    bundle = pickle.load(open(model_path, "rb"))
    cust = Custody(db, store)
    eid, digest, copia = cust.ingest(image)
    tmp, size_orig, size_an = normalize_for_analysis(copia)
    try:
        feats = fe.extract(tmp)
    finally:
        os.remove(tmp)
    p, lo, hi, raw = score(bundle, feats)
    lectura = interpretacion(p, lo, hi)
    cust.add_event(eid, "ANALISIS", f"modelo={bundle['meta']['version']}; p={p:.4f}; rango={lo:.4f}-{hi:.4f}; semilla={bundle['meta']['seed']}")
    ok, msg = cust.verify()
    ev = cust.events(eid)
    ingreso = cust.con.execute("SELECT ingreso FROM evidencia WHERE id=?", (eid,)).fetchone()[0]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / f"informe_{eid:04d}_{digest[:8]}.pdf"
    build_report(pdf, {"nombre": Path(image).name, "sha256": digest, "ingreso": ingreso, "eid": eid,
                       "size_orig": size_orig, "size_an": size_an, "p": p, "lo": lo, "hi": hi, "raw": raw,
                       "lectura": lectura, "meta": bundle["meta"], "events": ev,
                       "verif": ("OK - " if ok else "FALLA - ") + msg})
    cust.add_event(eid, "INFORME", f"pdf={pdf.name}; sha256={sha256_file(pdf)}")
    print(f"Evidencia {eid}: sha256={digest}")
    print(f"Probabilidad de edicion generativa: {p*100:.0f} %  (rango {lo*100:.0f}-{hi*100:.0f} %)  -> {lectura}")
    print(f"Informe: {pdf}")
    print(f"Custodia: {'OK' if ok else 'FALLA'} - {msg}")
    return {"eid": eid, "p": p, "lo": lo, "hi": hi, "pdf": pdf, "sha256": digest}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--features", default="resultados/features.csv")
    t.add_argument("--metrics", default="resultados/resultados.json")
    t.add_argument("--model", default=os.environ.get("FIDES_MODEL", "fides_modelo.pkl"))
    a = sub.add_parser("analyze")
    a.add_argument("image")
    a.add_argument("--model", default=os.environ.get("FIDES_MODEL", "fides_modelo.pkl"))
    a.add_argument("--out", default="informes")
    a.add_argument("--db", default="fides_custodia.db")
    a.add_argument("--store", default="evidencia")
    v = sub.add_parser("verify")
    v.add_argument("--db", default="fides_custodia.db")
    v.add_argument("--store", default="evidencia")
    args = ap.parse_args()
    if args.cmd == "train":
        train(args.features, args.model, args.metrics)
    elif args.cmd == "analyze":
        analyze(args.image, args.model, args.out, args.db, args.store)
    else:
        ok, msg = Custody(args.db, args.store).verify()
        print(("OK - " if ok else "FALLA - ") + msg)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
