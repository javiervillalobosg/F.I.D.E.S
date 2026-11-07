"""Pruebas del prototipo."""
import csv
import sqlite3

import numpy as np
import pytest
from PIL import Image

import fides_eval as fe
import fides_mvp as mvp


@pytest.fixture()
def img(tmp_path):
    rng = np.random.default_rng(0)
    p = tmp_path / "foto.jpg"
    Image.fromarray((rng.random((300, 400, 3)) * 255).astype("uint8")).save(p, quality=90)
    return p


@pytest.fixture()
def cust(tmp_path):
    return mvp.Custody(tmp_path / "c.db", tmp_path / "ev")


@pytest.fixture()
def features_csv(tmp_path):
    rng = np.random.default_rng(1)
    path = tmp_path / "features.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "pair_id", "label", "variant"] + fe.FEATURE_NAMES + ["bits_per_pixel"])
        for i in range(20):
            for label in (0, 1):
                for v in ("orig", "jpeg90"):
                    x = rng.normal(label * 0.8, 1.0, len(fe.FEATURE_NAMES))
                    w.writerow([f"{i}_{label}_{v}", f"P{i:03d}", label, v] + list(x) + [1.0])
    return path


def test_sha256_conocido(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"abc")
    assert mvp.sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_ingesta_registra_hash_copia_y_solo_lectura(cust, img):
    eid, digest, dest = cust.ingest(img)
    assert digest == mvp.sha256_file(img)
    assert dest.exists() and mvp.sha256_file(dest) == digest
    assert not (dest.stat().st_mode & 0o222)
    assert cust.events(eid)[0][1] == "INGRESO"


def test_cadena_integra_y_encadenada(cust, img):
    eid, _, _ = cust.ingest(img)
    cust.add_event(eid, "ANALISIS", "x")
    ok, _ = cust.verify()
    assert ok
    ev = cust.con.execute("SELECT hash_previo, hash FROM evento ORDER BY id").fetchall()
    assert ev[0][0] == mvp.GENESIS and ev[1][0] == ev[0][1]


def test_triggers_bloquean_update_y_delete(cust, img):
    cust.ingest(img)
    with pytest.raises(sqlite3.DatabaseError):
        cust.con.execute("UPDATE evento SET detalle='x'")
    with pytest.raises(sqlite3.DatabaseError):
        cust.con.execute("DELETE FROM evento")


def test_alteracion_directa_rompe_la_cadena(cust, img):
    eid, _, _ = cust.ingest(img)
    cust.add_event(eid, "ANALISIS", "x")
    cust.con.execute("DROP TRIGGER evento_no_update")
    cust.con.execute("UPDATE evento SET detalle='alterado' WHERE id=1")
    cust.con.commit()
    ok, msg = cust.verify()
    assert not ok and "alterado" in msg


def test_copia_modificada_se_detecta(cust, img):
    _, _, dest = cust.ingest(img)
    dest.chmod(0o644)
    dest.write_bytes(b"otro contenido")
    ok, msg = cust.verify()
    assert not ok and "modificada" in msg


def test_normalizacion_limita_lado_mayor(tmp_path):
    p = tmp_path / "grande.png"
    Image.new("RGB", (3000, 2000), (120, 130, 140)).save(p)
    tmp, orig, an = mvp.normalize_for_analysis(p)
    assert orig == (3000, 2000) and max(an) == mvp.MAX_SIDE
    Image.open(tmp).verify()


def test_entrenamiento_y_puntaje_en_rango(tmp_path, features_csv):
    model = tmp_path / "m.pkl"
    meta = mvp.train(features_csv, model)
    assert meta["n_pairs"] == 20
    import pickle
    bundle = pickle.load(open(model, "rb"))
    p, lo, hi, _ = mvp.score(bundle, np.zeros(len(fe.FEATURE_NAMES)))
    assert 0 <= lo <= hi <= 1 and 0 <= p <= 1


def test_entrenamiento_es_determinista(tmp_path, features_csv):
    import pickle
    mvp.train(features_csv, tmp_path / "a.pkl")
    mvp.train(features_csv, tmp_path / "b.pkl")
    a, b = (pickle.load(open(tmp_path / n, "rb")) for n in ("a.pkl", "b.pkl"))
    x = np.ones(len(fe.FEATURE_NAMES))
    assert mvp.score(a, x)[:3] == mvp.score(b, x)[:3]


def test_analisis_completo_genera_pdf_y_eventos(tmp_path, features_csv, img):
    model = tmp_path / "m.pkl"
    mvp.train(features_csv, model)
    r = mvp.analyze(img, model, tmp_path / "inf", tmp_path / "c.db", tmp_path / "ev")
    assert r["pdf"].exists() and r["pdf"].read_bytes()[:4] == b"%PDF"
    c = mvp.Custody(tmp_path / "c.db", tmp_path / "ev")
    assert [e[1] for e in c.events(r["eid"])] == ["INGRESO", "ANALISIS", "INFORME"]
    assert c.verify()[0]


def test_interpretacion_no_concluyente_en_el_centro():
    assert "no concluyente" in mvp.interpretacion(0.5, 0.3, 0.7)
    assert "fuerte" in mvp.interpretacion(0.9, 0.85, 0.95)
