"""Preparacion del dataset de pares real/editada: normalizacion, etiquetado y degradaciones."""
import argparse
import csv
import hashlib
import sys
from pathlib import Path

from PIL import Image, ImageOps

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
MAX_SIDE = 1024
QUALITY = 92
ASPECT_TOL = 0.02
JPEG_VARIANTS = (90, 70, 50)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rgb(path: Path):
    img = Image.open(path)
    had_exif = bool(img.getexif())
    img = ImageOps.exif_transpose(img).convert("RGB")
    return img, had_exif


def fit(img: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    scale = max_side / max(img.size)
    if scale < 1:
        new = (round(img.width * scale), round(img.height * scale))
        img = img.resize(new, Image.LANCZOS)
    return img


def save_jpeg(img: Image.Image, dst: Path, quality: int = QUALITY) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    img.save(dst, "JPEG", quality=quality, subsampling="4:2:0")


def index_dir(folder: Path) -> dict:
    if not folder.is_dir():
        sys.exit(f"No existe la carpeta {folder}")
    return {p.stem: p for p in sorted(folder.iterdir()) if p.suffix.lower() in EXTS}


def read_edit_log(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return {}
    if lines[0].startswith('"') and lines[0].rstrip().endswith('"') \
            and "source_stem" in lines[0]:
        lines = [ln.strip()[1:-1] for ln in lines]
    delim = ";" if lines[0].count(";") > lines[0].count(",") else ","
    reader = csv.DictReader(lines, delimiter=delim)
    return {r["source_stem"].strip(): r for r in reader if r.get("source_stem")}


def cmd_prepare(args) -> None:
    raw, out = Path(args.raw), Path(args.out)
    real = index_dir(raw / "real")
    edited = index_dir(raw / "edited")
    log_path = raw / "edits_log.csv"
    log = read_edit_log(log_path)

    pairs, missing = [], []
    for stem, rpath in real.items():
        epath = edited.get(stem + "_edit")
        (pairs if epath else missing).append((stem, rpath, epath))
    pairs = [p for p in pairs if p[2] is not None]
    skip = {x.strip() for x in args.exclude.split(",") if x.strip()}
    if skip:
        print("Excluidos por --exclude:", ", ".join(sorted(skip & {p[0] for p in pairs})) or "(ninguno coincide)")
        pairs = [p for p in pairs if p[0] not in skip]
    orphans = [s for s in edited if s.endswith("_edit") and s[:-5] not in real]

    if missing:
        print("AVISO: fotos reales SIN edicion (se omiten):",
              ", ".join(m[0] for m in missing))
    if orphans:
        print("AVISO: ediciones sin foto real (se omiten):", ", ".join(orphans))
    if not pairs:
        sys.exit("No se encontro ningun par. Revisa los nombres (IMG_001 / IMG_001_edit).")

    rows = []
    for i, (stem, rpath, epath) in enumerate(pairs, start=1):
        pair_id = f"P{i:03d}"
        r_img, r_exif = load_rgb(rpath)
        e_img, e_exif = load_rgb(epath)

        asp_r, asp_e = r_img.width / r_img.height, e_img.width / e_img.height
        if abs(asp_r - asp_e) / asp_r > ASPECT_TOL:
            print(f"AVISO {pair_id} ({stem}): distinta relacion de aspecto "
                  f"({r_img.size} vs {e_img.size}). La edicion cambio el encuadre; "
                  "conviene rehacerla o recortar la original.")

        r_fit, e_fit = fit(r_img), fit(e_img)
        if abs(asp_r - asp_e) / asp_r <= ASPECT_TOL and e_fit.size != r_fit.size:


            e_fit = e_fit.resize(r_fit.size, Image.LANCZOS)

        meta = log.get(stem, {})
        for label, cls, img, src, had in ((0, "real", r_fit, rpath, r_exif),
                                          (1, "edited", e_fit, epath, e_exif)):
            dst = out / cls / f"{pair_id}.jpg"
            save_jpeg(img, dst)
            rows.append({
                "pair_id": pair_id, "label": label, "class": cls,
                "file": str(dst), "source_file": src.name,
                "sha256_original": sha256(src), "sha256_prepared": sha256(dst),
                "width": img.width, "height": img.height,
                "orig_had_exif": int(had),
                "tool": ((meta.get("tool") or "").strip() or args.tool) if label else "",
                "edit_description": ((meta.get("edit_description") or "").strip()
                                     or args.description) if label else "",
            })

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


    if not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["source_stem", "tool", "model", "edit_description", "region"])
            for stem, _, _ in pairs:
                w.writerow([stem, "", "", "", ""])
        print(f"Se creo la plantilla {log_path}: completa herramienta y descripcion "
              "de cada edicion y vuelve a correr 'prepare'.")

    n_exif_r = sum(r["orig_had_exif"] for r in rows if r["label"] == 0)
    n_exif_e = sum(r["orig_had_exif"] for r in rows if r["label"] == 1)
    print(f"Listo: {len(pairs)} pares -> {out/'labels.csv'}")
    print(f"Originales con EXIF: reales {n_exif_r}/{len(pairs)}, "
          f"editadas {n_exif_e}/{len(pairs)} "
          "(las diferencias quedaron neutralizadas: ambas clases se guardan sin EXIF).")


def cmd_degrade(args) -> None:
    out = Path(args.out)
    src_csv = out / "labels.csv"
    if not src_csv.exists():
        sys.exit("Falta labels.csv: corre primero 'prepare'.")
    with open(src_csv, newline="", encoding="utf-8") as f:
        base = list(csv.DictReader(f))

    rows = []
    for r in base:
        src = Path(r["file"])
        img = Image.open(src).convert("RGB")
        variants = [("orig", src)]
        for q in JPEG_VARIANTS:
            dst = out / "degraded" / f"jpeg{q}" / r["class"] / src.name
            save_jpeg(img, dst, quality=q)
            variants.append((f"jpeg{q}", dst))
        half = img.resize((max(1, img.width // 2), max(1, img.height // 2)), Image.LANCZOS)
        dst = out / "degraded" / "scale50" / r["class"] / src.name
        save_jpeg(half, dst, quality=QUALITY)
        variants.append(("scale50", dst))

        for name, path in variants:
            rows.append({
                "sample_id": f"{r['pair_id']}_{r['class']}_{name}",
                "pair_id": r["pair_id"], "label": r["label"], "class": r["class"],
                "variant": name, "file": str(path),
            })

    with open(out / "labels_all.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Listo: {len(rows)} muestras -> {out/'labels_all.csv'} "
          "(agrupa SIEMPRE por pair_id al validar).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("prepare", help="emparejar, normalizar formato y etiquetar")
    p1.add_argument("--raw", default="raw")
    p1.add_argument("--out", default="data")
    p1.add_argument("--tool", default="",
                    help="herramienta de IA usada (se aplica a las filas sin tool en edits_log.csv)")
    p1.add_argument("--exclude", default="",
                    help="fotos originales a excluir, separadas por coma y sin extension")
    p1.add_argument("--description", default="",
                    help="descripcion general de la edicion (opcional)")
    p1.set_defaults(func=cmd_prepare)
    p2 = sub.add_parser("degrade", help="generar variantes degradadas")
    p2.add_argument("--out", default="data")
    p2.set_defaults(func=cmd_degrade)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
