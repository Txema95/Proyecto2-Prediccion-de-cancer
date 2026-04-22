from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

CLASES_OBJETIVO: tuple[str, ...] = (
    "normal-cecum",
    "polyps",
    "dyed-lifted-polyps",
    "ulcerative-colitis",
)


@dataclass
class EstadisticaRecorte:
    recorte_aplicado: bool
    pixeles_recortados: int
    ancho_original: int
    alto_original: int
    ancho_recorte: int
    alto_recorte: int


def raiz_proyecto() -> Path:
    return Path(__file__).resolve().parents[3]


def parsear_argumentos() -> argparse.Namespace:
    raiz = raiz_proyecto()
    parser = argparse.ArgumentParser(
        description=(
            "Preprocesado minimo comun para Kvasir: recorte de bordes negros, "
            "normalizacion geometrica y salida reproducible."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=raiz / "data" / "raw" / "kvasir-dataset-v2",
        help="Carpeta de entrada con las clases Kvasir.",
    )
    parser.add_argument(
        "--manifest-in",
        type=Path,
        default=raiz / "data" / "processed" / "kvasir_image_eda" / "paso2_manifest_muestreo.csv",
        help="Manifest del EDA para filtrar filas seleccionadas=true.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=raiz / "data" / "processed" / "kvasir_min_clean",
        help="Carpeta de salida para imagenes y reportes de cleaning.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Lado final (cuadrado) despues de preprocesar.",
    )
    parser.add_argument(
        "--umbral-negro",
        type=int,
        default=12,
        help="Umbral [0-255] para detectar viñeteado/bordes negros.",
    )
    parser.add_argument(
        "--padding-fraccion",
        type=float,
        default=0.02,
        help="Padding interno proporcional tras recorte para evitar bordes duros.",
    )
    parser.add_argument(
        "--max-imagenes",
        type=int,
        default=0,
        help="Limita cuantas imagenes procesar (0 = todas).",
    )
    return parser.parse_args()


def _filtrar_manifest_seleccionado(manifest: pd.DataFrame) -> pd.DataFrame:
    if "seleccionado" not in manifest.columns:
        raise ValueError("El manifest debe incluir la columna 'seleccionado'.")
    serie = manifest["seleccionado"]
    if serie.dtype == object:
        mask = serie.astype(str).str.lower().isin(("true", "1", "yes"))
    else:
        mask = serie.astype(bool)
    return manifest[mask].copy()


def cargar_rutas_objetivo(dataset_root: Path, manifest_path: Path) -> list[dict[str, str]]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"No existe dataset-root: {dataset_root}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No existe manifest-in: {manifest_path}")

    man = pd.read_csv(manifest_path)
    sel = _filtrar_manifest_seleccionado(man)
    filas: list[dict[str, str]] = []
    for _, row in sel.iterrows():
        clase = str(row["clase"])
        if clase not in CLASES_OBJETIVO:
            continue
        ruta_rel = str(row["ruta_relativa"]).replace("\\", "/")
        ruta_abs = (dataset_root / ruta_rel).resolve()
        filas.append(
            {
                "clase": clase,
                "ruta_relativa": ruta_rel,
                "ruta_absoluta": str(ruta_abs),
            }
        )
    return filas


def recortar_bordes_negros(img: Image.Image, umbral_negro: int) -> tuple[Image.Image, EstadisticaRecorte]:
    rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.uint8)
    gris = np.max(arr, axis=2)
    mascara = gris > umbral_negro

    alto, ancho = mascara.shape
    ys, xs = np.where(mascara)
    if len(xs) == 0 or len(ys) == 0:
        estad = EstadisticaRecorte(
            recorte_aplicado=False,
            pixeles_recortados=0,
            ancho_original=ancho,
            alto_original=alto,
            ancho_recorte=ancho,
            alto_recorte=alto,
        )
        return rgb, estad

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    recortada = rgb.crop((x0, y0, x1, y1))
    pix_original = ancho * alto
    pix_recorte = recortada.width * recortada.height

    estad = EstadisticaRecorte(
        recorte_aplicado=(x0 > 0 or y0 > 0 or x1 < ancho or y1 < alto),
        pixeles_recortados=max(0, pix_original - pix_recorte),
        ancho_original=ancho,
        alto_original=alto,
        ancho_recorte=recortada.width,
        alto_recorte=recortada.height,
    )
    return recortada, estad


def normalizar_geometria(img: Image.Image, size: int, padding_fraccion: float) -> Image.Image:
    if img.width == 0 or img.height == 0:
        return Image.new("RGB", (size, size), (0, 0, 0))

    # Paso 1: recorte centrado cuadrado para estandarizar encuadre.
    lado = min(img.width, img.height)
    x0 = (img.width - lado) // 2
    y0 = (img.height - lado) // 2
    cuadrada = img.crop((x0, y0, x0 + lado, y0 + lado))

    # Paso 2: leve padding interno para no dejar bordes abruptos.
    pad = int(max(0, round(size * padding_fraccion)))
    size_interno = max(1, size - 2 * pad)
    redim = cuadrada.resize((size_interno, size_interno), Image.Resampling.LANCZOS)
    if pad == 0:
        return redim
    return ImageOps.expand(redim, border=pad, fill=(0, 0, 0))


def procesar_todas(
    filas: list[dict[str, str]],
    output_root: Path,
    size: int,
    umbral_negro: int,
    padding_fraccion: float,
    max_imagenes: int,
) -> tuple[pd.DataFrame, dict]:
    dir_img = output_root / "imagenes"
    dir_img.mkdir(parents=True, exist_ok=True)

    registros: list[dict] = []
    errores: list[dict] = []
    total = len(filas) if max_imagenes <= 0 else min(len(filas), max_imagenes)

    for idx, fila in enumerate(filas[:total], start=1):
        ruta_in = Path(fila["ruta_absoluta"])
        clase = fila["clase"]
        ruta_rel = fila["ruta_relativa"]

        out_clase = dir_img / clase
        out_clase.mkdir(parents=True, exist_ok=True)
        out_name = Path(ruta_rel).name
        ruta_out = out_clase / out_name

        if not ruta_in.is_file():
            errores.append({"ruta": str(ruta_in), "error": "archivo_no_existe"})
            continue

        try:
            with Image.open(ruta_in) as im:
                im = im.convert("RGB")
                rec, est = recortar_bordes_negros(im, umbral_negro=umbral_negro)
                fin = normalizar_geometria(rec, size=size, padding_fraccion=padding_fraccion)
                fin.save(ruta_out, format="JPEG", quality=95)
        except OSError as exc:
            errores.append({"ruta": str(ruta_in), "error": str(exc)})
            continue

        registros.append(
            {
                "clase": clase,
                "ruta_entrada": str(ruta_in),
                "ruta_salida": str(ruta_out),
                "nombre_archivo": out_name,
                "recorte_borde_negro_aplicado": est.recorte_aplicado,
                "pixeles_recortados": est.pixeles_recortados,
                "ancho_original": est.ancho_original,
                "alto_original": est.alto_original,
                "ancho_recorte": est.ancho_recorte,
                "alto_recorte": est.alto_recorte,
                "ancho_final": size,
                "alto_final": size,
            }
        )

        if idx % 250 == 0:
            print(f"Procesadas: {idx}/{total}")

    df = pd.DataFrame(registros)
    resumen = {
        "input_total_filas_manifest": len(filas),
        "procesadas": int(len(df)),
        "errores": int(len(errores)),
        "size_final": size,
        "umbral_negro": umbral_negro,
        "padding_fraccion": padding_fraccion,
        "por_clase": {},
        "errores_detalle_muestra": errores[:50],
    }

    if not df.empty:
        for clase in CLASES_OBJETIVO:
            sub = df[df["clase"] == clase]
            if sub.empty:
                continue
            resumen["por_clase"][clase] = {
                "n": int(len(sub)),
                "recorte_borde_negro_aplicado": int(sub["recorte_borde_negro_aplicado"].sum()),
                "pixeles_recortados_media": float(sub["pixeles_recortados"].mean()),
            }

    return df, resumen


def guardar_salidas(output_root: Path, df: pd.DataFrame, resumen: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    ruta_manifest = output_root / "manifest_clean.csv"
    ruta_resumen = output_root / "resumen_cleaning.json"
    df.to_csv(ruta_manifest, index=False, encoding="utf-8-sig")
    with ruta_resumen.open("w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parsear_argumentos()
    if args.size <= 0:
        raise ValueError("--size debe ser mayor que 0.")
    if args.umbral_negro < 0 or args.umbral_negro > 255:
        raise ValueError("--umbral-negro debe estar entre 0 y 255.")
    if args.padding_fraccion < 0 or args.padding_fraccion >= 0.5:
        raise ValueError("--padding-fraccion debe estar en [0.0, 0.5).")

    filas = cargar_rutas_objetivo(args.dataset_root, args.manifest_in)
    df, resumen = procesar_todas(
        filas=filas,
        output_root=args.output_root,
        size=args.size,
        umbral_negro=args.umbral_negro,
        padding_fraccion=args.padding_fraccion,
        max_imagenes=args.max_imagenes,
    )
    guardar_salidas(args.output_root, df, resumen)

    print(f"Cleaning completado. Imagenes procesadas: {len(df)}")
    print(f"Manifest limpio: {args.output_root / 'manifest_clean.csv'}")
    print(f"Resumen: {args.output_root / 'resumen_cleaning.json'}")


if __name__ == "__main__":
    main()
