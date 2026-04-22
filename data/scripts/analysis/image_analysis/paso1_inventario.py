"""Paso 1: inventario de carpetas, conteos por extension, archivos corruptos o vacios."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

from configuracion import CLASES_ESPERADAS, EXTENSIONES_IMAGEN, directorio_salida, ruta_dataset_kvasir


@dataclass
class RegistroArchivo:
    clase: str
    ruta_relativa: str
    extension: str
    tamano_bytes: int
    legible_pillow: bool
    ancho: int | None
    alto: int | None
    error: str | None


def listar_imagenes_en_clase(carpeta_clase: Path) -> list[Path]:
    if not carpeta_clase.is_dir():
        return []
    salida: list[Path] = []
    for item in sorted(carpeta_clase.iterdir()):
        if item.is_file() and item.suffix.lower() in EXTENSIONES_IMAGEN:
            salida.append(item)
    return salida


def inspeccionar_archivo(clase: str, raiz_dataset: Path, archivo: Path) -> RegistroArchivo:
    rel = str(archivo.relative_to(raiz_dataset)).replace("\\", "/")
    ext = archivo.suffix.lower()
    try:
        tamano = archivo.stat().st_size
    except OSError as exc:
        return RegistroArchivo(clase, rel, ext, 0, False, None, None, f"stat: {exc}")

    if tamano == 0:
        return RegistroArchivo(clase, rel, ext, 0, False, None, None, "tamano_cero")

    ancho = alto = None
    legible = False
    error: str | None = None
    try:
        with Image.open(archivo) as img:
            img.verify()
        with Image.open(archivo) as img:
            img = img.convert("RGB")
            ancho, alto = img.size
        legible = True
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        error = str(exc)

    return RegistroArchivo(clase, rel, ext, tamano, legible, ancho, alto, error)


def ejecutar_inventario(raiz_dataset: Path) -> tuple[pd.DataFrame, dict]:
    """Devuelve detalle por archivo y un resumen agregado por clase."""
    if not raiz_dataset.is_dir():
        raise FileNotFoundError(f"No existe el directorio del dataset: {raiz_dataset}")

    carpetas_presentes = {p.name for p in raiz_dataset.iterdir() if p.is_dir()}
    faltantes = [c for c in CLASES_ESPERADAS if c not in carpetas_presentes]
    extra = sorted(carpetas_presentes - set(CLASES_ESPERADAS))

    registros: list[RegistroArchivo] = []
    for clase in CLASES_ESPERADAS:
        carpeta = raiz_dataset / clase
        for archivo in listar_imagenes_en_clase(carpeta):
            registros.append(inspeccionar_archivo(clase, raiz_dataset, archivo))

    df = pd.DataFrame([asdict(r) for r in registros])
    resumen: dict = {
        "ruta_dataset": str(raiz_dataset),
        "clases_esperadas": list(CLASES_ESPERADAS),
        "carpetas_faltantes": faltantes,
        "carpetas_extra_no_usadas": extra,
        "total_archivos_imagen": int(len(df)),
        "por_clase_total": {},
        "por_clase_legibles": {},
        "por_clase_corruptos": {},
        "extensiones_globales": {},
    }

    if df.empty:
        return df, resumen

    for clase in CLASES_ESPERADAS:
        sub = df[df["clase"] == clase]
        resumen["por_clase_total"][clase] = int(len(sub))
        resumen["por_clase_legibles"][clase] = int(sub["legible_pillow"].sum())
        resumen["por_clase_corruptos"][clase] = int((~sub["legible_pillow"]).sum())

    resumen["extensiones_globales"] = dict(Counter(df["extension"]))

    return df, resumen


def guardar_inventario(df: pd.DataFrame, resumen: dict, salida: Path) -> None:
    salida.mkdir(parents=True, exist_ok=True)
    csv_detalle = salida / "paso1_inventario_detalle.csv"
    json_resumen = salida / "paso1_inventario_resumen.json"
    df.to_csv(csv_detalle, index=False, encoding="utf-8-sig")
    with open(json_resumen, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)


def main() -> None:
    from configuracion import raiz_proyecto

    raiz = raiz_proyecto()
    dataset = ruta_dataset_kvasir(raiz)
    out = directorio_salida(raiz)
    df, resumen = ejecutar_inventario(dataset)
    guardar_inventario(df, resumen, out)
    print(f"Paso 1 listo. Detalle: {out / 'paso1_inventario_detalle.csv'}")
    print(f"Resumen JSON: {out / 'paso1_inventario_resumen.json'}")
    print(f"Total imagenes listadas: {len(df)}")
    if resumen.get("carpetas_faltantes"):
        print(f"Aviso: faltan carpetas de clase: {resumen['carpetas_faltantes']}")


if __name__ == "__main__":
    main()
