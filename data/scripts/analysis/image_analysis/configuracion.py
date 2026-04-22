"""Rutas y constantes para el analisis del dataset Kvasir (pasos 1-4)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Cuatro clases acordadas para el clasificador (nombres de carpeta en Kvasir v2).
CLASES_ESPERADAS: tuple[str, ...] = (
    "normal-cecum",
    "polyps",
    "dyed-lifted-polyps",
    "ulcerative-colitis",
)

EXTENSIONES_IMAGEN: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})

# Objetivo de muestreo reproducible por clase (segun diseno del proyecto).
IMAGENES_OBJETIVO_POR_CLASE: int = 1000

SEMILLA_PREDETERMINADA: int = 42


def raiz_proyecto() -> Path:
    """Sube desde data/scripts/analysis/image_analysis/ hasta la raiz del repo."""
    return Path(__file__).resolve().parents[4]


def ruta_dataset_kvasir(raiz: Path | None = None) -> Path:
    if raiz is None:
        raiz = raiz_proyecto()
    return raiz / "data" / "raw" / "kvasir-dataset-v2"


def directorio_salida(raiz: Path | None = None) -> Path:
    if raiz is None:
        raiz = raiz_proyecto()
    out = raiz / "data" / "processed" / "kvasir_image_eda"
    out.mkdir(parents=True, exist_ok=True)
    return out


def manifest_filtrar_seleccionado(man: pd.DataFrame) -> pd.DataFrame:
    """Evita fallos si `seleccionado` llega como texto tras leer CSV."""
    if "seleccionado" not in man.columns:
        raise ValueError("Falta la columna 'seleccionado' en el manifest.")
    serie = man["seleccionado"]
    if serie.dtype == object:
        mask = serie.astype(str).str.lower().isin(("true", "1", "yes"))
        return man[mask].copy()
    return man[serie.astype(bool)].copy()
