"""
Aplica el mismo preprocesado minimo que `data/scripts/cleaning/kvasir_preprocesado_minimo.py`
a una imagen PIL (p. ej. subida al simulador), sin escribir a disco.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from PIL import Image

_NOMBRE_MODULO_DIN = "kvasir_preprocesado_minimo_inferencia"


def _cargar_modulo_cleaning(raiz: Path) -> Any:
    ruta = raiz / "data" / "scripts" / "cleaning" / "kvasir_preprocesado_minimo.py"
    if not ruta.is_file():
        return None
    spec = importlib.util.spec_from_file_location(_NOMBRE_MODULO_DIN, ruta)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    # Necesario para dataclasses y otros al usar exec_module dinamico
    sys.modules[_NOMBRE_MODULO_DIN] = mod
    spec.loader.exec_module(mod)
    return mod


def aplicar_preprocesado_minimo_entrenamiento(
    im: Image.Image,
    raiz: Path,
    size: int = 512,
    umbral_negro: int = 12,
    padding_fraccion: float = 0.02,
) -> tuple[Image.Image, dict[str, Any]]:
    """
    Recorte bordes negros + recorte cuadrado + resize con padding, como en el cleaning.
    Si no se encuentra el script de cleaning, devuelve la imagen original y un dict con error.
    """
    mod = _cargar_modulo_cleaning(raiz)
    if mod is None:
        return im.convert("RGB"), {
            "aplicado": False,
            "motivo": f"No se encontro: {raiz / 'data/scripts/cleaning/kvasir_preprocesado_minimo.py'}",
        }
    rgb = im.convert("RGB")
    rec, est = mod.recortar_bordes_negros(rgb, umbral_negro)
    fin = mod.normalizar_geometria(rec, size=size, padding_fraccion=padding_fraccion)
    meta: dict[str, Any] = {
        "aplicado": True,
        "size_salida": size,
        "umbral_negro": umbral_negro,
        "padding_fraccion": padding_fraccion,
        "recorte_borde_negro": est.recorte_aplicado,
        "pixeles_recortados_borde": int(est.pixeles_recortados),
        "ancho_original": int(est.ancho_original),
        "alto_original": int(est.alto_original),
        "ancho_tras_borde": int(est.ancho_recorte),
        "alto_tras_borde": int(est.alto_recorte),
    }
    return fin, meta
