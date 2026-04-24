"""
Crea `splits_kvasir_multiclase.csv` desde el manifest multiclase, con partición por
`group_id` y estratificación aproximada por etiqueta (clase).

Uso (raíz del repo):
    uv run python dl/vision_baseline_kvasir/crear_splits.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    _raiz = Path(__file__).resolve().parents[2]
    if str(_raiz) not in sys.path:
        sys.path.insert(0, str(_raiz))

from dl.vision_baseline_kvasir import particion
from dl.vision_baseline_kvasir.paths import raiz_proyecto


def main() -> None:
    r = raiz_proyecto()
    p = argparse.ArgumentParser(description="Train/val/test a partir de manifest Kvasir.")
    p.add_argument(
        "--manifest",
        type=Path,
        default=r / "data" / "processed" / "kvasir_min_clean" / "manifest_kvasir_multiclase.csv",
    )
    p.add_argument(
        "--salida",
        type=Path,
        default=r / "data" / "processed" / "kvasir_min_clean" / "splits_kvasir_multiclase.csv",
    )
    p.add_argument("--resumen", type=Path, default=None, help="JSON opcional con resumen de conteos.")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--semilla", type=int, default=42)
    a = p.parse_args()
    particion.validar_proporciones(a.train_ratio, a.val_ratio, a.test_ratio)
    filas = particion.leer_manifest(a.manifest)
    asig = particion.dividir_por_etiqueta(
        filas, a.train_ratio, a.val_ratio, a.test_ratio, a.semilla
    )
    particion.validar_fuga(filas, asig)
    res = particion.escribir_splits(filas, asig, a.salida, a.semilla)
    if a.resumen is not None:
        a.resumen.parent.mkdir(parents=True, exist_ok=True)
        a.resumen.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    particion.imprimir_resumen(res)


if __name__ == "__main__":
    main()
