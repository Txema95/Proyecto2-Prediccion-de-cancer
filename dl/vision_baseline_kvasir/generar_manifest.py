"""
Genera `manifest_kvasir_multiclase.csv` deduplicado por MD5 (EDA paso 3) a partir de
`manifest_clean.csv` del preprocesado mínimo.

Uso (raíz del repo):
    uv run python dl/vision_baseline_kvasir/generar_manifest.py
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

from dl.vision_baseline_kvasir.manifest_entrenamiento import construir_dataframe_deduplicado
from dl.vision_baseline_kvasir.paths import raiz_proyecto


def main() -> None:
    r = raiz_proyecto()
    pred = r / "data" / "processed" / "kvasir_min_clean" / "manifest_kvasir_multiclase.csv"
    pred_r = r / "data" / "processed" / "kvasir_min_clean" / "resumen_manifest_dedup.json"
    p = argparse.ArgumentParser(description="Manifest multiclase deduplicado (Kvasir).")
    p.add_argument(
        "--manifest-clean",
        type=Path,
        default=r / "data" / "processed" / "kvasir_min_clean" / "manifest_clean.csv",
    )
    p.add_argument(
        "--hashes-eda",
        type=Path,
        default=r / "data" / "processed" / "kvasir_image_eda" / "paso3_hashes_por_archivo.csv",
    )
    p.add_argument("--salida", type=Path, default=pred, help="CSV con columnas estables para ML.")
    p.add_argument(
        "--resumen-json",
        type=Path,
        default=pred_r,
        help="Métricas de deduplicación y trazabilidad.",
    )
    a = p.parse_args()
    df, meta = construir_dataframe_deduplicado(a.manifest_clean, a.hashes_eda, raiz=r)
    a.salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.salida, index=False, encoding="utf-8-sig")
    a.resumen_json.write_text(
        json.dumps({**meta, "ruta_manifest_salida": a.salida.as_posix()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Filas en manifest de entrenamiento: {len(df)}", flush=True)
    print(f"Escrito: {a.salida}", flush=True)
    print(f"Resumen: {a.resumen_json}", flush=True)


if __name__ == "__main__":
    main()
