"""
Ejecuta los pasos 1 a 4 sobre Kvasir v2 (inventario, manifest, duplicados, EDA).

Uso (desde la raiz del proyecto, con uv o el Python del entorno):
    uv run python data/scripts/analysis/image_analysis/ejecutar_analisis.py
    uv run python data/scripts/analysis/image_analysis/ejecutar_analisis.py --dataset-root "C:/ruta/kvasir-dataset-v2"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite importar modulos del mismo directorio al ejecutar como script.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from configuracion import (  # noqa: E402
    IMAGENES_OBJETIVO_POR_CLASE,
    SEMILLA_PREDETERMINADA,
    directorio_salida,
    manifest_filtrar_seleccionado,
    raiz_proyecto,
    ruta_dataset_kvasir,
)
from paso1_inventario import ejecutar_inventario, guardar_inventario  # noqa: E402
from paso2_balance import ejecutar_paso2_desde_csv_inventario, guardar_manifest  # noqa: E402
from paso3_duplicados import ejecutar_duplicados, guardar_duplicados  # noqa: E402
from paso4_eda_vision import ejecutar_eda, guardar_eda  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Analisis Kvasir: pasos 1-4")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Carpeta raiz del Kvasir v2 (por defecto data/raw/kvasir-dataset-v2 bajo la raiz del repo).",
    )
    parser.add_argument("--semilla", type=int, default=SEMILLA_PREDETERMINADA)
    parser.add_argument("--por-clase", type=int, default=IMAGENES_OBJETIVO_POR_CLASE)
    parser.add_argument("--umbral-hamming", type=int, default=8, help="dHash maximo para unir pares (0 = solo identicos).")
    parser.add_argument("--max-pares", type=int, default=2000, help="Tope de pares cercanos listados en CSV.")
    parser.add_argument("--montaje", type=int, default=8, help="Imagenes por clase en la rejilla de muestras (paso 4).")
    args = parser.parse_args()

    raiz = raiz_proyecto()
    dataset = args.dataset_root if args.dataset_root is not None else ruta_dataset_kvasir(raiz)
    salida = directorio_salida(raiz)

    print("=== Paso 1: inventario ===")
    df_inv, resumen_inv = ejecutar_inventario(dataset)
    guardar_inventario(df_inv, resumen_inv, salida)
    print(f"  Detalle: {salida / 'paso1_inventario_detalle.csv'}")
    print(f"  Total archivos: {len(df_inv)}")
    if resumen_inv.get("carpetas_faltantes"):
        print(f"  Aviso: carpetas faltantes: {resumen_inv['carpetas_faltantes']}")

    print("=== Paso 2: muestreo por clase ===")
    inv_csv = salida / "paso1_inventario_detalle.csv"
    manifest, informe_m = ejecutar_paso2_desde_csv_inventario(inv_csv, dataset, args.por_clase, args.semilla)
    guardar_manifest(manifest, informe_m, salida)
    n_sel = len(manifest_filtrar_seleccionado(manifest))
    print(f"  Manifest: {salida / 'paso2_manifest_muestreo.csv'} | seleccionadas: {n_sel}")

    print("=== Paso 3: duplicados (MD5 + dHash) ===")
    manifest_csv = salida / "paso2_manifest_muestreo.csv"
    df_h, df_p, res_dup = ejecutar_duplicados(
        manifest_csv, dataset, umbral_hamming=args.umbral_hamming, max_pares_reporte=args.max_pares
    )
    guardar_duplicados(df_h, df_p, res_dup, salida)
    print(f"  Hashes: {salida / 'paso3_hashes_por_archivo.csv'}")
    print(f"  Grupos MD5 duplicados: {res_dup.get('grupos_md5_duplicados', 0)}")

    print("=== Paso 4: EDA vision ===")
    df_eda, res_eda = ejecutar_eda(manifest_csv, dataset)
    guardar_eda(df_eda, res_eda, salida, dataset, max_muestra_montaje=args.montaje)
    print(f"  Metricas: {salida / 'paso4_metricas_por_imagen.csv'}")
    print(f"  Figuras: {salida / 'paso4_figuras'}")

    print("Listo. Salida en:", salida)


if __name__ == "__main__":
    main()
