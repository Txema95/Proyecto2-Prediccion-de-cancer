"""Paso 2: muestreo reproducible hasta N imagenes legibles por clase."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from configuracion import CLASES_ESPERADAS, IMAGENES_OBJETIVO_POR_CLASE, SEMILLA_PREDETERMINADA, directorio_salida, raiz_proyecto, ruta_dataset_kvasir


def construir_manifest_balanceado(
    df_inventario: pd.DataFrame,
    raiz_dataset: Path,
    imagenes_por_clase: int,
    semilla: int,
) -> tuple[pd.DataFrame, dict]:
    """
    Selecciona hasta `imagenes_por_clase` filas por clase entre las legibles.
    Si hay menos, se incluyen todas y se deja constancia en el informe.
    """
    if df_inventario.empty:
        vacio = pd.DataFrame(
            columns=["clase", "ruta_relativa", "ruta_absoluta", "seleccionado", "motivo", "orden_muestreo"]
        )
        return vacio, {"error": "inventario_vacio"}

    rng = np.random.default_rng(semilla)
    filas: list[dict] = []
    informe: dict = {
        "imagenes_objetivo_por_clase": imagenes_por_clase,
        "semilla": semilla,
        "por_clase": {},
    }

    df_ok = df_inventario[df_inventario["legible_pillow"]].copy()

    for clase in CLASES_ESPERADAS:
        sub = df_ok[df_ok["clase"] == clase].copy()
        n_disponible = len(sub)
        n_tomar = min(imagenes_por_clase, n_disponible)
        informe_clase = {
            "disponibles_legibles": int(n_disponible),
            "seleccionadas": int(n_tomar),
            "deficit_respecto_objetivo": int(max(0, imagenes_por_clase - n_disponible)),
        }
        informe["por_clase"][clase] = informe_clase

        if n_disponible == 0:
            continue

        indices = rng.choice(n_disponible, size=n_tomar, replace=False)
        elegidos = sub.iloc[indices].sort_values("ruta_relativa")

        for orden, (_, row) in enumerate(elegidos.iterrows()):
            abs_path = (raiz_dataset / row["ruta_relativa"]).resolve()
            filas.append(
                {
                    "clase": clase,
                    "ruta_relativa": row["ruta_relativa"],
                    "ruta_absoluta": str(abs_path),
                    "seleccionado": True,
                    "motivo": "muestreo_aleatorio_estratificado" if n_disponible > n_tomar else "todas_las_legibles",
                    "orden_muestreo": orden,
                }
            )

        # Registrar no seleccionados (solo si hubo recorte)
        if n_disponible > n_tomar:
            mask = np.ones(n_disponible, dtype=bool)
            mask[indices] = False
            no_sel = sub.iloc[mask]
            for _, row in no_sel.iterrows():
                abs_path = (raiz_dataset / row["ruta_relativa"]).resolve()
                filas.append(
                    {
                        "clase": clase,
                        "ruta_relativa": row["ruta_relativa"],
                        "ruta_absoluta": str(abs_path),
                        "seleccionado": False,
                        "motivo": "excluido_por_limite_muestreo",
                        "orden_muestreo": None,
                    }
                )

    manifest = pd.DataFrame(filas)
    return manifest, informe


def ejecutar_paso2_desde_csv_inventario(
    ruta_inventario_csv: Path,
    raiz_dataset: Path,
    imagenes_por_clase: int,
    semilla: int,
) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(ruta_inventario_csv)
    return construir_manifest_balanceado(df, raiz_dataset, imagenes_por_clase, semilla)


def guardar_manifest(manifest: pd.DataFrame, informe: dict, salida: Path) -> None:
    salida.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(salida / "paso2_manifest_muestreo.csv", index=False, encoding="utf-8-sig")
    with open(salida / "paso2_manifest_informe.json", "w", encoding="utf-8") as f:
        json.dump(informe, f, indent=2, ensure_ascii=False)


def main() -> None:
    raiz = raiz_proyecto()
    inv = directorio_salida(raiz) / "paso1_inventario_detalle.csv"
    if not inv.is_file():
        raise FileNotFoundError(f"Ejecute antes el paso 1 o cree: {inv}")

    dataset = ruta_dataset_kvasir(raiz)
    manifest, informe = ejecutar_paso2_desde_csv_inventario(
        inv, dataset, IMAGENES_OBJETIVO_POR_CLASE, SEMILLA_PREDETERMINADA
    )
    out = directorio_salida(raiz)
    guardar_manifest(manifest, informe, out)
    sel = manifest[manifest["seleccionado"]]
    print(f"Paso 2 listo. Manifest: {out / 'paso2_manifest_muestreo.csv'}")
    print(f"Imagenes seleccionadas (todas las clases): {len(sel)}")
    for clase in CLASES_ESPERADAS:
        n = int((sel["clase"] == clase).sum())
        print(f"  - {clase}: {n}")


if __name__ == "__main__":
    main()
