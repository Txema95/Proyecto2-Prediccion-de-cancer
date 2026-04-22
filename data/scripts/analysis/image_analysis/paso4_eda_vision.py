"""Paso 4: EDA breve de vision (resolucion, aspecto, luminancia, saturacion, contraste)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from configuracion import CLASES_ESPERADAS, directorio_salida, manifest_filtrar_seleccionado, raiz_proyecto, ruta_dataset_kvasir


def estadisticas_imagen(ruta: Path) -> dict:
    with Image.open(ruta) as img:
        img = img.convert("RGB")
        ancho, alto = img.size
        arr = np.asarray(img, dtype=np.float32) / 255.0
        # Luminancia aproximada (Rec. 601)
        luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        hsv = img.convert("HSV")
        hsv_arr = np.asarray(hsv, dtype=np.float32)
        # En Pillow H,S,V estan en 0-255
        s = hsv_arr[..., 1] / 255.0
        v = hsv_arr[..., 2] / 255.0
    aspecto = ancho / alto if alto else np.nan
    return {
        "ancho": float(ancho),
        "alto": float(alto),
        "aspecto": float(aspecto),
        "brillo_medio": float(np.mean(luma)),
        "brillo_std": float(np.std(luma)),
        "saturacion_media": float(np.mean(s)),
        "valor_v_medio": float(np.mean(v)),
    }


def ejecutar_eda(
    manifest_csv: Path,
    raiz_dataset: Path,
) -> tuple[pd.DataFrame, dict]:
    man = pd.read_csv(manifest_csv)
    sel = manifest_filtrar_seleccionado(man)
    filas: list[dict] = []
    errores: list[str] = []

    for _, row in sel.iterrows():
        clase = str(row["clase"])
        if "ruta_absoluta" in row and pd.notna(row["ruta_absoluta"]):
            ruta = Path(row["ruta_absoluta"])
        else:
            ruta = (raiz_dataset / row["ruta_relativa"]).resolve()
        if not ruta.is_file():
            errores.append(str(ruta))
            continue
        try:
            stats = estadisticas_imagen(ruta)
        except (OSError, ValueError) as exc:
            errores.append(f"{ruta}: {exc}")
            continue
        try:
            rel = str(ruta.relative_to(raiz_dataset.resolve())).replace("\\", "/")
        except ValueError:
            rel = str(ruta).replace("\\", "/")
        filas.append({"clase": clase, "ruta_relativa": rel, **stats})

    df = pd.DataFrame(filas)
    resumen: dict = {
        "n_filas": int(len(df)),
        "errores": errores[:100],
        "por_clase": {},
    }
    if df.empty:
        return df, resumen

    agg = df.groupby("clase").agg(
        n=("ruta_relativa", "count"),
        ancho_medio=("ancho", "mean"),
        alto_medio=("alto", "mean"),
        aspecto_medio=("aspecto", "mean"),
        brillo_medio=("brillo_medio", "mean"),
        brillo_std_medio=("brillo_std", "mean"),
        saturacion_media=("saturacion_media", "mean"),
        valor_v_medio=("valor_v_medio", "mean"),
    )
    resumen["por_clase"] = {k: {kk: float(vv) if hasattr(vv, "item") else vv for kk, vv in row.items()} for k, row in agg.iterrows()}

    return df, resumen


def guardar_eda(df: pd.DataFrame, resumen: dict, salida: Path, raiz_dataset: Path, max_muestra_montaje: int) -> None:
    salida.mkdir(parents=True, exist_ok=True)
    df.to_csv(salida / "paso4_metricas_por_imagen.csv", index=False, encoding="utf-8-sig")
    agg_path = salida / "paso4_resumen_por_clase.csv"
    if not df.empty:
        df.groupby("clase").agg(
            n=("ruta_relativa", "count"),
            ancho_medio=("ancho", "mean"),
            ancho_std=("ancho", "std"),
            alto_medio=("alto", "mean"),
            alto_std=("alto", "std"),
            aspecto_medio=("aspecto", "mean"),
            brillo_medio=("brillo_medio", "mean"),
            brillo_std_medio=("brillo_std", "mean"),
            saturacion_media=("saturacion_media", "mean"),
            valor_v_medio=("valor_v_medio", "mean"),
        ).reindex(CLASES_ESPERADAS).to_csv(agg_path, encoding="utf-8-sig")
    with open(salida / "paso4_eda_resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    fig_root = salida / "paso4_figuras"
    fig_root.mkdir(parents=True, exist_ok=True)

    if df.empty:
        return

    metricas = [
        ("ancho", "Ancho (px)"),
        ("alto", "Alto (px)"),
        ("aspecto", "Relacion de aspecto"),
        ("brillo_medio", "Brillo medio (luma)"),
        ("brillo_std", "Contraste local (std luma)"),
        ("saturacion_media", "Saturacion media (HSV)"),
    ]

    for col, titulo in metricas:
        plt.figure(figsize=(8, 4))
        for clase in CLASES_ESPERADAS:
            sub = df[df["clase"] == clase][col]
            if sub.empty:
                continue
            plt.hist(sub, bins=30, alpha=0.45, label=clase, density=True)
        plt.title(titulo)
        plt.xlabel(titulo)
        plt.ylabel("Densidad")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(fig_root / f"hist_{col}.png", dpi=120)
        plt.close()

    n_cols = min(4, max_muestra_montaje)
    for clase in CLASES_ESPERADAS:
        sub = df[df["clase"] == clase]
        if sub.empty:
            continue
        muestra = sub.head(max_muestra_montaje)
        rutas = []
        for _, row in muestra.iterrows():
            ruta = (raiz_dataset / row["ruta_relativa"]).resolve()
            rutas.append(ruta)
        n = len(rutas)
        n_cols_eff = max(1, min(n_cols, n))
        n_rows = int(np.ceil(n / n_cols_eff))
        fig, axes = plt.subplots(n_rows, n_cols_eff, figsize=(2.2 * n_cols_eff, 2.2 * n_rows))
        axes_arr = np.atleast_1d(axes).flatten()
        for ax in axes_arr:
            ax.axis("off")
        for i, ruta in enumerate(rutas):
            if i >= len(axes_arr):
                break
            try:
                with Image.open(ruta) as im:
                    im = im.convert("RGB")
                    axes_arr[i].imshow(np.asarray(im))
            except OSError:
                axes_arr[i].text(0.1, 0.5, "Error lectura", transform=axes_arr[i].transAxes)
        plt.suptitle(f"Muestra: {clase}", fontsize=10)
        plt.tight_layout()
        plt.savefig(fig_root / f"muestras_{clase.replace('-', '_')}.png", dpi=120)
        plt.close(fig)


def main() -> None:
    raiz = raiz_proyecto()
    out = directorio_salida(raiz)
    manifest = out / "paso2_manifest_muestreo.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Ejecute antes el paso 2 o cree: {manifest}")

    dataset = ruta_dataset_kvasir(raiz)
    df, res = ejecutar_eda(manifest, dataset)
    guardar_eda(df, res, out, dataset, max_muestra_montaje=8)
    print(f"Paso 4 listo. Metricas por imagen: {out / 'paso4_metricas_por_imagen.csv'}")
    print(f"Figuras en: {out / 'paso4_figuras'}")


if __name__ == "__main__":
    main()
