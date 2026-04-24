from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from .constantes import CLASES_ORDEN, clase_a_indice
from .paths import raiz_proyecto


def _sin_tildes(texto: str) -> str:
    """Mantiene nombres de fichero seguros y ASCII básico para `image_id`."""
    nf = unicodedata.normalize("NFD", texto)
    return "".join(c for c in nf if unicodedata.category(c) != "Mn")


def _slug_para_image_id(nombre: str) -> str:
    base = Path(nombre).stem
    s = _sin_tildes(base)
    s = s.lower()
    s = re.sub(r"[^a-z0-9-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "id"


def construir_dataframe_deduplicado(
    ruta_manifest_clean: Path,
    ruta_hashes_eda: Path,
    raiz: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Une `manifest_clean.csv` (salida del preprocesado) con `paso3_hashes` del EDA,
    y deja un único `filepath` por hash MD5 (ficheros idénticos en bruto).
    """
    if raiz is None:
        raiz = raiz_proyecto()
    ruta_manifest_clean = Path(ruta_manifest_clean).resolve()
    ruta_hashes_eda = Path(ruta_hashes_eda).resolve()
    if not ruta_manifest_clean.is_file():
        raise FileNotFoundError(f"Falta manifest clean: {ruta_manifest_clean}")
    if not ruta_hashes_eda.is_file():
        raise FileNotFoundError(f"Falta CSV de hashes EDA: {ruta_hashes_eda}")

    m = pd.read_csv(ruta_manifest_clean, encoding="utf-8-sig")
    h = pd.read_csv(ruta_hashes_eda, encoding="utf-8-sig")
    m["ruta_abs_entrada"] = m["ruta_entrada"].map(lambda s: str(Path(s).resolve()))
    h["ruta_absoluta_n"] = h["ruta_absoluta"].map(lambda s: str(Path(s).resolve()))
    # La metadata del cleaning puede excluir comillas en rutas, pero unimos por abs normalizado
    m = m.drop_duplicates(subset=["ruta_abs_entrada"], keep="first")
    m = m.merge(
        h[["ruta_absoluta_n", "md5"]], left_on="ruta_abs_entrada", right_on="ruta_absoluta_n", how="left"
    )
    filas_faltan = m["md5"].isna().sum()
    if filas_faltan:
        # Intentar alinear con normalización mínima de espacios en rutas
        raise ValueError(
            f"Filas de manifest sin MD5 (no encontradas en {ruta_hashes_eda.name}): {filas_faltan}. "
            "Asegúrate de que el manifest_clean corresponde a la misma muestreo/EDA."
        )
    m["ruta_salida_abs"] = m["ruta_salida"].map(lambda s: str(Path(s).resolve()))

    m["_n_md5"] = m.groupby("md5")["ruta_salida"].transform("count")
    m["_ruta_min"] = m.groupby("md5")["ruta_salida"].transform("min")
    sel = m[m["ruta_salida"].astype(str) == m["_ruta_min"].astype(str)].copy()
    sel = sel.sort_values(["md5", "ruta_salida"]).drop_duplicates(subset=["md5"], keep="first")
    n_quitadas = int(len(m) - len(sel))
    n_grupos_dup = int((m.groupby("md5").size() > 1).sum()) if not m.empty else 0
    meta = {
        "filas_manifest_clean": int(len(m)),
        "grupos_md5_con_mas_de_un_archivo": n_grupos_dup,
        "filas_eliminadas_por_dedup_md5": n_quitadas,
        "filas_finales": int(len(sel)),
    }
    if sel.empty:
        vacio = pd.DataFrame(columns=["filepath", "label", "source", "group_id", "image_id"])
        return vacio, {**meta, "clases_esperadas": list(CLASES_ORDEN)}

    filas: list[dict[str, str | int]] = []
    for _, r in sel.iterrows():
        clase = str(r["clase"])
        ruta_sal = Path(r["ruta_salida_abs"])
        try:
            filepath_rel = ruta_sal.resolve().relative_to(raiz.resolve())
        except ValueError as e:
            raise ValueError(
                f"La ruta {ruta_sal} no está bajo la raiz del repo {raiz}."
            ) from e
        fp_posix = filepath_rel.as_posix()
        ind = clase_a_indice(clase)
        es_dup = r["_n_md5"] > 1
        stem_slug = _slug_para_image_id(r.get("nombre_archivo", ruta_sal.name))
        if es_dup:
            g_id = f"md5_{r['md5']}"
            img_id = f"{g_id}__{stem_slug}"
        else:
            g_id = f"img_{stem_slug}"
            img_id = stem_slug
        filas.append(
            {
                "filepath": fp_posix,
                "label": str(ind),
                "source": f"kvasir_{clase.replace('-', '_')}",
                "group_id": g_id,
                "image_id": img_id,
            }
        )
    return pd.DataFrame(filas), {**meta, "clases_esperadas": list(CLASES_ORDEN)}
