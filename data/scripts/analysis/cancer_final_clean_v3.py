"""
Limpieza v3: `data/raw/cancer_final.csv` → `data/processed/cancer_final_clean_v3.csv`.

- Columnas binarias y sexo: texto → enteros (sex: man=0, woman=1; resto no/yes → 0/1).
- `alcohol`, `tobacco`, `intestinal_habit`, `age`, `id`: numéricos (no label encoding categórico).
- `digestive_family_history`: 4 categorías con códigos enteros 0–3 (cuatro clases → índices 0..3).
      0 = no
      1 = yes (incluye yes metav, yes mutations, yes stresses y "yes" genérico)
      2 = yes_colon (p. ej. yes(colon), colon)
      3 = yes_gastric (p. ej. yes(gastric))
  Cualquier otro valor: fila descartada.

Ejecución desde la raíz del proyecto:
    uv run python data/scripts/analysis/cancer_final_clean_v3.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Raíz del repo (donde existe data/raw/cancer_final.csv)
_here = Path(__file__).resolve().parent
PROJECT_ROOT = next(p for p in [_here, *_here.parents] if (p / "data" / "raw" / "cancer_final.csv").is_file())
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "cancer_final.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "cancer_final_clean_v3.csv"

BINARY_COLS = [
    "sof",
    "diabetes",
    "tenesmus",
    "previous_rt",
    "rectorrhagia",
    "cancer_diagnosis",
]


def _repair_utf8_mojibake_if_latin1_wrapped(value: object) -> object:
    if pd.isna(value):
        return value
    s = str(value).strip()
    if not s:
        return s
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return s


def _digestive_family_history_to_code(value: object) -> int | None:
    """Devuelve 0–3 o None si el valor se descarta (no entra en las 4 categorías)."""
    if pd.isna(value):
        return None
    raw = str(value).strip()
    raw = _repair_utf8_mojibake_if_latin1_wrapped(raw)
    low = str(raw).strip().lower()

    if low == "no":
        return 0
    if low in {"yes", "yes metav", "yes mutations", "yes stresses"}:
        return 1
    if low in {"yes(colon)", "colon"}:
        return 2
    if low == "yes(gastric)":
        return 3
    return None


def main() -> None:
    if not RAW_PATH.is_file():
        raise FileNotFoundError(RAW_PATH)

    df = pd.read_csv(RAW_PATH, sep=";", encoding="latin-1")
    n_in = len(df)

    df = df.drop_duplicates(subset=["id"], keep="first")

    df["digestive_family_history"] = df["digestive_family_history"].map(
        _repair_utf8_mojibake_if_latin1_wrapped
    )
    df["digestive_family_history"] = df["digestive_family_history"].map(_digestive_family_history_to_code)
    mask_digestive_ok = df["digestive_family_history"].notna()
    n_drop_digestive = int((~mask_digestive_ok).sum())
    df = df.loc[mask_digestive_ok].copy()
    df["digestive_family_history"] = df["digestive_family_history"].astype("Int64")

    df["sex"] = df["sex"].map(lambda x: str(x).strip().lower())
    df["sex"] = df["sex"].replace(
        {"man": "man", "male": "man", "m": "man", "hombre": "man",
         "woman": "woman", "female": "woman", "f": "woman", "w": "woman", "mujer": "woman"}
    )
    for col in BINARY_COLS:
        s = df[col].map(lambda x: str(x).strip().lower())
        s = s.replace({"yes": "yes", "y": "yes", "1": "yes", "true": "yes",
                       "no": "no", "n": "no", "0": "no", "false": "no"})
        df[col] = s

    mask_sex = df["sex"].isin(["man", "woman"])
    mask_bin = pd.Series(True, index=df.index)
    for col in BINARY_COLS:
        mask_bin &= df[col].isin(["yes", "no"])
    mask_ok = mask_sex & mask_bin
    n_drop_invalid = int((~mask_ok).sum())
    df = df.loc[mask_ok].copy()

    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df["age"] = pd.to_numeric(df["age"], errors="coerce").astype("Int64")
    df["alcohol"] = pd.to_numeric(df["alcohol"], errors="coerce").astype("Int64")
    df["tobacco"] = pd.to_numeric(df["tobacco"], errors="coerce").astype("Int64")
    df["intestinal_habit"] = pd.to_numeric(df["intestinal_habit"], errors="coerce").astype("Int64")

    mask_num = df[["id", "age", "alcohol", "tobacco", "intestinal_habit"]].notna().all(axis=1)
    n_drop_na_num = int((~mask_num).sum())
    df = df.loc[mask_num].copy()

    df["sex"] = df["sex"].map({"man": 0, "woman": 1}).astype("Int64")
    for col in BINARY_COLS:
        df[col] = df[col].map({"no": 0, "yes": 1}).astype("Int64")

    cols_order = [
        "id",
        "age",
        "sex",
        "sof",
        "diabetes",
        "tenesmus",
        "previous_rt",
        "rectorrhagia",
        "cancer_diagnosis",
        "digestive_family_history",
        "intestinal_habit",
        "alcohol",
        "tobacco",
    ]
    df = df[cols_order]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, sep=";", index=False, encoding="utf-8-sig")

    # Comprobaciones rápidas
    check = pd.read_csv(OUT_PATH, sep=";", encoding="utf-8-sig")
    obj_cols = check.select_dtypes(include=["object", "string"]).columns.tolist()
    if obj_cols:
        raise RuntimeError(f"Quedan columnas texto: {obj_cols}")
    if not check["digestive_family_history"].dropna().isin([0, 1, 2, 3]).all():
        raise RuntimeError("digestive_family_history debe ser solo 0, 1, 2 o 3.")
    for c in ["sex", *BINARY_COLS]:
        if not check[c].dropna().isin([0, 1]).all():
            raise RuntimeError(f"{c} debe ser solo 0 o 1.")

    print(f"Entrada:  {RAW_PATH.resolve()} ({n_in} filas)")
    print(f"Salida:   {OUT_PATH.resolve()} ({len(df)} filas)")
    print(f"  filas eliminadas por digestive_family_history no válido: {n_drop_digestive}")
    print(f"  filas eliminadas por sex/binarios inválidos: {n_drop_invalid}")
    print(f"  filas eliminadas por NaN en numéricos obligatorios: {n_drop_na_num}")
    print("digestive_family_history: 0=no, 1=yes, 2=yes_colon, 3=yes_gastric")


if __name__ == "__main__":
    main()
