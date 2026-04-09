"""
Limpieza de `data/raw/cancer_final.csv` → `data/processed/cancer_final_clean.csv`.

Codificación numérica (enteros 0/1):
  - sex: man=0, woman=1
  - sof, diabetes, tenesmus, previous_rt, rectorrhagia, cancer_diagnosis: no=0, yes=1
  - Antecedentes digestivos: one-hot (cada columna digestive_* es 0 o 1; una sola activa por fila)
    digestive_no, digestive_yes, digestive_yes_colon, digestive_yes_gastric, digestive_unknown

Ejecución desde la raíz del proyecto:
    uv run python data/scripts/cleaning/cancer_final_clean.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _find_project_root() -> Path:
    """Sube directorios desde este script hasta encontrar data/raw/cancer_final.csv."""
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        candidate = p / "data" / "raw" / "cancer_final.csv"
        if candidate.is_file():
            return p
    msg = (
        "No se encontró 'data/raw/cancer_final.csv' ascendiendo desde "
        f"{here}. Coloca el script bajo …/data/scripts/ o ajusta la raíz del proyecto."
    )
    raise FileNotFoundError(msg)


PROJECT_ROOT = _find_project_root()
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "cancer_final.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "cancer_final_clean.csv"

if RAW_PATH.resolve() == OUT_PATH.resolve():
    raise ValueError("La ruta de entrada y salida es la misma; revisa las rutas.")

BINARY_COLS = [
    "sex",
    "sof",
    "diabetes",
    "tenesmus",
    "previous_rt",
    "rectorrhagia",
    "cancer_diagnosis",
]

DIGESTIVE_CATEGORIES = ("no", "yes", "yes_colon", "yes_gastric", "unknown")


def _normalize_binary(value: object) -> str:
    s = str(value).strip().lower()
    if s in {"yes", "y", "1", "true"}:
        return "yes"
    if s in {"no", "n", "0", "false"}:
        return "no"
    return s


def _clean_sex(value: object) -> str:
    s = str(value).strip().lower()
    if s in {"man", "male", "m", "hombre"}:
        return "man"
    if s in {"woman", "female", "f", "w", "mujer"}:
        return "woman"
    return s


def _clean_digestive_family_history(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    raw = str(value).strip()
    low = raw.lower()

    if low == "no":
        return "no"
    if low == "yes(colon)":
        return "yes_colon"
    if low == "yes(gastric)":
        return "yes_gastric"
    if low == "colon":
        return "yes_colon"
    if low in {"yes", "yes metav", "yes mutations", "yes stresses"}:
        return "yes"

    noise = {"unesco", "pak", "anque", "ança"}
    if low in noise:
        return "unknown"
    if raw.startswith("#") or "nombre?" in low or "nombre" in low:
        return "unknown"

    return "unknown"


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH, sep=";", encoding="latin-1")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    stats: dict[str, int] = {}
    out = df.copy()

    before = len(out)
    out = out.drop_duplicates(subset=["id"], keep="first")
    stats["duplicate_ids_removed"] = before - len(out)

    for col in BINARY_COLS:
        if col == "sex":
            out[col] = out[col].map(_clean_sex)
        else:
            out[col] = out[col].map(_normalize_binary)

    bad_sex = ~out["sex"].isin(["man", "woman"])
    stats["invalid_sex_rows"] = int(bad_sex.sum())
    for col in ["sof", "diabetes", "tenesmus", "previous_rt", "rectorrhagia", "cancer_diagnosis"]:
        bad = ~out[col].isin(["yes", "no"])
        stats[f"invalid_{col}_rows"] = int(bad.sum())

    bad_any = bad_sex.copy()
    for col in ["sof", "diabetes", "tenesmus", "previous_rt", "rectorrhagia", "cancer_diagnosis"]:
        bad_any |= ~out[col].isin(["yes", "no"])
    stats["rows_removed_invalid_classes"] = int(bad_any.sum())
    out = out.loc[~bad_any].copy()

    out["digestive_family_history_orig"] = out["digestive_family_history"]
    out["digestive_family_history"] = out["digestive_family_history"].map(_clean_digestive_family_history)
    changed_hist = out["digestive_family_history_orig"] != out["digestive_family_history"]
    stats["digestive_family_history_recoded"] = int(changed_hist.sum())
    out = out.drop(columns=["digestive_family_history_orig"])

    out["id"] = pd.to_numeric(out["id"], errors="coerce").astype("Int64")
    out["age"] = pd.to_numeric(out["age"], errors="coerce").astype("Int64")
    out["alcohol"] = pd.to_numeric(out["alcohol"], errors="coerce").astype("Int64")
    out["tobacco"] = pd.to_numeric(out["tobacco"], errors="coerce").astype("Int64")
    out["intestinal_habit"] = pd.to_numeric(out["intestinal_habit"], errors="coerce").astype("Int64")

    na_rows = out[["id", "age", "alcohol", "tobacco", "intestinal_habit"]].isna().any(axis=1)
    stats["rows_with_na_numeric"] = int(na_rows.sum())
    out = out.dropna(subset=["id", "age", "alcohol", "tobacco", "intestinal_habit"])
    stats["rows_after_drop_na_numeric"] = len(out)

    out = _label_encode_binary_and_digestive(out)

    return out, stats


def _label_encode_binary_and_digestive(out: pd.DataFrame) -> pd.DataFrame:
    """Convierte categóricas binarias a 0/1 y antecedentes familiares a one-hot 0/1."""
    df = out.copy()
    df["sex"] = df["sex"].map({"man": 0, "woman": 1}).astype("Int64")
    for col in ["sof", "diabetes", "tenesmus", "previous_rt", "rectorrhagia", "cancer_diagnosis"]:
        df[col] = df[col].map({"no": 0, "yes": 1}).astype("Int64")

    hist = df["digestive_family_history"]
    missing = ~hist.isin(list(DIGESTIVE_CATEGORIES))
    if missing.any():
        raise ValueError(
            f"Valores no esperados en digestive_family_history: {hist[missing].unique().tolist()}"
        )

    dummies = pd.DataFrame(
        {f"digestive_{c}": (hist == c).astype("Int64") for c in DIGESTIVE_CATEGORIES}
    )
    df = pd.concat([df.drop(columns=["digestive_family_history"]), dummies], axis=1)
    return df


def _verify_export(df_raw: pd.DataFrame) -> None:
    """Comprueba que el export esté codificado y no sea texto como el raw."""
    written = pd.read_csv(OUT_PATH, sep=";", encoding="utf-8")
    if len(written) != len(df_raw):
        return

    for col in ["sex", "sof", "diabetes", "tenesmus", "previous_rt", "rectorrhagia", "cancer_diagnosis"]:
        if not written[col].dropna().isin([0, 1]).all():
            raise RuntimeError(f"La columna {col} debe contener solo 0 y 1.")

    dig = [f"digestive_{c}" for c in DIGESTIVE_CATEGORIES]
    for c in dig:
        if c not in written.columns:
            raise RuntimeError(f"Falta la columna one-hot {c}.")
        if not written[c].dropna().isin([0, 1]).all():
            raise RuntimeError(f"La columna {c} debe contener solo 0 y 1.")

    if "digestive_family_history" in written.columns:
        raise RuntimeError("No debe quedar la columna textual digestive_family_history.")

    s = written[dig].sum(axis=1)
    if not (s == 1).all():
        raise RuntimeError("Cada fila debe tener exactamente un digestivo one-hot activo (suma=1).")

    if written.select_dtypes(include=["object", "string"]).shape[1] > 0:
        obj = written.select_dtypes(include=["object", "string"]).columns.tolist()
        raise RuntimeError(f"Quedan columnas de texto en el export: {obj}")


def main() -> None:
    df_raw = load_raw()
    df_clean, stats = clean(df_raw)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    abs_raw, abs_out = RAW_PATH.resolve(), OUT_PATH.resolve()
    print(f"Entrada (abs):  {abs_raw}")
    print(f"Salida (abs):   {abs_out}")

    # BOM opcional: Excel en Windows reconoce mejor UTF-8
    df_clean.to_csv(OUT_PATH, sep=";", index=False, encoding="utf-8-sig")

    _verify_export(df_raw)

    print(f"Leídas {len(df_raw)} filas desde {RAW_PATH.relative_to(PROJECT_ROOT)}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"Exportadas {len(df_clean)} filas a {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
