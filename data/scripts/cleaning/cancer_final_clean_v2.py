"""
Limpieza v2: `data/raw/cancer_final.csv` → `data/processed/cancer_final_clean_v2.csv`.

- Label encoding (0/1): sex, sof, diabetes, tenesmus, previous_rt, rectorrhagia,
  cancer_diagnosis (no=0, yes=1; woman=1, man=0).
- Columna `digestive_family_risk_level` (entero 0-3) según antecedente digestivo
  (deducido internamente tras normalizar `digestive_family_history`):
      0 = No_Risk
      1 = Unknown / Noise
      2 = Medium_Risk
      3 = High_Risk
  La columna textual `digestive_family_history` no se exporta (solo el nivel 0-3).

Ejecución desde la raíz del proyecto:
    uv run python data/scripts/cleaning/cancer_final_clean_v2.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

def _find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        candidate = p / "data" / "raw" / "cancer_final.csv"
        if candidate.is_file():
            return p
    raise FileNotFoundError(
        f"No se encontró data/raw/cancer_final.csv desde {here}"
    )


PROJECT_ROOT = _find_project_root()
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "cancer_final.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "cancer_final_clean_v2.csv"

if RAW_PATH.resolve() == OUT_PATH.resolve():
    raise ValueError("Ruta de entrada y salida coincide.")

BINARY_COLS = [
    "sex",
    "sof",
    "diabetes",
    "tenesmus",
    "previous_rt",
    "rectorrhagia",
    "cancer_diagnosis",
]


def _repair_utf8_mojibake_if_latin1_wrapped(value: object) -> object:
    """
    Si el CSV guarda UTF-8 pero se leyó como Latin-1, secuencias como c3+a7 (ç)
    aparecen como 'anÃ§a'. latin-1 -> bytes -> utf-8 recupera el texto correcto.
    Si la cadena ya es Latin-1 válido sin mezcla, suele fallar el decode y se devuelve igual.
    """
    if pd.isna(value):
        return value
    s = str(value).strip()
    if not s:
        return s
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return s


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

    noise = {"unesco", "pak", "anque", "ança", "anã§a"}
    if low in noise:
        return "unknown"
    if raw.startswith("#") or "nombre?" in low or "nombre" in low:
        return "unknown"

    return "unknown"


# Según categoría interna tras _clean_digestive_family_history
_DIGESTIVE_CLEAN_TO_RISK: dict[str, int] = {
    "no": 0,  # No_Risk
    "unknown": 1,  # Unknown / Noise
    "yes": 2,  # Medium_Risk (antecedente genérico)
    "yes_gastric": 2,  # Medium_Risk
    "yes_colon": 3,  # High_Risk
}


def _digestive_family_risk_level_from_clean(clean: object) -> int:
    if pd.isna(clean):
        return 1
    key = str(clean).strip().lower()
    return _DIGESTIVE_CLEAN_TO_RISK.get(key, 1)


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH, sep=";", encoding="latin-1")


def _base_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
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

    out["digestive_family_history"] = out["digestive_family_history"].map(
        _repair_utf8_mojibake_if_latin1_wrapped
    )

    # Texto digestive corregido; categoría interna para digestive_family_risk_level
    out["_digestive_clean"] = out["digestive_family_history"].map(
        _clean_digestive_family_history
    )

    out["id"] = pd.to_numeric(out["id"], errors="coerce").astype("Int64")
    out["age"] = pd.to_numeric(out["age"], errors="coerce").astype("Int64")
    out["alcohol"] = pd.to_numeric(out["alcohol"], errors="coerce").astype("Int64")
    out["tobacco"] = pd.to_numeric(out["tobacco"], errors="coerce").astype("Int64")
    out["intestinal_habit"] = pd.to_numeric(out["intestinal_habit"], errors="coerce").astype(
        "Int64"
    )

    na_rows = out[["id", "age", "alcohol", "tobacco", "intestinal_habit"]].isna().any(axis=1)
    stats["rows_with_na_numeric"] = int(na_rows.sum())
    out = out.dropna(subset=["id", "age", "alcohol", "tobacco", "intestinal_habit"])
    stats["rows_after_drop_na_numeric"] = len(out)

    return out, stats


def build_v2(df: pd.DataFrame, stats: dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    df["digestive_family_risk_level"] = df["_digestive_clean"].map(
        _digestive_family_risk_level_from_clean
    ).astype("Int64")

    df = df.drop(columns=["_digestive_clean", "digestive_family_history"])

    df["sex"] = df["sex"].map({"man": 0, "woman": 1}).astype("Int64")
    for col in ["sof", "diabetes", "tenesmus", "previous_rt", "rectorrhagia", "cancer_diagnosis"]:
        df[col] = df[col].map({"no": 0, "yes": 1}).astype("Int64")

    base = [
        "id",
        "age",
        "sex",
        "sof",
        "diabetes",
        "tenesmus",
        "previous_rt",
        "rectorrhagia",
        "cancer_diagnosis",
        "digestive_family_risk_level",
        "intestinal_habit",
        "alcohol",
        "tobacco",
    ]
    df = df[base]

    return df


def _verify_v2(written: pd.DataFrame) -> None:
    for col in ["sex", "sof", "diabetes", "tenesmus", "previous_rt", "rectorrhagia", "cancer_diagnosis"]:
        if not written[col].dropna().isin([0, 1]).all():
            raise RuntimeError(f"{col} debe ser solo 0 o 1.")
    if "digestive_family_risk_level" not in written.columns:
        raise RuntimeError("Falta digestive_family_risk_level.")
    if not written["digestive_family_risk_level"].dropna().isin([0, 1, 2, 3]).all():
        raise RuntimeError("digestive_family_risk_level debe ser 0, 1, 2 o 3.")
    if "digestive_family_history" in written.columns:
        raise RuntimeError("No se debe exportar la columna digestive_family_history.")
    for col in ["intestinal_habit", "alcohol", "tobacco"]:
        if col not in written.columns:
            raise RuntimeError(f"Debe conservarse la columna {col}.")
    if "_digestive_clean" in written.columns:
        raise RuntimeError("No debe exportarse la columna interna _digestive_clean.")
    obj_cols = set(written.select_dtypes(include=["object", "string"]).columns.tolist())
    if obj_cols:
        raise RuntimeError(f"No deben quedar columnas de texto: {obj_cols}")


def main() -> None:
    df_raw = load_raw()
    df_base, stats = _base_clean(df_raw)
    df_out = build_v2(df_base, stats)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Entrada (abs):  {RAW_PATH.resolve()}")
    print(f"Salida (abs):   {OUT_PATH.resolve()}")

    df_out.to_csv(OUT_PATH, sep=";", index=False, encoding="utf-8-sig")

    written = pd.read_csv(OUT_PATH, sep=";", encoding="utf-8-sig")
    _verify_v2(written)

    print(f"Leídas {len(df_raw)} filas")
    print(f"Exportadas {len(df_out)} filas a {OUT_PATH.relative_to(PROJECT_ROOT)}")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
