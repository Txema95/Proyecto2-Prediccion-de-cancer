"""
Informe de tipos y valores por columna de `data/raw/cancer_final.csv`.

Ejecución desde la raíz del proyecto:
    uv run python data/scripts/analysis/cancer_final_column_types.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / "data" / "raw" / "cancer_final.csv").is_file():
            return p
    raise FileNotFoundError(f"No se encontró data/raw/cancer_final.csv desde {here}")


def _classify(series: pd.Series) -> str:
    s = series.dropna()
    if s.empty:
        return "vacia"
    if pd.api.types.is_integer_dtype(series):
        return "entero"
    if pd.api.types.is_float_dtype(series):
        return "decimal"
    if pd.api.types.is_bool_dtype(series):
        return "booleano"
    # Tras read_csv, casi todo es object
    try:
        pd.to_numeric(s, errors="raise")
        return "texto coercible a numerico"
    except (ValueError, TypeError):
        pass
    n = s.nunique()
    if n <= 30:
        return f"categorico texto ({n} valores distintos)"
    return f"texto ({n} valores distintos)"


COUNT_BY_VALUE_COLS = ("alcohol", "tobacco", "intestinal_habit", "digestive_family_history")


def main() -> None:
    root = _find_project_root()
    path = root / "data" / "raw" / "cancer_final.csv"
    df = pd.read_csv(path, sep=";", encoding="latin-1")

    print(f"Fichero: {path}")
    print(f"Filas: {len(df):,} | Columnas: {len(df.columns)}\n")
    print(f"{'Columna':<28} {'dtype pandas':<12} {'nulos':>6} {'unicos':>8}  Clasificacion / valores")
    print("-" * 100)

    for col in df.columns:
        s = df[col]
        n_null = int(s.isna().sum())
        n_unique = int(s.nunique(dropna=True))
        kind = _classify(s)

        if n_unique <= 20 and n_unique > 0:
            vals = sorted(s.dropna().astype(str).unique().tolist(), key=str.lower)
            extra = f"valores: {vals}"
        elif pd.api.types.is_numeric_dtype(s):
            extra = f"min={s.min()} max={s.max()}"
        else:
            examples = s.dropna().astype(str).head(5).tolist()
            extra = f"ejemplos: {examples}"

        print(f"{col:<28} {str(s.dtype):<12} {n_null:>6} {n_unique:>8}  {kind}")
        print(f"{'':28} {'':12} {'':6} {'':8}  {extra}")

        if col in COUNT_BY_VALUE_COLS:
            n = len(s.dropna())
            print(f"{'':28} {'':12} {'':6} {'':8}  conteo por valor (n validos={n}):")
            for val, cnt in s.value_counts().sort_index().items():
                pct = 100.0 * cnt / n if n else 0.0
                print(f"{'':28} {'':12} {'':6} {'':8}    {val!r}: {cnt} ({pct:.2f}%)")
        print()


if __name__ == "__main__":
    main()
