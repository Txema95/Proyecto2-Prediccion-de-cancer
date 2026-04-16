"""
Inspecciona el Excel original `rectesestadistica.xlsx` para volcar hojas, columnas
y frecuencias de las variables que interesan para documentar leyendas.

Coloca el fichero en: data/raw/rectesestadistica.xlsx
(acepta variaciones de nombre, ver EXCEL_NAMES).

Ejecución desde la raíz del proyecto:
    uv run python data/scripts/analysis/inspect_rectes_excel.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

EXCEL_NAMES = (
    "rectesestadistica.xlsx",
    "rectesEstadistica.xlsx",
    "RectesEstadistica.xlsx",
)


def _find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        raw = p / "data" / "raw"
        if raw.is_dir():
            return p
    raise FileNotFoundError("No se encontró data/raw")


def _find_excel(raw_dir: Path) -> Path:
    for name in EXCEL_NAMES:
        cand = raw_dir / name
        if cand.is_file():
            return cand
    found = list(raw_dir.glob("*.xlsx"))
    if len(found) == 1:
        return found[0]
    if found:
        print("Varios .xlsx en data/raw; usa uno de los nombres esperados:")
        for f in found:
            print(f"  - {f.name}")
    raise FileNotFoundError(
        f"No hay rectesestadistica.xlsx en {raw_dir}. "
        f"Copia ahí el Excel original del estudio."
    )


def _norm_col(s: str) -> str:
    return str(s).strip().lower().replace(" ", "")


def _column_tags(norm: str) -> list[str]:
    tags: list[str] = []
    if "alcohol" in norm:
        tags.append("alcohol")
    if "tabac" in norm or "tobacco" in norm:
        tags.append("tobacco")
    if "intestinal" in norm and "habit" in norm:
        tags.append("intestinal_habit")
    elif "habit" in norm or "habito" in norm or "intestinal" in norm:
        tags.append("intestinal_habit?")
    if "digest" in norm or "family" in norm or "familia" in norm or "anteced" in norm:
        tags.append("digestive_family?")
    return tags


def main() -> None:
    root = _find_project_root()
    raw = root / "data" / "raw"
    path = _find_excel(raw)
    print(f"Fichero: {path}\n")

    book = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    for sheet_name, df in book.items():
        print("=" * 72)
        print(f"Hoja: {sheet_name!r} | Filas: {len(df)} | Columnas: {len(df.columns)}")
        print("Columnas:", list(df.columns))
        if len(df) > 0:
            print("\nPrimeras filas:")
            print(df.head(3).to_string())
        print()

    # Buscar columnas parecidas a las del CSV en la primera hoja "grande" o en todas
    print("=" * 72)
    print("Búsqueda de columnas candidatas y conteos (todas las hojas)")
    for sheet_name, df in book.items():
        for orig in df.columns:
            norm = _norm_col(str(orig))
            hits = _column_tags(norm)
            if hits:
                vc = df[orig].value_counts(dropna=False).sort_index()
                print(f"\n[{sheet_name}] columna {orig!r} (candidato: {hits})")
                print(vc.to_string())


if __name__ == "__main__":
    main()
