"""Compara tipos y valores de columnas objetivo usando pandas."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]
CSV_PATH = ROOT_DIR / "data" / "raw" / "cancer_final.csv"
XLSX_PATH = ROOT_DIR / "data" / "raw" / "cancer_original.xlsx"

TARGET_COLUMNS = [
    "alcohol",
    "tobacco",
    "intestinal_habit",
    "digestive_family_history",
]

CSV_COLUMN_MAP = {
    "alcohol": "alcohol",
    "tobacco": "tobacco",
    "intestinal_habit": "intestinal_habit",
    "digestive_family_history": "digestive_family_history",
}

XLSX_COLUMN_MAP = {
    "alcohol": "alcohol",
    "tabac": "tobacco",
    "habit_intestinal": "intestinal_habit",
    "antecedents_familiars_digestius": "digestive_family_history",
}

MISSING_MARKERS = {"", "na", "n/a", "null", "none", "nan", "-", "?"}


def normalize_text(text: object) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    value = unicodedata.normalize("NFKD", str(text).strip())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_name(name: str) -> str:
    value = normalize_text(name)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def read_csv_df(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, sep=";", encoding=encoding, dtype=str)
        except UnicodeDecodeError:
            continue
    raise RuntimeError("No se pudo leer el CSV con utf-8-sig ni latin-1.")


def read_xlsx_df(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, dtype=str)


def resolve_column(df: pd.DataFrame, raw_name: str) -> str | None:
    target = normalize_name(raw_name)
    for col in df.columns:
        if normalize_name(col) == target:
            return col
    return None


def infer_type(token: str) -> str:
    if token in MISSING_MARKERS:
        return "missing"
    if re.fullmatch(r"[+-]?\d+", token):
        return "integer"
    if re.fullmatch(r"[+-]?\d+\.\d+", token):
        return "float"
    return "text"


def analyze_series(series: pd.Series) -> tuple[dict[str, int], dict[str, int]]:
    normalized = series.map(normalize_text)
    type_counts = {"text": 0, "integer": 0, "float": 0, "missing": 0}
    value_counts: dict[str, int] = {}

    for token in normalized:
        detected = infer_type(token)
        type_counts[detected] += 1
        if detected != "missing":
            value_counts[token] = value_counts.get(token, 0) + 1

    value_counts = dict(sorted(value_counts.items(), key=lambda item: item[1], reverse=True))
    return type_counts, value_counts


def print_report(column: str, source: str, series: pd.Series) -> dict[str, int]:
    type_counts, value_counts = analyze_series(series)
    print(f"\n[{source}] {column}")
    print(f"- Registros revisados: {len(series)}")
    print(f"- Valores distintos (no missing): {len(value_counts)}")
    print("- Tipos detectados:")
    for key in ("text", "integer", "float", "missing"):
        if type_counts[key] > 0:
            print(f"  - {key}: {type_counts[key]}")

    if value_counts:
        print("- Valores normalizados (conteo):")
        for val, count in value_counts.items():
            print(f"  - {val}: {count}")
    else:
        print("- Valores no vacios: (ninguno)")

    return value_counts


def compare_value_sets(column: str, csv_values: dict[str, int], xlsx_values: dict[str, int]) -> None:
    csv_set = set(csv_values.keys())
    xlsx_set = set(xlsx_values.keys())

    common = sorted(csv_set & xlsx_set)
    only_csv = sorted(csv_set - xlsx_set)
    only_xlsx = sorted(xlsx_set - csv_set)

    print(f"\n[COMPARACION] {column}")
    print(f"- Valores comunes (categorias): {len(common)}")
    for item in common:
        c_csv = csv_values.get(item, 0)
        c_xlsx = xlsx_values.get(item, 0)
        print(f"  - {item} | CSV: {c_csv} | XLSX: {c_xlsx}")
    print(f"- Solo en CSV (categorias): {len(only_csv)}")
    for item in only_csv:
        print(f"  - {item} | CSV: {csv_values[item]}")
    print(f"- Solo en XLSX (categorias): {len(only_xlsx)}")
    for item in only_xlsx:
        print(f"  - {item} | XLSX: {xlsx_values[item]}")


def get_logical_series(df: pd.DataFrame, column_map: dict[str, str]) -> dict[str, pd.Series]:
    output: dict[str, pd.Series] = {}
    for raw_col, logical_col in column_map.items():
        real_col = resolve_column(df, raw_col)
        if real_col is None:
            output[logical_col] = pd.Series([], dtype=str)
        else:
            output[logical_col] = df[real_col].astype(str)
    return output


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {CSV_PATH}")
    if not XLSX_PATH.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {XLSX_PATH}")

    csv_df = read_csv_df(CSV_PATH)
    xlsx_df = read_xlsx_df(XLSX_PATH)

    csv_series = get_logical_series(csv_df, CSV_COLUMN_MAP)
    xlsx_series = get_logical_series(xlsx_df, XLSX_COLUMN_MAP)

    print("=== COMPARACION DE TIPOS Y VALORES (PANDAS) ===")
    print(f"CSV: {CSV_PATH.name} | filas: {len(csv_df)}")
    print(f"XLSX: {XLSX_PATH.name} | filas: {len(xlsx_df)}")
    print(f"Columnas objetivo: {', '.join(TARGET_COLUMNS)}")

    for col in TARGET_COLUMNS:
        csv_values = print_report(col, "CSV", csv_series[col])
        xlsx_values = print_report(col, "XLSX", xlsx_series[col])
        compare_value_sets(col, csv_values, xlsx_values)


if __name__ == "__main__":
    main()
