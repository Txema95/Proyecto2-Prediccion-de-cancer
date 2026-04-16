"""
EDA v2 para `data/processed/cancer_final_clean_v2.csv`.

Genera:
- Informe textual en consola.
- Gráficos PNG en `data/processed/eda_v2/`.

Ejecución desde la raíz del proyecto:
    uv run python data/scripts/analysis/eda_v2.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


COLUMNA_OBJETIVO = "cancer_diagnosis"


def _buscar_raiz_proyecto() -> Path:
    carpeta_actual = Path(__file__).resolve().parent
    for ruta in [carpeta_actual, *carpeta_actual.parents]:
        if (ruta / "data" / "processed" / "cancer_final_clean_v2.csv").is_file():
            return ruta
    raise FileNotFoundError(
        "No se encontró data/processed/cancer_final_clean_v2.csv. "
        "Ejecuta antes el pipeline de limpieza v2."
    )


def _imprimir_encabezado(datos: pd.DataFrame, ruta_csv: Path, carpeta_salida: Path) -> None:
    print("=" * 84)
    print("EDA v2 — Simulador de apoyo al diagnostico de cancer de colon")
    print("=" * 84)
    print(f"CSV: {ruta_csv}")
    print(f"Filas: {len(datos):,} | Columnas: {len(datos.columns)}")
    print(f"Columnas: {list(datos.columns)}")
    print(f"Salida de graficos: {carpeta_salida}\n")


def _reporte_calidad(datos: pd.DataFrame) -> None:
    print("1) Calidad de datos")
    nulos = datos.isna().sum()
    nulos = nulos[nulos > 0].sort_values(ascending=False)
    if nulos.empty:
        print("   - No hay valores nulos.")
    else:
        print("   - Valores nulos por columna:")
        print((100 * nulos / len(datos)).round(3).to_string())

    duplicadas = int(datos.duplicated().sum())
    print(f"   - Filas duplicadas exactas: {duplicadas}")
    if "id" in datos.columns:
        ids_duplicados = int(datos["id"].duplicated().sum())
        print(f"   - IDs duplicados: {ids_duplicados}")
    print()


def _balance_objetivo(datos: pd.DataFrame) -> None:
    print("2) Balance de la variable objetivo")
    conteos = datos[COLUMNA_OBJETIVO].value_counts().sort_index()
    total = int(conteos.sum())
    for clase, conteo in conteos.items():
        porcentaje = 100 * conteo / total if total else 0.0
        print(f"   - Clase {int(clase)}: {int(conteo)} ({porcentaje:.2f}%)")
    n0, n1 = int(conteos.get(0, 0)), int(conteos.get(1, 0))
    ratio = (n0 / n1) if n1 else float("inf")
    print(f"   - Ratio neg:pos = {ratio:.2f}:1")
    if ratio > 10:
        print("   - Recomendacion: usar metricas robustas a desbalance (AUC-PR, F1, recall).")
    print()


def _estadistica_descriptiva(datos: pd.DataFrame) -> None:
    print("3) Estadistica descriptiva")
    columnas_numericas = [c for c in datos.columns if pd.api.types.is_numeric_dtype(datos[c])]
    descripcion = datos[columnas_numericas].describe().T[
        ["mean", "std", "min", "25%", "50%", "75%", "max"]
    ]
    print(descripcion.round(3).to_string())
    print()

    print("   - Distribucion de codigos esperados (segun README y pipeline v2):")
    rangos_esperados = {
        "sex": (0, 1),
        "sof": (0, 1),
        "diabetes": (0, 1),
        "tenesmus": (0, 1),
        "previous_rt": (0, 1),
        "rectorrhagia": (0, 1),
        "cancer_diagnosis": (0, 1),
        "digestive_family_risk_level": (0, 3),
        "intestinal_habit": (0, 5),
        "alcohol": (0, 4),
        "tobacco": (0, 2),
    }
    for columna, (minimo, maximo) in rangos_esperados.items():
        if columna not in datos.columns:
            continue
        fuera_rango = int((~datos[columna].between(minimo, maximo)).sum())
        valores_unicos = sorted(datos[columna].dropna().unique().tolist())
        print(
            f"   - {columna}: valores={valores_unicos} | "
            f"fuera de rango [{minimo},{maximo}] = {fuera_rango}"
        )
    print()


def _graficar_balance_objetivo(datos: pd.DataFrame, carpeta_salida: Path) -> None:
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(
        x=COLUMNA_OBJETIVO,
        hue=COLUMNA_OBJETIVO,
        data=datos,
        palette=["#4e79a7", "#e15759"],
        legend=False,
    )
    ax.set_title("Balance de clases: cancer_diagnosis")
    ax.set_xlabel("cancer_diagnosis (0=no, 1=yes)")
    ax.set_ylabel("N pacientes")
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(carpeta_salida / "1_target_balance.png", dpi=140)
    plt.close()


def _graficar_distribucion_edad(datos: pd.DataFrame, carpeta_salida: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    sns.histplot(data=datos, x="age", hue=COLUMNA_OBJETIVO, bins=22, kde=True, multiple="layer")
    plt.title("Distribucion de edad por diagnostico")
    plt.xlabel("Edad")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(carpeta_salida / "2_age_distribution_by_target.png", dpi=140)
    plt.close()


def _graficar_correlaciones(datos: pd.DataFrame, carpeta_salida: Path) -> None:
    datos_numericos = datos.select_dtypes(include=["number"])
    correlaciones = datos_numericos.corr(numeric_only=True)

    plt.figure(figsize=(11, 8.5))
    sns.heatmap(
        correlaciones,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.3,
    )
    plt.title("Matriz de correlacion (Pearson)")
    plt.tight_layout()
    plt.savefig(carpeta_salida / "3_correlation_heatmap.png", dpi=140)
    plt.close()

    if COLUMNA_OBJETIVO in correlaciones.columns:
        correlacion_objetivo = (
            correlaciones[COLUMNA_OBJETIVO]
            .drop(COLUMNA_OBJETIVO)
            .sort_values(key=abs, ascending=False)
        )
        print("4) Correlacion con cancer_diagnosis (orden |r| desc)")
        print(correlacion_objetivo.round(4).to_string())
        print()


def _graficar_binarias_vs_objetivo(datos: pd.DataFrame, carpeta_salida: Path) -> None:
    variables = ["sof", "diabetes", "tenesmus", "previous_rt", "rectorrhagia", "sex"]
    filas = 2
    columnas = 3
    _, ejes = plt.subplots(filas, columnas, figsize=(14, 7))
    ejes = ejes.flatten()

    for i, variable in enumerate(variables):
        tabla = pd.crosstab(datos[variable], datos[COLUMNA_OBJETIVO], normalize="index")
        tabla = tabla.reindex(columns=[0, 1], fill_value=0)
        tabla[[1]].plot(kind="bar", ax=ejes[i], legend=False, color="#e15759")
        ejes[i].set_title(f"{variable} -> P(cancer=1)")
        ejes[i].set_xlabel(variable)
        ejes[i].set_ylabel("Probabilidad")
        ejes[i].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(carpeta_salida / "4_binary_features_vs_target.png", dpi=140)
    plt.close()


def _graficar_ordinales_vs_objetivo(datos: pd.DataFrame, carpeta_salida: Path) -> None:
    variables_ordinales = ["digestive_family_risk_level", "intestinal_habit", "alcohol", "tobacco"]
    _, ejes = plt.subplots(2, 2, figsize=(12, 8))
    ejes = ejes.flatten()

    print("5) Tasa de cancer por variables ordinales")
    for i, variable in enumerate(variables_ordinales):
        tasa = datos.groupby(variable, observed=True)[COLUMNA_OBJETIVO].mean().sort_index()
        n = datos.groupby(variable, observed=True)[COLUMNA_OBJETIVO].count().sort_index()
        tasa_porcentaje = (100 * tasa).round(2)
        print(f"   - {variable}:")
        for nivel in tasa.index:
            print(f"       nivel {nivel}: {tasa_porcentaje.loc[nivel]:>6.2f}% (n={int(n.loc[nivel])})")

        ejes[i].bar(tasa.index.astype(int), 100 * tasa.values, color="#f28e2b")
        ejes[i].set_title(f"{variable} -> % cancer")
        ejes[i].set_xlabel(variable)
        ejes[i].set_ylabel("% cancer")
        ejes[i].set_ylim(0, 100)

    print()
    plt.tight_layout()
    plt.savefig(carpeta_salida / "5_ordinal_features_vs_target.png", dpi=140)
    plt.close()


def _analisis_triada(datos: pd.DataFrame, carpeta_salida: Path) -> None:
    triada = datos.copy()
    triada["n_sintomas"] = triada["sof"] + triada["tenesmus"] + triada["rectorrhagia"]
    estadisticas = triada.groupby("n_sintomas", observed=True)[COLUMNA_OBJETIVO].agg(["mean", "count"])
    estadisticas.columns = ["tasa_cancer", "n"]

    print("6) Tríada de sintomas (sof + tenesmus + rectorrhagia)")
    for nivel in estadisticas.index:
        porcentaje = 100 * estadisticas.loc[nivel, "tasa_cancer"]
        n = int(estadisticas.loc[nivel, "n"])
        print(f"   - {int(nivel)} sintomas: {porcentaje:6.2f}% (n={n})")
    print()

    plt.figure(figsize=(7, 4.5))
    plt.bar(estadisticas.index.astype(int), 100 * estadisticas["tasa_cancer"], color="#59a14f")
    plt.title("Probabilidad empirica de cancer por numero de sintomas")
    plt.xlabel("Numero de sintomas activos (0-3)")
    plt.ylabel("% con cancer_diagnosis=1")
    plt.xticks(estadisticas.index.astype(int))
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(carpeta_salida / "6_triad_symptoms.png", dpi=140)
    plt.close()


def main() -> None:
    raiz = _buscar_raiz_proyecto()
    ruta_csv = raiz / "data" / "processed" / "cancer_final_clean_v2.csv"
    carpeta_salida = raiz / "data" / "processed" / "eda_v2"
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    datos = pd.read_csv(ruta_csv, sep=";", encoding="utf-8-sig")

    _imprimir_encabezado(datos, ruta_csv, carpeta_salida)
    _reporte_calidad(datos)
    _balance_objetivo(datos)
    _estadistica_descriptiva(datos)

    _graficar_balance_objetivo(datos, carpeta_salida)
    _graficar_distribucion_edad(datos, carpeta_salida)
    _graficar_correlaciones(datos, carpeta_salida)
    _graficar_binarias_vs_objetivo(datos, carpeta_salida)
    _graficar_ordinales_vs_objetivo(datos, carpeta_salida)
    _analisis_triada(datos, carpeta_salida)

    print("EDA completado.")
    print(f"Graficos guardados en: {carpeta_salida}")


if __name__ == "__main__":
    main()
