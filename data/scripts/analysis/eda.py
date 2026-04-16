"""
Exploracion de datos (EDA) sobre `data/processed/cancer_final_clean_v2.csv`.

Genera PNG en `data/processed/eda/` y resume resultados en consola (en español).

Ejecución desde la raíz del proyecto:
    uv run python data/scripts/analysis/eda.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _buscar_raiz_proyecto() -> Path:
    carpeta_actual = Path(__file__).resolve().parent
    for ruta in [carpeta_actual, *carpeta_actual.parents]:
        if (ruta / "data" / "processed" / "cancer_final_clean_v2.csv").is_file():
            return ruta
    raise FileNotFoundError(
        "No se encontró data/processed/cancer_final_clean_v2.csv. "
        "Ejecuta antes: uv run python data/scripts/cleaning/cancer_final_clean_v2.py"
    )


def main() -> None:
    raiz = _buscar_raiz_proyecto()
    ruta_csv = raiz / "data" / "processed" / "cancer_final_clean_v2.csv"
    carpeta_salida = raiz / "data" / "processed" / "eda"
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    datos = pd.read_csv(ruta_csv, sep=";", encoding="utf-8-sig")
    sns.set_theme(style="whitegrid")

    print("=" * 72)
    print(f"Filas: {len(datos)} | Columnas: {list(datos.columns)}")
    print(f"Gráficos en: {carpeta_salida}\n")

    # ----- 1. Correlación (heatmap) -----
    datos_numericos = datos.select_dtypes(include=["number"])
    correlaciones = datos_numericos.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlaciones, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("1. Matriz de correlacion de Pearson (todas las variables numericas)")
    plt.tight_layout()
    plt.savefig(carpeta_salida / "1_correlacion_heatmap.png", dpi=120)
    plt.close()

    correlacion_objetivo = (
        correlaciones["cancer_diagnosis"].drop("cancer_diagnosis").sort_values(key=abs, ascending=False)
    )
    print("1. Correlacion con cancer_diagnosis (orden |r| descendente):")
    print(correlacion_objetivo.to_string())
    print()

    # ----- 2. Perfil positivo vs negativo -----
    negativos = datos[datos["cancer_diagnosis"] == 0]
    positivos = datos[datos["cancer_diagnosis"] == 1]
    print("2. Perfil: media / proporciones (neg=0 vs pos=1)")
    print(
        f"   Edad media: sin cancer {negativos['age'].mean():.2f} | con cancer {positivos['age'].mean():.2f}"
    )
    print(
        "   Alcohol medio: sin cancer "
        f"{negativos['alcohol'].mean():.3f} | con cancer {positivos['alcohol'].mean():.3f}"
    )
    print(
        f"   Tabaco medio: sin cancer {negativos['tobacco'].mean():.3f} | con cancer {positivos['tobacco'].mean():.3f}"
    )
    print(
        f"   % con sangre en heces (sof=1): sin cancer {100 * negativos['sof'].mean():.2f}% | "
        f"con cancer {100 * positivos['sof'].mean():.2f}%"
    )
    print(
        f"   % tenesmus: sin cancer {100 * negativos['tenesmus'].mean():.2f}% | "
        f"con cancer {100 * positivos['tenesmus'].mean():.2f}%"
    )
    print(
        f"   % rectorragia: sin cancer {100 * negativos['rectorrhagia'].mean():.2f}% | "
        f"con cancer {100 * positivos['rectorrhagia'].mean():.2f}%\n"
    )

    # ----- 3. digestive_family_risk_level (barras apiladas) -----
    tabla_cruzada = pd.crosstab(datos["digestive_family_risk_level"], datos["cancer_diagnosis"])
    tabla_cruzada = tabla_cruzada.reindex(columns=[0, 1], fill_value=0)
    eje = tabla_cruzada.plot(
        kind="bar",
        stacked=True,
        figsize=(8, 5),
        color=["#2ecc71", "#e74c3c"],
        edgecolor="white",
    )
    eje.set_xlabel("digestive_family_risk_level (0=No, 1=Unknown, 2=Med, 3=High)")
    eje.set_ylabel("N pacientes")
    eje.set_title("3. Riesgo digestivo vs diagnostico (barras apiladas)")
    eje.legend(["Sin cancer (0)", "Con cancer (1)"])
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(carpeta_salida / "3_riesgo_digestivo_apilado.png", dpi=120)
    plt.close()

    print("3. Proporcion de cancer por nivel de riesgo digestivo:")
    proporcion = tabla_cruzada.div(tabla_cruzada.sum(axis=1), axis=0)
    print(proporcion)
    print()

    # ----- 4. Tríada de síntomas -----
    datos_sintomas = datos.copy()
    datos_sintomas["n_sintomas"] = (
        datos_sintomas["sof"] + datos_sintomas["tenesmus"] + datos_sintomas["rectorrhagia"]
    )
    tasa = datos_sintomas.groupby("n_sintomas", observed=True)["cancer_diagnosis"].agg(["mean", "count"])
    tasa.columns = ["tasa_cancer", "n"]
    print("4. Tríada (sof+tenesmus+rectorrhagia): tasa de cancer por nº de síntomas activos")
    print(tasa)
    print()

    plt.figure(figsize=(7, 5))
    eje_x = tasa.index.astype(int)
    plt.bar(eje_x, 100 * tasa["tasa_cancer"], color="#3498db", edgecolor="white")
    plt.xlabel("N síntomas (de 3)")
    plt.ylabel("% con cancer_diagnosis = 1")
    plt.title("4. Probabilidad empirica de cancer segun síntomas acumulados")
    plt.xticks(eje_x)
    plt.tight_layout()
    plt.savefig(carpeta_salida / "4_triada_sintomas.png", dpi=120)
    plt.close()

    # ----- 5. Desequilibrio de clase -----
    conteo_clases = datos["cancer_diagnosis"].value_counts().sort_index()
    n0, n1 = int(conteo_clases.get(0, 0)), int(conteo_clases.get(1, 0))
    ratio = n0 / n1 if n1 else float("inf")
    print("5. Balance de clases (cancer_diagnosis)")
    print(f"   Clase 0: {n0} | Clase 1: {n1} | ratio neg:pos = {ratio:.2f} : 1")
    if ratio > 10:
        print(
            "   >>> Desequilibrio fuerte (ratio > 10). Valorar class_weight, "
            "AUC-PR, SMOTE u otras técnicas.\n"
        )
    else:
        print("   Ratio moderado o aceptable para muchos algoritmos (revisión de métricas igualmente).\n")

    # ----- 6. Histogramas edad por sexo -----
    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=datos,
        x="age",
        hue="sex",
        bins=18,
        multiple="layer",
        kde=True,
        palette=["#3498db", "#e91e63"],
    )
    plt.xlabel("Edad")
    plt.ylabel("Frecuencia")
    plt.title("6. Distribucion de edad por sexo (0=hombre, 1=mujer)")
    plt.tight_layout()
    plt.savefig(carpeta_salida / "6_edad_por_sexo.png", dpi=120)
    plt.close()

    print("6. Edad media (solo pacientes CON cancer) por sexo")
    for s, nombre in [(0, "hombre"), (1, "mujer")]:
        media_edad = datos[(datos["sex"] == s) & (datos["cancer_diagnosis"] == 1)]["age"].mean()
        print(f"   {nombre}: {media_edad:.2f} años")
    print("\nListo.")


if __name__ == "__main__":
    main()
