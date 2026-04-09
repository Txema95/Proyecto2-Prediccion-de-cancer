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


def _find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / "data" / "processed" / "cancer_final_clean_v2.csv").is_file():
            return p
    raise FileNotFoundError(
        "No se encontró data/processed/cancer_final_clean_v2.csv. "
        "Ejecuta antes: uv run python data/scripts/cleaning/cancer_final_clean_v2.py"
    )


def main() -> None:
    root = _find_project_root()
    path_csv = root / "data" / "processed" / "cancer_final_clean_v2.csv"
    out_dir = root / "data" / "processed" / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path_csv, sep=";", encoding="utf-8-sig")
    sns.set_theme(style="whitegrid")

    print("=" * 72)
    print(f"Filas: {len(df)} | Columnas: {list(df.columns)}")
    print(f"Gráficos en: {out_dir}\n")

    # ----- 1. Correlación (heatmap) -----
    num = df.select_dtypes(include=["number"])
    corr = num.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("1. Matriz de correlacion de Pearson (todas las variables numericas)")
    plt.tight_layout()
    plt.savefig(out_dir / "1_correlacion_heatmap.png", dpi=120)
    plt.close()

    tgt = corr["cancer_diagnosis"].drop("cancer_diagnosis").sort_values(key=abs, ascending=False)
    print("1. Correlacion con cancer_diagnosis (orden |r| descendente):")
    print(tgt.to_string())
    print()

    # ----- 2. Perfil positivo vs negativo -----
    neg = df[df["cancer_diagnosis"] == 0]
    pos = df[df["cancer_diagnosis"] == 1]
    print("2. Perfil: media / proporciones (neg=0 vs pos=1)")
    print(f"   Edad media: sin cancer {neg['age'].mean():.2f} | con cancer {pos['age'].mean():.2f}")
    print(
        f"   Alcohol medio: sin cancer {neg['alcohol'].mean():.3f} | con cancer {pos['alcohol'].mean():.3f}"
    )
    print(
        f"   Tabaco medio: sin cancer {neg['tobacco'].mean():.3f} | con cancer {pos['tobacco'].mean():.3f}"
    )
    print(
        f"   % con sangre en heces (sof=1): sin cancer {100 * neg['sof'].mean():.2f}% | "
        f"con cancer {100 * pos['sof'].mean():.2f}%"
    )
    print(
        f"   % tenesmus: sin cancer {100 * neg['tenesmus'].mean():.2f}% | "
        f"con cancer {100 * pos['tenesmus'].mean():.2f}%"
    )
    print(
        f"   % rectorragia: sin cancer {100 * neg['rectorrhagia'].mean():.2f}% | "
        f"con cancer {100 * pos['rectorrhagia'].mean():.2f}%\n"
    )

    # ----- 3. digestive_family_risk_level (barras apiladas) -----
    ct = pd.crosstab(df["digestive_family_risk_level"], df["cancer_diagnosis"])
    ct = ct.reindex(columns=[0, 1], fill_value=0)
    ax = ct.plot(
        kind="bar",
        stacked=True,
        figsize=(8, 5),
        color=["#2ecc71", "#e74c3c"],
        edgecolor="white",
    )
    ax.set_xlabel("digestive_family_risk_level (0=No, 1=Unknown, 2=Med, 3=High)")
    ax.set_ylabel("N pacientes")
    ax.set_title("3. Riesgo digestivo vs diagnostico (barras apiladas)")
    ax.legend(["Sin cancer (0)", "Con cancer (1)"])
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "3_riesgo_digestivo_apilado.png", dpi=120)
    plt.close()

    print("3. Proporcion de cancer por nivel de riesgo digestivo:")
    prop = ct.div(ct.sum(axis=1), axis=0)
    print(prop)
    print()

    # ----- 4. Tríada de síntomas -----
    df_sym = df.copy()
    df_sym["n_sintomas"] = df_sym["sof"] + df_sym["tenesmus"] + df_sym["rectorrhagia"]
    tasa = df_sym.groupby("n_sintomas", observed=True)["cancer_diagnosis"].agg(["mean", "count"])
    tasa.columns = ["tasa_cancer", "n"]
    print("4. Tríada (sof+tenesmus+rectorrhagia): tasa de cancer por nº de síntomas activos")
    print(tasa)
    print()

    plt.figure(figsize=(7, 5))
    x = tasa.index.astype(int)
    plt.bar(x, 100 * tasa["tasa_cancer"], color="#3498db", edgecolor="white")
    plt.xlabel("N síntomas (de 3)")
    plt.ylabel("% con cancer_diagnosis = 1")
    plt.title("4. Probabilidad empirica de cancer segun síntomas acumulados")
    plt.xticks(x)
    plt.tight_layout()
    plt.savefig(out_dir / "4_triada_sintomas.png", dpi=120)
    plt.close()

    # ----- 5. Desequilibrio de clase -----
    vc = df["cancer_diagnosis"].value_counts().sort_index()
    n0, n1 = int(vc.get(0, 0)), int(vc.get(1, 0))
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
        data=df,
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
    plt.savefig(out_dir / "6_edad_por_sexo.png", dpi=120)
    plt.close()

    print("6. Edad media (solo pacientes CON cancer) por sexo")
    for s, nombre in [(0, "hombre"), (1, "mujer")]:
        m = df[(df["sex"] == s) & (df["cancer_diagnosis"] == 1)]["age"].mean()
        print(f"   {nombre}: {m:.2f} años")
    print("\nListo.")


if __name__ == "__main__":
    main()
