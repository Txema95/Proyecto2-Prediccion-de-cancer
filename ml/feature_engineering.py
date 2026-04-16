"""
Seleccion automatica de features candidatas con ablation step-by-step.

Salida:
- Ranking de columnas nuevas por baseline.
- Ganancia marginal por metrica.
- Conjunto final recomendado por baseline.

"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


SEMILLA = 42
FOLDS = 5
GANANCIA_MINIMA_PR_AUC = 0.0005
TOLERANCIA_CAIDA_RECALL = 0.0020
METRICAS = ["pr_auc", "recall", "f1", "roc_auc"]


def buscar_raiz_proyecto() -> Path:
    carpeta_actual = Path(__file__).resolve().parent
    for ruta in [carpeta_actual, *carpeta_actual.parents]:
        if (ruta / "data" / "processed" / "cancer_final_clean_v2.csv").is_file():
            return ruta
    raise FileNotFoundError("No se encontro data/processed/cancer_final_clean_v2.csv")


def crear_modelo(nombre_modelo: str, columnas_modelo: list[str]) -> Pipeline:
    imputador = Pipeline(steps=[("imputador", SimpleImputer(strategy="median"))])
    imputador_escalado = Pipeline(
        steps=[("imputador", SimpleImputer(strategy="median")), ("escalador", StandardScaler())]
    )
    if nombre_modelo == "regresion_logistica":
        preprocesado = ColumnTransformer(
            transformers=[("numerico", imputador_escalado, columnas_modelo)],
            remainder="drop",
        )
        clasificador = LogisticRegression(max_iter=2000, random_state=SEMILLA, class_weight="balanced")
    elif nombre_modelo == "random_forest":
        preprocesado = ColumnTransformer(
            transformers=[("numerico", imputador, columnas_modelo)],
            remainder="drop",
        )
        clasificador = RandomForestClassifier(
            n_estimators=350,
            random_state=SEMILLA,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    else:
        preprocesado = ColumnTransformer(
            transformers=[("numerico", imputador, columnas_modelo)],
            remainder="drop",
        )
        clasificador = XGBClassifier(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=SEMILLA,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
        )
    return Pipeline(steps=[("preprocesado", preprocesado), ("clasificador", clasificador)])


def preparar_dataset_con_candidatas(ruta_csv: Path) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    datos = pd.read_csv(ruta_csv, sep=";", encoding="utf-8-sig")
    objetivo = "cancer_diagnosis"
    columnas_base = [col for col in datos.columns if col not in {"id", objetivo}]

    candidatas = [
        "n_sintomas",
        "edad_x_sof",
        "edad_x_tenesmus",
        "edad_x_rectorragia",
        "riesgo_familiar_x_edad",
        "sof_x_rectorragia",
        "sof_x_tenesmus",
        "tabaco_x_alcohol",
        "edad_x_tabaco",
        "edad_x_habito_intestinal",
    ]
    datos["n_sintomas"] = datos["sof"] + datos["tenesmus"] + datos["rectorrhagia"]
    datos["edad_x_sof"] = datos["age"] * datos["sof"]
    datos["edad_x_tenesmus"] = datos["age"] * datos["tenesmus"]
    datos["edad_x_rectorragia"] = datos["age"] * datos["rectorrhagia"]
    datos["riesgo_familiar_x_edad"] = datos["digestive_family_risk_level"] * datos["age"]
    datos["sof_x_rectorragia"] = datos["sof"] * datos["rectorrhagia"]
    datos["sof_x_tenesmus"] = datos["sof"] * datos["tenesmus"]
    datos["tabaco_x_alcohol"] = datos["tobacco"] * datos["alcohol"]
    datos["edad_x_tabaco"] = datos["age"] * datos["tobacco"]
    datos["edad_x_habito_intestinal"] = datos["age"] * datos["intestinal_habit"]

    x = datos[columnas_base + candidatas].copy()
    y = datos[objetivo].astype(int).copy()
    return x, y, columnas_base, candidatas


def evaluar_por_cv(nombre_modelo: str, x: pd.DataFrame, y: pd.Series, columnas: list[str]) -> dict:
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEMILLA)
    acumulado = {metrica: [] for metrica in METRICAS}
    for indices_entreno, indices_val in cv.split(x, y):
        x_entreno = x.iloc[indices_entreno][columnas]
        y_entreno = y.iloc[indices_entreno]
        x_val = x.iloc[indices_val][columnas]
        y_val = y.iloc[indices_val]
        modelo = crear_modelo(nombre_modelo, columnas)
        modelo.fit(x_entreno, y_entreno)
        probas = modelo.predict_proba(x_val)[:, 1]
        pred = (probas >= 0.5).astype(int)
        acumulado["pr_auc"].append(float(average_precision_score(y_val, probas)))
        acumulado["recall"].append(float(recall_score(y_val, pred, zero_division=0)))
        acumulado["f1"].append(float(f1_score(y_val, pred, zero_division=0)))
        acumulado["roc_auc"].append(float(roc_auc_score(y_val, probas)))
    return {metrica: float(sum(valores) / len(valores)) for metrica, valores in acumulado.items()}


def calcular_ganancia(metricas_nuevas: dict, metricas_base: dict) -> dict:
    return {metrica: metricas_nuevas[metrica] - metricas_base[metrica] for metrica in METRICAS}


def hacer_ablation_para_modelo(nombre_modelo: str, x: pd.DataFrame, y: pd.Series, columnas_base: list[str], candidatas: list[str]) -> dict:
    metricas_base = evaluar_por_cv(nombre_modelo, x, y, columnas_base)
    seleccionadas = []
    ranking = []
    candidatas_pendientes = candidatas.copy()
    metricas_actuales = metricas_base

    while candidatas_pendientes:
        mejor_candidata = ""
        mejores_metricas = None
        mejor_ganancia = None
        mejor_pr_auc = -1e9
        for candidata in candidatas_pendientes:
            columnas_prueba = columnas_base + seleccionadas + [candidata]
            metricas_prueba = evaluar_por_cv(nombre_modelo, x, y, columnas_prueba)
            ganancia = calcular_ganancia(metricas_prueba, metricas_actuales)
            fila_ranking = {
                "paso": len(seleccionadas) + 1,
                "feature_candidata": candidata,
                "seleccion_actual": seleccionadas.copy(),
                "metricas_cv": metricas_prueba,
                "ganancia_marginal": ganancia,
            }
            ranking.append(fila_ranking)
            if ganancia["pr_auc"] > mejor_pr_auc:
                mejor_pr_auc = ganancia["pr_auc"]
                mejor_candidata = candidata
                mejores_metricas = metricas_prueba
                mejor_ganancia = ganancia

        if mejor_candidata == "":
            break
        mejora_pr_auc = mejor_ganancia["pr_auc"] >= GANANCIA_MINIMA_PR_AUC
        recall_no_cae = mejor_ganancia["recall"] >= -TOLERANCIA_CAIDA_RECALL
        if mejora_pr_auc and recall_no_cae:
            seleccionadas.append(mejor_candidata)
            candidatas_pendientes.remove(mejor_candidata)
            metricas_actuales = mejores_metricas
        else:
            break

    ranking_ordenado = sorted(
        ranking,
        key=lambda fila: fila["ganancia_marginal"]["pr_auc"],
        reverse=True,
    )
    return {
        "modelo": nombre_modelo,
        "metricas_base_cv": metricas_base,
        "features_recomendadas": seleccionadas,
        "metricas_finales_cv": metricas_actuales,
        "ganancia_total_vs_base": calcular_ganancia(metricas_actuales, metricas_base),
        "ranking_candidatas": ranking_ordenado,
    }


def main() -> None:
    raiz = buscar_raiz_proyecto()
    ruta_csv = raiz / "data" / "processed" / "cancer_final_clean_v2.csv"
    carpeta_salida = raiz / "ml" / "feature_engineering" / "v1"
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    x, y, columnas_base, candidatas = preparar_dataset_con_candidatas(ruta_csv)
    resultados = {}
    for nombre_modelo in ["regresion_logistica", "random_forest", "xgboost"]:
        resultados[nombre_modelo] = hacer_ablation_para_modelo(
            nombre_modelo=nombre_modelo,
            x=x,
            y=y,
            columnas_base=columnas_base,
            candidatas=candidatas,
        )

    resumen = []
    for nombre_modelo, info in resultados.items():
        fila = {"modelo": nombre_modelo, "features_recomendadas": ",".join(info["features_recomendadas"])}
        for metrica in METRICAS:
            fila[f"{metrica}_base_cv"] = info["metricas_base_cv"][metrica]
            fila[f"{metrica}_final_cv"] = info["metricas_finales_cv"][metrica]
            fila[f"ganancia_{metrica}"] = info["ganancia_total_vs_base"][metrica]
        resumen.append(fila)
    tabla_resumen = pd.DataFrame(resumen)
    tabla_resumen.to_csv(carpeta_salida / "resumen_seleccion_features.csv", index=False, encoding="utf-8")
    (carpeta_salida / "resultados_ablation.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 88)
    print("Seleccion automatica de features completada")
    print("=" * 88)
    print(f"Columnas base: {len(columnas_base)} | Candidatas evaluadas: {len(candidatas)}")
    print("Resumen por baseline:")
    print(tabla_resumen.to_string(index=False))
    print(f"\nArtefactos guardados en: {carpeta_salida}")


if __name__ == "__main__":
    main()
