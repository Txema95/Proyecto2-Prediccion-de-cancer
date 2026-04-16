"""
Aplicacion Streamlit para entrenar y evaluar baselines.

Los tres algoritmos usan el mismo conjunto final de features derivadas.

Ejecucion:
    uv run streamlit run ml/main.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATURES_GENERADAS = ["n_sintomas", "riesgo_familiar_x_edad"]
EXPLICACION_FEATURES = {
    "n_sintomas": "Resume la carga sintomatica en una sola variable al combinar sof, tenesmus y rectorrhagia.",
    "riesgo_familiar_x_edad": "Combina la edad con el riesgo familiar digestivo para capturar un efecto conjunto de contexto clinico.",
}


def buscar_raiz_proyecto() -> Path:
    carpeta_actual = Path(__file__).resolve().parent
    for ruta in [carpeta_actual, *carpeta_actual.parents]:
        if (ruta / "data" / "processed" / "cancer_final_clean_v2.csv").is_file():
            return ruta
    raise FileNotFoundError("No se encontro data/processed/cancer_final_clean_v2.csv")


def obtener_rutas(raiz: Path) -> dict:
    rutas = {
        "csv": raiz / "data" / "processed" / "cancer_final_clean_v2.csv",
        "comun": raiz / "ml" / "comun" / "v1",
        "regresion_logistica": raiz / "ml" / "regresion_logistica" / "v1" / "artefactos",
        "random_forest": raiz / "ml" / "random_forest" / "v1" / "artefactos",
        "xgboost": raiz / "ml" / "xgboost" / "v1" / "artefactos",
    }
    rutas["comun"].mkdir(parents=True, exist_ok=True)
    rutas["regresion_logistica"].mkdir(parents=True, exist_ok=True)
    rutas["random_forest"].mkdir(parents=True, exist_ok=True)
    rutas["xgboost"].mkdir(parents=True, exist_ok=True)
    return rutas


def calcular_metricas(y_real: pd.Series, probabilidades: pd.Series, predicciones: pd.Series) -> dict:
    return {
        "accuracy": float(accuracy_score(y_real, predicciones)),
        "recall": float(recall_score(y_real, predicciones, zero_division=0)),
        "f1": float(f1_score(y_real, predicciones, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_real, probabilidades)),
        "pr_auc": float(average_precision_score(y_real, probabilidades)),
    }


def calcular_scale_pos_weight(y: pd.Series) -> float:
    negativos = int((y == 0).sum())
    positivos = int((y == 1).sum())
    if positivos == 0:
        return 1.0
    return negativos / positivos


def crear_modelo(
    nombre_modelo: str, columnas_modelo: list[str], scale_pos_weight: float | None = None
) -> Pipeline:
    imputador = Pipeline(steps=[("imputador", SimpleImputer(strategy="median"))])
    imputador_y_escalado = Pipeline(
        steps=[("imputador", SimpleImputer(strategy="median")), ("escalador", StandardScaler())]
    )
    if nombre_modelo == "regresion_logistica":
        preprocesado = ColumnTransformer(
            transformers=[("numerico", imputador_y_escalado, columnas_modelo)],
            remainder="drop",
        )
        clasificador = LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")
    elif nombre_modelo == "random_forest":
        preprocesado = ColumnTransformer(
            transformers=[("numerico", imputador, columnas_modelo)],
            remainder="drop",
        )
        clasificador = RandomForestClassifier(
            n_estimators=350,
            random_state=42,
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
            scale_pos_weight=scale_pos_weight if scale_pos_weight is not None else 1.0,
            random_state=42,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
        )
    return Pipeline(steps=[("preprocesado", preprocesado), ("clasificador", clasificador)])


def preparar_datos_modelo(ruta_csv: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    datos = pd.read_csv(ruta_csv, sep=";", encoding="utf-8-sig")
    datos["n_sintomas"] = datos["sof"] + datos["tenesmus"] + datos["rectorrhagia"]
    datos["riesgo_familiar_x_edad"] = datos["digestive_family_risk_level"] * datos["age"]
    objetivo = "cancer_diagnosis"
    columnas_modelo = [col for col in datos.columns if col not in {"id", objetivo}]
    x = datos[columnas_modelo].copy()
    y = datos[objetivo].astype(int).copy()
    ids = datos["id"].copy()
    return x, y, ids, columnas_modelo


def hacer_particiones(x: pd.DataFrame, y: pd.Series, ids: pd.Series) -> dict:
    x_train_val, x_test, y_train_val, y_test, ids_train_val, ids_test = train_test_split(
        x, y, ids, test_size=0.15, stratify=y, random_state=42
    )
    x_train, x_val, y_train, y_val, _, _ = train_test_split(
        x_train_val,
        y_train_val,
        ids_train_val,
        test_size=0.1765,
        stratify=y_train_val,
        random_state=42,
    )
    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_train_val": x_train_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_train_val": y_train_val,
        "y_test": y_test,
        "ids_test": ids_test,
    }


def guardar_artefactos_comunes(ruta_comun: Path, columnas_modelo: list[str], ids_test: pd.Series) -> None:
    (ruta_comun / "columnas_modelo.json").write_text(
        json.dumps(columnas_modelo, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ids_test.to_frame(name="id").to_csv(ruta_comun / "ids_test.csv", index=False, encoding="utf-8")


def entrenar_y_evaluar_modelo(nombre_modelo: str, rutas: dict) -> dict:
    x, y, ids, columnas_modelo = preparar_datos_modelo(rutas["csv"])
    particiones = hacer_particiones(x, y, ids)
    guardar_artefactos_comunes(rutas["comun"], columnas_modelo, particiones["ids_test"])

    scale_pos_weight_train = None
    scale_pos_weight_train_val = None
    if nombre_modelo == "xgboost":
        scale_pos_weight_train = calcular_scale_pos_weight(particiones["y_train"])
        scale_pos_weight_train_val = calcular_scale_pos_weight(particiones["y_train_val"])

    modelo = crear_modelo(nombre_modelo, columnas_modelo, scale_pos_weight=scale_pos_weight_train)
    modelo.fit(particiones["x_train"], particiones["y_train"])
    probas_val = modelo.predict_proba(particiones["x_val"])[:, 1]
    pred_val = (probas_val >= 0.5).astype(int)
    metricas_val = calcular_metricas(particiones["y_val"], probas_val, pred_val)

    modelo = crear_modelo(nombre_modelo, columnas_modelo, scale_pos_weight=scale_pos_weight_train_val)
    modelo.fit(particiones["x_train_val"], particiones["y_train_val"])
    probas_test = modelo.predict_proba(particiones["x_test"])[:, 1]
    pred_test = (probas_test >= 0.5).astype(int)
    metricas_test = calcular_metricas(particiones["y_test"], probas_test, pred_test)

    carpeta_modelo = rutas[nombre_modelo]
    joblib.dump(modelo, carpeta_modelo / "modelo.joblib")
    (carpeta_modelo / "metricas_validacion.json").write_text(
        json.dumps(metricas_val, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (carpeta_modelo / "metricas_test.json").write_text(
        json.dumps(metricas_test, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    matriz = confusion_matrix(particiones["y_test"], pred_test, labels=[0, 1])
    plt.figure(figsize=(6, 4.5))
    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["Real 0", "Real 1"],
    )
    plt.title(f"Matriz de confusion - {nombre_modelo}")
    plt.xlabel("Prediccion")
    plt.ylabel("Valor real")
    plt.tight_layout()
    plt.savefig(carpeta_modelo / "matriz_confusion_test.png", dpi=140)
    plt.close()

    reporte = classification_report(particiones["y_test"], pred_test, digits=4, zero_division=0)
    texto_reporte = [
        f"Evaluacion baseline: {nombre_modelo}",
        "",
        f"Registros en test: {len(particiones['x_test'])}",
        "",
        "Metricas validacion:",
        json.dumps(metricas_val, indent=2, ensure_ascii=False),
        "",
        "Metricas test:",
        json.dumps(metricas_test, indent=2, ensure_ascii=False),
        "",
        "Classification report:",
        reporte,
    ]
    (carpeta_modelo / "reporte_test.txt").write_text("\n".join(texto_reporte), encoding="utf-8")

    return {
        "metricas_validacion": metricas_val,
        "metricas_test": metricas_test,
        "ruta_modelo": carpeta_modelo,
    }


def cargar_artefactos_modelo(ruta_modelo: Path) -> dict | None:
    ruta_val = ruta_modelo / "metricas_validacion.json"
    ruta_test = ruta_modelo / "metricas_test.json"
    ruta_reporte = ruta_modelo / "reporte_test.txt"
    ruta_matriz = ruta_modelo / "matriz_confusion_test.png"
    if not ruta_val.exists() or not ruta_test.exists():
        return None
    return {
        "metricas_validacion": json.loads(ruta_val.read_text(encoding="utf-8")),
        "metricas_test": json.loads(ruta_test.read_text(encoding="utf-8")),
        "reporte": ruta_reporte.read_text(encoding="utf-8") if ruta_reporte.exists() else "",
        "ruta_matriz": ruta_matriz,
    }


def construir_tabla_comparacion(rutas: dict) -> pd.DataFrame:
    filas = []
    for nombre_modelo in ["regresion_logistica", "random_forest", "xgboost"]:
        artefactos = cargar_artefactos_modelo(rutas[nombre_modelo])
        if artefactos is None:
            continue
        metricas_test = artefactos["metricas_test"]
        filas.append(
            {
                "modelo": nombre_modelo,
                "accuracy": metricas_test.get("accuracy"),
                "recall": metricas_test.get("recall"),
                "f1": metricas_test.get("f1"),
                "roc_auc": metricas_test.get("roc_auc"),
                "pr_auc": metricas_test.get("pr_auc"),
            }
        )
    if not filas:
        return pd.DataFrame()
    tabla = pd.DataFrame(filas)
    return tabla.sort_values(by="pr_auc", ascending=False).reset_index(drop=True)


def entrenar_todos_los_baselines(rutas: dict) -> dict:
    resultados = {}
    for nombre_modelo in ["regresion_logistica", "random_forest", "xgboost"]:
        resultados[nombre_modelo] = entrenar_y_evaluar_modelo(nombre_modelo, rutas)
    return resultados


def main() -> None:
    st.set_page_config(page_title="Baselines ML - Cancer Colon", layout="wide")
    st.title("Entrenamiento y evaluacion de baselines")
    st.caption("Version v1 de modelos con el conjunto final de variables derivadas incorporado.")

    raiz = buscar_raiz_proyecto()
    rutas = obtener_rutas(raiz)
    x, y, ids, columnas_modelo = preparar_datos_modelo(rutas["csv"])
    particiones = hacer_particiones(x, y, ids)
    guardar_artefactos_comunes(rutas["comun"], columnas_modelo, particiones["ids_test"])

    st.subheader("Variables derivadas")
    if st.button("Mostrar features generadas", use_container_width=True):
        st.write("Features incorporadas al dataset del modelo:")
        st.json(FEATURES_GENERADAS)
        st.write("Motivo de seleccion:")
        for nombre_feature, explicacion in EXPLICACION_FEATURES.items():
            st.markdown(f"- `{nombre_feature}`: {explicacion}")

    st.divider()
    st.subheader("Entrenamiento y comparacion de baselines")
    with st.spinner("Entrenando regresion logistica, random forest y xgboost..."):
        entrenar_todos_los_baselines(rutas)
    st.success("Entrenamiento y evaluacion completados para los 3 baselines.")

    st.subheader("Tabla comparativa de resultados")
    tabla_comparacion = construir_tabla_comparacion(rutas)
    if tabla_comparacion.empty:
        st.info("Todavia no hay baselines con artefactos de metricas para comparar.")
    else:
        st.write("Resultados actuales:")
        st.dataframe(tabla_comparacion, use_container_width=True)
        mejor = tabla_comparacion.iloc[0]
        st.success(
            f"Mejor modelo actual por PR-AUC: {mejor['modelo']} "
            f"(pr_auc={mejor['pr_auc']:.4f}, recall={mejor['recall']:.4f})"
        )

    st.subheader("Matrices de confusion (test)")
    columnas = st.columns(3)
    modelos = ["regresion_logistica", "random_forest", "xgboost"]
    for i, nombre_modelo in enumerate(modelos):
        with columnas[i]:
            st.write(f"**{nombre_modelo}**")
            artefactos = cargar_artefactos_modelo(rutas[nombre_modelo])
            if artefactos is None:
                st.info("Sin artefactos todavia.")
            else:
                if artefactos["ruta_matriz"].exists():
                    st.image(str(artefactos["ruta_matriz"]))
                else:
                    st.info("No existe matriz de confusion para este modelo.")


if __name__ == "__main__":
    main()
