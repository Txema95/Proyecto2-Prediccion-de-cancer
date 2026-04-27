"""
Aplicacion Streamlit para entrenar y evaluar baselines.

Los tres algoritmos usan el mismo conjunto final de features derivadas.

Ejecucion:
    uv run streamlit run ml/main.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from catboost import CatBoostClassifier
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
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.svm import SVC

FEATURES_GENERADAS = ["n_sintomas", "riesgo_familiar_x_edad"]
EXPLICACION_FEATURES = {
    "n_sintomas": "Resume la carga sintomatica en una sola variable al combinar sof, tenesmus y rectorrhagia.",
    "riesgo_familiar_x_edad": "Combina la edad con el riesgo familiar digestivo para capturar un efecto conjunto de contexto clinico.",
}


def obtener_umbral_decision() -> float:
    valor_bruto = os.environ.get("SIMULATOR_DECISION_THRESHOLD", "0.1").strip()
    try:
        valor = float(valor_bruto)
    except ValueError as exc:
        raise ValueError("SIMULATOR_DECISION_THRESHOLD debe ser numerico") from exc
    if not 0.0 <= valor <= 1.0:
        raise ValueError("SIMULATOR_DECISION_THRESHOLD debe estar entre 0.0 y 1.0")
    return valor


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
        "svm": raiz / "ml" / "svm" / "v1" / "artefactos",
        "catboost": raiz / "ml" / "catboost" / "v1" / "artefactos",
    }
    rutas["comun"].mkdir(parents=True, exist_ok=True)
    rutas["regresion_logistica"].mkdir(parents=True, exist_ok=True)
    rutas["random_forest"].mkdir(parents=True, exist_ok=True)
    rutas["xgboost"].mkdir(parents=True, exist_ok=True)
    rutas["svm"].mkdir(parents=True, exist_ok=True)
    rutas["catboost"].mkdir(parents=True, exist_ok=True)
    return rutas


def calcular_metricas(y_real: pd.Series, probabilidades: pd.Series, predicciones: pd.Series) -> dict:
    return {
        "accuracy": float(accuracy_score(y_real, predicciones)),
        "precision": float(precision_score(y_real, predicciones, zero_division=0)),
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
    elif nombre_modelo == "svm":
        preprocesado = ColumnTransformer(
            transformers=[("numerico", imputador_y_escalado, columnas_modelo)],
            remainder="drop",
        )
        clasificador = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
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
    elif nombre_modelo == "catboost":
        preprocesado = ColumnTransformer(
            transformers=[("numerico", imputador, columnas_modelo)],
            remainder="drop",
        )
        clasificador = CatBoostClassifier(
            iterations=350,
            depth=6,
            learning_rate=0.05,
            eval_metric="Logloss",
            random_seed=42,
            scale_pos_weight=scale_pos_weight if scale_pos_weight is not None else 1.0,
            verbose=False,
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
    for nombre_modelo in ["regresion_logistica", "random_forest", "xgboost", "svm", "catboost"]:
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


def construir_tabla_comparacion_desde_resultados(resultados: dict) -> pd.DataFrame:
    def clasificar_estado_generalizacion(pr_auc_test: float | None, pr_auc_cv: float | None) -> str:
        if pr_auc_test is None or pr_auc_cv is None:
            return "sin_datos"
        gap_cv_test = float(pr_auc_cv - pr_auc_test)

        # Bajo rendimiento en ambos escenarios sugiere alto sesgo (underfitting).
        if pr_auc_cv < 0.75 and pr_auc_test < 0.75:
            return "underfitting"
        # CV muy por encima de test indica sobreajuste en distinta severidad.
        if gap_cv_test >= 0.10:
            return "severe_overfitting"
        if gap_cv_test >= 0.05:
            return "overfitting"
        return "good_generalization"

    filas = []
    for nombre_modelo, info in resultados.items():
        metricas_test = info["metricas_test"]
        metricas_cv = info.get("metricas_cv_media", {})
        umbral_usado = info.get("umbral_usado")
        detalle_umbral = info.get("detalle_umbral", {})
        pr_auc_test = metricas_test.get("pr_auc")
        pr_auc_cv = metricas_cv.get("pr_auc")
        gap_cv_test_pr_auc = None
        if pr_auc_test is not None and pr_auc_cv is not None:
            gap_cv_test_pr_auc = float(pr_auc_cv - pr_auc_test)
        filas.append(
            {
                "modelo": nombre_modelo,
                "umbral_usado": umbral_usado,
                "accuracy": metricas_test.get("accuracy"),
                "precision": metricas_test.get("precision"),
                "recall": metricas_test.get("recall"),
                "f1": metricas_test.get("f1"),
                "roc_auc": metricas_test.get("roc_auc"),
                "pr_auc": pr_auc_test,
                "umbral_cumple_precision_minima": detalle_umbral.get("cumple_precision"),
                "cv_recall_mean": metricas_cv.get("recall"),
                "cv_f1_mean": metricas_cv.get("f1"),
                "cv_pr_auc_mean": pr_auc_cv,
                "gap_cv_test_pr_auc": gap_cv_test_pr_auc,
                "diagnostico_generalizacion": clasificar_estado_generalizacion(pr_auc_test, pr_auc_cv),
            }
        )
    if not filas:
        return pd.DataFrame()
    tabla = pd.DataFrame(filas)
    return tabla.sort_values(by="pr_auc", ascending=False).reset_index(drop=True)


def _resumen_metricas_cv(metricas_por_fold: list[dict]) -> tuple[dict, dict]:
    columnas_metricas = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    df_cv = pd.DataFrame(metricas_por_fold)
    medias = {k: float(df_cv[k].mean()) for k in columnas_metricas}
    desv = {k: float(df_cv[k].std(ddof=0)) for k in columnas_metricas}
    return medias, desv


def seleccionar_umbral_alta_sensibilidad(
    y_real: pd.Series, probabilidades: pd.Series, precision_minima: float
) -> tuple[float, dict]:
    mejor = None
    candidatos = [i / 100 for i in range(1, 100)]
    for umbral in candidatos:
        pred = (probabilidades >= umbral).astype(int)
        precision = float(precision_score(y_real, pred, zero_division=0))
        recall = float(recall_score(y_real, pred, zero_division=0))
        f1 = float(f1_score(y_real, pred, zero_division=0))
        cumple_precision = precision >= precision_minima
        fila = {
            "umbral": float(umbral),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "cumple_precision": cumple_precision,
        }
        if mejor is None:
            mejor = fila
            continue

        # Regla principal: priorizar candidatos que cumplan precision minima.
        if fila["cumple_precision"] and not mejor["cumple_precision"]:
            mejor = fila
            continue
        if fila["cumple_precision"] == mejor["cumple_precision"]:
            # Dentro del mismo grupo, priorizar recall (sensibilidad).
            if fila["recall"] > mejor["recall"]:
                mejor = fila
                continue
            if fila["recall"] == mejor["recall"]:
                # Si recall empata, priorizar mejor f1 y luego mayor precision.
                if fila["f1"] > mejor["f1"]:
                    mejor = fila
                    continue
                if fila["f1"] == mejor["f1"] and fila["precision"] > mejor["precision"]:
                    mejor = fila
                    continue

    if mejor is None:
        return 0.5, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "cumple_precision": False}
    return mejor["umbral"], mejor


def evaluar_cv_modelo(
    nombre_modelo: str,
    x_train_val: pd.DataFrame,
    y_train_val: pd.Series,
    columnas_modelo: list[str],
    umbral_decision: float,
    folds_cv: int,
) -> tuple[dict, dict]:
    cv = StratifiedKFold(n_splits=folds_cv, shuffle=True, random_state=42)
    metricas_por_fold: list[dict] = []
    for indices_train, indices_val in cv.split(x_train_val, y_train_val):
        x_train_fold = x_train_val.iloc[indices_train]
        y_train_fold = y_train_val.iloc[indices_train]
        x_val_fold = x_train_val.iloc[indices_val]
        y_val_fold = y_train_val.iloc[indices_val]

        scale_pos_weight_fold = None
        if nombre_modelo in {"xgboost", "catboost"}:
            scale_pos_weight_fold = calcular_scale_pos_weight(y_train_fold)

        modelo_fold = crear_modelo(nombre_modelo, columnas_modelo, scale_pos_weight=scale_pos_weight_fold)
        modelo_fold.fit(x_train_fold, y_train_fold)
        probas_fold = modelo_fold.predict_proba(x_val_fold)[:, 1]
        pred_fold = (probas_fold >= umbral_decision).astype(int)
        metricas_por_fold.append(calcular_metricas(y_val_fold, probas_fold, pred_fold))

    return _resumen_metricas_cv(metricas_por_fold)


def entrenar_y_evaluar_modelo(
    nombre_modelo: str,
    rutas: dict,
    umbral_decision: float,
    folds_cv: int,
    usar_umbral_automatico: bool,
    precision_minima_umbral: float,
) -> dict:
    x, y, ids, columnas_modelo = preparar_datos_modelo(rutas["csv"])
    particiones = hacer_particiones(x, y, ids)
    guardar_artefactos_comunes(rutas["comun"], columnas_modelo, particiones["ids_test"])

    scale_pos_weight_train = None
    scale_pos_weight_train_val = None
    if nombre_modelo in {"xgboost", "catboost"}:
        scale_pos_weight_train = calcular_scale_pos_weight(particiones["y_train"])
        scale_pos_weight_train_val = calcular_scale_pos_weight(particiones["y_train_val"])

    modelo = crear_modelo(nombre_modelo, columnas_modelo, scale_pos_weight=scale_pos_weight_train)
    modelo.fit(particiones["x_train"], particiones["y_train"])
    probas_val = modelo.predict_proba(particiones["x_val"])[:, 1]
    umbral_modelo = umbral_decision
    detalle_umbral = {
        "precision": None,
        "recall": None,
        "f1": None,
        "cumple_precision": None,
    }
    if usar_umbral_automatico:
        umbral_modelo, detalle_umbral = seleccionar_umbral_alta_sensibilidad(
            y_real=particiones["y_val"],
            probabilidades=probas_val,
            precision_minima=precision_minima_umbral,
        )
    pred_val = (probas_val >= umbral_modelo).astype(int)
    metricas_val = calcular_metricas(particiones["y_val"], probas_val, pred_val)

    metricas_cv_media, metricas_cv_std = evaluar_cv_modelo(
        nombre_modelo=nombre_modelo,
        x_train_val=particiones["x_train_val"],
        y_train_val=particiones["y_train_val"],
        columnas_modelo=columnas_modelo,
        umbral_decision=umbral_modelo,
        folds_cv=folds_cv,
    )

    modelo = crear_modelo(nombre_modelo, columnas_modelo, scale_pos_weight=scale_pos_weight_train_val)
    modelo.fit(particiones["x_train_val"], particiones["y_train_val"])
    probas_test = modelo.predict_proba(particiones["x_test"])[:, 1]
    pred_test = (probas_test >= umbral_modelo).astype(int)
    metricas_test = calcular_metricas(particiones["y_test"], probas_test, pred_test)

    carpeta_modelo = rutas[nombre_modelo]
    joblib.dump(modelo, carpeta_modelo / "modelo.joblib")
    (carpeta_modelo / "metricas_cv.json").write_text(
        json.dumps(
            {
                "folds": folds_cv,
                "media": metricas_cv_media,
                "desviacion_estandar": metricas_cv_std,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
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
        (
            f"Umbral de decision (automatico): {umbral_modelo:.4f} "
            f"(precision_minima={precision_minima_umbral:.2f}, "
            f"cumple_precision={detalle_umbral['cumple_precision']})"
            if usar_umbral_automatico
            else f"Umbral de decision (manual): {umbral_modelo:.4f}"
        ),
        f"Cross-validation (folds): {folds_cv}",
        "Metricas CV (media):",
        json.dumps(metricas_cv_media, indent=2, ensure_ascii=False),
        "Metricas CV (desviacion_estandar):",
        json.dumps(metricas_cv_std, indent=2, ensure_ascii=False),
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
        "umbral_usado": float(umbral_modelo),
        "detalle_umbral": detalle_umbral,
        "metricas_cv_media": metricas_cv_media,
        "metricas_cv_std": metricas_cv_std,
        "metricas_validacion": metricas_val,
        "metricas_test": metricas_test,
        "ruta_modelo": carpeta_modelo,
    }


def entrenar_todos_los_baselines(
    rutas: dict,
    umbral_decision: float,
    folds_cv: int,
    usar_umbral_automatico: bool,
    precision_minima_umbral: float,
) -> dict:
    resultados = {}
    for nombre_modelo in ["regresion_logistica", "random_forest", "xgboost", "svm", "catboost"]:
        resultados[nombre_modelo] = entrenar_y_evaluar_modelo(
            nombre_modelo=nombre_modelo,
            rutas=rutas,
            umbral_decision=umbral_decision,
            folds_cv=folds_cv,
            usar_umbral_automatico=usar_umbral_automatico,
            precision_minima_umbral=precision_minima_umbral,
        )
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
    umbral_por_defecto = obtener_umbral_decision()
    umbral_decision = st.slider(
        "Umbral de decision para clasificar positivo (se aplica a validacion y test)",
        min_value=0.0,
        max_value=1.0,
        value=float(umbral_por_defecto),
        step=0.01,
    )
    usar_umbral_automatico = st.checkbox(
        "Optimizar umbral automaticamente (maximizar sensibilidad con precision minima)",
        value=True,
    )
    precision_minima_umbral = st.slider(
        "Precision minima objetivo para la optimizacion de umbral",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.01,
        disabled=not usar_umbral_automatico,
    )
    folds_cv = st.slider(
        "Numero de folds para cross-validation estratificada",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
    )
    st.caption(
        "El valor del slider tiene prioridad en esta ejecucion. "
        "Si no lo tocas, usa el de SIMULATOR_DECISION_THRESHOLD."
    )
    with st.spinner(
        "Entrenando regresion logistica, random forest, xgboost, svm y catboost con cross-validation..."
    ):
        resultados = entrenar_todos_los_baselines(
            rutas=rutas,
            umbral_decision=umbral_decision,
            folds_cv=folds_cv,
            usar_umbral_automatico=usar_umbral_automatico,
            precision_minima_umbral=precision_minima_umbral,
        )
    st.success("Entrenamiento y evaluacion completados para los 5 baselines.")

    st.subheader("Tabla comparativa de resultados")
    tabla_comparacion = construir_tabla_comparacion_desde_resultados(resultados)
    if tabla_comparacion.empty:
        st.info("Todavia no hay baselines con artefactos de metricas para comparar.")
    else:
        st.write("Resultados actuales:")
        st.dataframe(tabla_comparacion, use_container_width=True)
        mejor = tabla_comparacion.iloc[0]
        st.success(
            f"Mejor modelo actual por PR-AUC: {mejor['modelo']} "
            f"(pr_auc={mejor['pr_auc']:.4f}, recall={mejor['recall']:.4f}, "
            f"precision={mejor['precision']:.4f}, umbral={mejor['umbral_usado']:.2f})"
        )
        tabla_cv = tabla_comparacion.sort_values(by="cv_pr_auc_mean", ascending=False).reset_index(drop=True)
        st.write("Ranking adicional por robustez (CV PR-AUC medio):")
        st.dataframe(tabla_cv, use_container_width=True)
        mejor_cv = tabla_cv.iloc[0]
        st.info(
            f"Mejor por CV PR-AUC medio: {mejor_cv['modelo']} "
            f"(cv_pr_auc_mean={mejor_cv['cv_pr_auc_mean']:.4f}, cv_recall_mean={mejor_cv['cv_recall_mean']:.4f})"
        )
        st.caption(
            "Diagnostico de generalizacion: "
            "`good_generalization` (gap CV-test < 0.05), "
            "`overfitting` (gap >= 0.05), "
            "`severe_overfitting` (gap >= 0.10), "
            "`underfitting` (PR-AUC CV y test < 0.75)."
        )
        if usar_umbral_automatico:
            st.caption(
                "Con optimizacion automatica, cada modelo busca su mejor umbral en validacion "
                f"maximizando recall y exigiendo precision >= {precision_minima_umbral:.2f}."
            )

    st.subheader("Matrices de confusion (test)")
    modelos = ["regresion_logistica", "random_forest", "xgboost", "svm", "catboost"]
    columnas = st.columns(len(modelos))
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
