"""Carga del modelo tabular e inferencia en el servidor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from app.core.paths import raiz_proyecto

OBJETIVO = "cancer_diagnosis"
COLUMNA_ID = "id"

_contexto: ContextoModeloTabular | None = None


@dataclass
class ContextoModeloTabular:
    modelo: Pipeline
    columnas_modelo: list[str]
    medias_referencia: dict[str, float]


def _rutas_ml() -> dict[str, Path]:
    raiz = raiz_proyecto()
    return {
        "csv": raiz / "data" / "processed" / "cancer_final_clean_v2.csv",
        "modelo_tabular": raiz / "ml" / "catboost" / "v1" / "artefactos" / "modelo.joblib",
        "modelo_imagen": raiz / "ml" / "imagen" / "v1" / "artefactos" / "modelo.joblib",
    }


def _cargar_csv(ruta_csv: Path) -> pd.DataFrame:
    return pd.read_csv(ruta_csv, sep=";", encoding="utf-8-sig")


def _calcular_scale_pos_weight(y: pd.Series) -> float:
    negativos = int((y == 0).sum())
    positivos = int((y == 1).sum())
    if positivos == 0:
        return 1.0
    return negativos / positivos


def _construir_features(datos: pd.DataFrame) -> pd.DataFrame:
    datos_transformados = datos.copy()
    datos_transformados["n_sintomas"] = (
        datos_transformados["sof"] + datos_transformados["tenesmus"] + datos_transformados["rectorrhagia"]
    )
    datos_transformados["riesgo_familiar_x_edad"] = (
        datos_transformados["digestive_family_risk_level"] * datos_transformados["age"]
    )
    return datos_transformados


def _entrenar_modelo_tabular(datos: pd.DataFrame) -> tuple[Pipeline, list[str]]:
    datos_transformados = _construir_features(datos)
    columnas_modelo = [col for col in datos_transformados.columns if col not in {COLUMNA_ID, OBJETIVO}]
    x = datos_transformados[columnas_modelo].copy()
    y = datos_transformados[OBJETIVO].astype(int).copy()
    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.15, stratify=y, random_state=42)
    scale_pos_weight = _calcular_scale_pos_weight(y_train)

    modelo = XGBClassifier(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
    )
    modelo.fit(x_train, y_train)
    return modelo, columnas_modelo


def obtener_contexto_tabular() -> ContextoModeloTabular:
    global _contexto
    if _contexto is not None:
        return _contexto

    rutas = _rutas_ml()
    ruta_csv = rutas["csv"]
    ruta_modelo_tabular = rutas["modelo_tabular"]

    if not ruta_csv.is_file():
        raise FileNotFoundError(f"No existe el dataset procesado: {ruta_csv}")

    datos = _cargar_csv(ruta_csv)
    datos_transformados = _construir_features(datos)
    columnas_modelo = [col for col in datos_transformados.columns if col not in {COLUMNA_ID, OBJETIVO}]
    medias_referencia = datos_transformados[columnas_modelo].median(numeric_only=True).to_dict()

    if ruta_modelo_tabular.exists():
        modelo = joblib.load(ruta_modelo_tabular)
        _contexto = ContextoModeloTabular(
            modelo=modelo,
            columnas_modelo=columnas_modelo,
            medias_referencia=medias_referencia,
        )
        return _contexto

    modelo, columnas_entrenadas = _entrenar_modelo_tabular(datos)
    ruta_modelo_tabular.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta_modelo_tabular)
    _contexto = ContextoModeloTabular(
        modelo=modelo,
        columnas_modelo=columnas_entrenadas,
        medias_referencia=medias_referencia,
    )
    return _contexto


def preparar_fila_prediccion(
    contexto: ContextoModeloTabular,
    datos_formulario: dict[str, float],
) -> pd.DataFrame:
    fila = {}
    for columna in contexto.columnas_modelo:
        fila[columna] = float(datos_formulario.get(columna, contexto.medias_referencia.get(columna, 0.0)))
    fila["n_sintomas"] = float(fila.get("sof", 0.0) + fila.get("tenesmus", 0.0) + fila.get("rectorrhagia", 0.0))
    fila["riesgo_familiar_x_edad"] = float(
        fila.get("digestive_family_risk_level", 0.0) * fila.get("age", 0.0)
    )
    return pd.DataFrame([fila], columns=contexto.columnas_modelo)


def procesar_resultado_imagen(ruta_modelo_imagen: Path, num_imagenes: int) -> dict[str, float | str | None]:
    if num_imagenes <= 0:
        return {"estado": "sin_imagen", "mensaje": "No se han adjuntado imagenes.", "probabilidad": None}
    if not ruta_modelo_imagen.exists():
        return {
            "estado": "modelo_no_disponible",
            "mensaje": "El modelo de imagen todavia no esta integrado. Se usa solo la prediccion tabular.",
            "probabilidad": None,
        }
    return {
        "estado": "modelo_disponible",
        "probabilidad": None,
        "mensaje": "Modelo de imagen detectado, pendiente de adaptar su preprocesado en el servidor.",
    }


def ejecutar_inferencia(datos_clinicos: dict[str, float], num_imagenes_adjuntas: int) -> tuple[float, float, dict]:
    contexto = obtener_contexto_tabular()
    rutas = _rutas_ml()
    fila = preparar_fila_prediccion(contexto, datos_clinicos)
    prob_tabular = float(contexto.modelo.predict_proba(fila)[0][1])
    resultado_imagen = procesar_resultado_imagen(rutas["modelo_imagen"], num_imagenes_adjuntas)

    prob_combinada = prob_tabular
    if resultado_imagen.get("estado") == "modelo_disponible" and resultado_imagen.get("probabilidad") is not None:
        prob_combinada = (0.6 * prob_tabular) + (0.4 * float(resultado_imagen["probabilidad"]))

    return prob_tabular, prob_combinada, resultado_imagen
