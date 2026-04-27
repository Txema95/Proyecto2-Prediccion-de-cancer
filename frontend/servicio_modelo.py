"""Datos locales para el formulario (CSV) y delegacion de inferencia al backend."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import state
from config import DECISION_THRESHOLD
from servicio_api import ejecutar_prediccion_http


def buscar_raiz_proyecto() -> Path:
    carpeta_actual = Path(__file__).resolve().parent
    for ruta in [carpeta_actual, *carpeta_actual.parents]:
        if (ruta / "data" / "processed" / "cancer_final_clean_v2.csv").is_file():
            return ruta
    raise FileNotFoundError("No se encontro data/processed/cancer_final_clean_v2.csv")


def obtener_rutas(raiz: Path) -> dict[str, Path]:
    return {
        "csv": raiz / "data" / "processed" / "cancer_final_clean_v2.csv",
    }


@st.cache_data(show_spinner=False)
def cargar_dataset(ruta_csv: Path) -> pd.DataFrame:
    return pd.read_csv(ruta_csv, sep=";", encoding="utf-8-sig")


def normalizar_valor(valor: float, minimo: float, maximo: float) -> float:
    if maximo == minimo:
        return 0.0
    return (valor - minimo) / (maximo - minimo)


def etiqueta_desde_probabilidad(probabilidad: float, umbral: float = DECISION_THRESHOLD) -> str:
    return "Riesgo alto (positivo)" if probabilidad >= umbral else "Riesgo bajo (negativo)"


def ejecutar_prediccion(datos_formulario: dict[str, float], imagenes: list) -> None:
    respuesta = ejecutar_prediccion_http(datos_formulario, len(imagenes))
    st.session_state[state.PROB_TABULAR] = float(respuesta["probabilidad_tabular"])
    st.session_state[state.RESULTADO_COMBINADO] = float(respuesta["probabilidad_combinada"])
    ri = respuesta["resultado_imagen"]
    st.session_state[state.RESULTADO_IMAGEN] = {
        "estado": ri["estado"],
        "mensaje": ri["mensaje"],
        "probabilidad": ri.get("probabilidad"),
    }
    st.session_state[state.PASO_ACTUAL] = 3
