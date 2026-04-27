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


def _firma_imagenes_cargadas(imagenes: list) -> tuple[tuple[str, int], ...]:
    return tuple((str(f.name), int(f.size)) for f in imagenes)


def _predicciones_kvasir_sobre_imagenes(raiz: Path, imagenes: list) -> list[dict]:
    from servicio_vision_kvasir import predecir_fichero_uploader

    predicciones: list[dict] = []
    for f in imagenes:
        try:
            predicciones.append(predecir_fichero_uploader(raiz, f))
        except Exception as exc:  # noqa: BLE001
            predicciones.append(
                {"archivo": getattr(f, "name", "imagen"), "error": str(exc)}
            )
    return predicciones


def tipo_riesgo_terciles(probabilidad: float) -> str:
    """Tres franjas 0-33, 33-66, 66-100% sobre la probabilidad o confianza mostrada."""
    p = max(0.0, min(1.0, float(probabilidad)))
    if p < 1.0 / 3.0:
        return "Bajo (tercil inferior de la puntuación)"
    if p < 2.0 / 3.0:
        return "Moderado (tercil intermedio)"
    return "Alto (tercil superior de la puntuación)"


def ejecutar_prediccion(datos_formulario: dict[str, float], imagenes: list) -> None:
    """
    Llama al API (`/ejecutar-prediccion` tabular) y, si hay imagenes, ejecuta en local
    el baseline Kvasir (misma accion, tras la respuesta HTTP; no se envian bytes al backend).
    """
    respuesta = ejecutar_prediccion_http(datos_formulario, len(imagenes))
    st.session_state[state.PROB_TABULAR] = float(respuesta["probabilidad_tabular"])
    st.session_state[state.RESULTADO_COMBINADO] = float(respuesta["probabilidad_combinada"])
    ri = respuesta["resultado_imagen"]
    st.session_state[state.RESULTADO_IMAGEN] = {
        "estado": ri["estado"],
        "mensaje": ri["mensaje"],
        "probabilidad": ri.get("probabilidad"),
    }
    if imagenes:
        raiz = buscar_raiz_proyecto()
        st.session_state[state.PRED_KVASIR] = _predicciones_kvasir_sobre_imagenes(raiz, imagenes)
        st.session_state[state.PRED_KVASIR_FIRMA] = _firma_imagenes_cargadas(imagenes)
    else:
        st.session_state[state.PRED_KVASIR] = None
        st.session_state[state.PRED_KVASIR_FIRMA] = None
    st.session_state[state.PASO_ACTUAL] = 3
