"""Elementos de UI comunes a todas las vistas."""

import streamlit as st

from config import PASOS
import state


def pintar_encabezado() -> None:
    st.title("Simulador de apoyo al diagnostico de cancer de colon")
    st.caption(
        "Herramienta academica de apoyo a decision. No sustituye el criterio medico profesional."
    )
    st.warning("Uso educativo: la salida del simulador no constituye diagnostico clinico.")


def pintar_progreso() -> None:
    paso_actual = int(st.session_state[state.PASO_ACTUAL])
    st.progress(
        (paso_actual + 1) / len(PASOS),
        text=f"Paso {paso_actual + 1}/{len(PASOS)}: {PASOS[paso_actual]}",
    )
