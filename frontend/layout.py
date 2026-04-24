"""Elementos de UI comunes a todas las vistas."""

import streamlit as st

from config import PASOS
import state


def pintar_encabezado() -> None:
    st.markdown(
        """
        <div class="clinical-notice">
        <strong>Uso académico y de apoyo.</strong>
        No sustituye el criterio clínico profesional ni constituye diagnóstico.
        </div>
        """,
        unsafe_allow_html=True,
    )


def pintar_progreso() -> None:
    paso_actual = int(st.session_state[state.PASO_ACTUAL])
    st.progress(
        (paso_actual + 1) / len(PASOS),
        text=f"Paso {paso_actual + 1}/{len(PASOS)}: {PASOS[paso_actual]}",
    )
