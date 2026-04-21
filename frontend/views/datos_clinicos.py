"""Paso 1: introduccion de datos clinicos tabulares."""

import pandas as pd
import streamlit as st

from formulario_clinico import dibujar_formulario_datos
import state


def render(datos: pd.DataFrame) -> None:
    st.subheader("1) Datos clinicos")
    st.write("Introduce historial y resultados de pruebas.")
    formulario = dibujar_formulario_datos(datos)
    col_izq, col_der = st.columns([1, 1])
    with col_izq:
        if st.button("Guardar y continuar", width="stretch"):
            st.session_state[state.DATOS_FORMULARIO] = formulario
            st.session_state[state.PASO_ACTUAL] = 1
            st.rerun()
    with col_der:
        if st.button("Limpiar formulario", width="stretch"):
            st.session_state[state.DATOS_FORMULARIO] = {}
            st.rerun()
