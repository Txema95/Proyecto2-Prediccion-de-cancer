"""Paso 3: revision del caso antes de inferencia."""

import pandas as pd
import streamlit as st

from labels import etiqueta_columna
import state


def render() -> None:
    st.subheader("3) Revision del caso")
    st.write("Revisa informacion antes de ejecutar la prediccion.")
    datos = st.session_state[state.DATOS_FORMULARIO]
    if not datos:
        st.error("No hay datos clinicos guardados.")
    else:
        datos_visuales = {etiqueta_columna(clave): valor for clave, valor in datos.items()}
        st.dataframe(pd.DataFrame([datos_visuales]), width="stretch")
    st.write(f"Imagenes adjuntas: {len(st.session_state[state.IMAGENES])}")
    confirmacion = st.checkbox(
        "Confirmo que entiendo que esta herramienta solo ofrece apoyo academico y no diagnostico definitivo."
    )
    col_izq, col_centro, col_der = st.columns([1, 1, 1])
    with col_izq:
        if st.button("Volver a imagenes", width="stretch"):
            st.session_state[state.PASO_ACTUAL] = 1
            st.rerun()
    with col_centro:
        if st.button("Ejecutar prediccion", width="stretch", disabled=not confirmacion):
            st.session_state[state.PASO_ACTUAL] = 3
            st.rerun()
    with col_der:
        if st.button("Reiniciar caso", width="stretch"):
            state.reiniciar_caso()
            st.rerun()
