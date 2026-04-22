"""Paso 2: carga de imagenes adjuntas al caso."""

import streamlit as st

from config import MAX_MB_POR_IMAGEN
import state


def render() -> None:
    st.subheader("2) Carga de imagenes")
    st.write("Adjunta una o varias imagenes (PNG/JPG/JPEG).")
    ficheros = st.file_uploader(
        "Selecciona imagenes",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    imagenes_validas = []
    if ficheros:
        for fichero in ficheros:
            tam_mb = len(fichero.getvalue()) / (1024 * 1024)
            if tam_mb > MAX_MB_POR_IMAGEN:
                st.error(f"{fichero.name}: supera {MAX_MB_POR_IMAGEN} MB y no se incluira.")
                continue
            imagenes_validas.append(fichero)
        st.session_state[state.IMAGENES] = imagenes_validas
        if imagenes_validas:
            st.success(f"Imagenes validas cargadas: {len(imagenes_validas)}")
            columnas = st.columns(min(3, len(imagenes_validas)))
            for i, fichero in enumerate(imagenes_validas):
                with columnas[i % len(columnas)]:
                    st.image(fichero, caption=fichero.name, use_container_width=True)
    col_izq, col_der = st.columns([1, 1])
    with col_izq:
        if st.button("Volver a datos clinicos", width="stretch"):
            st.session_state[state.PASO_ACTUAL] = 0
            st.rerun()
    with col_der:
        if st.button("Continuar a revision", width="stretch"):
            st.session_state[state.PASO_ACTUAL] = 2
            st.rerun()
