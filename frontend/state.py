"""Estado de sesion Streamlit compartido entre vistas."""

import streamlit as st

PASO_ACTUAL = "paso_actual"
DATOS_FORMULARIO = "datos_formulario"
IMAGENES = "imagenes"
PROB_TABULAR = "prob_tabular"
RESULTADO_IMAGEN = "resultado_imagen"
RESULTADO_COMBINADO = "resultado_combinado"


def inicializar_estado() -> None:
    if PASO_ACTUAL not in st.session_state:
        st.session_state[PASO_ACTUAL] = 0
    if DATOS_FORMULARIO not in st.session_state:
        st.session_state[DATOS_FORMULARIO] = {}
    if IMAGENES not in st.session_state:
        st.session_state[IMAGENES] = []
    if PROB_TABULAR not in st.session_state:
        st.session_state[PROB_TABULAR] = None
    if RESULTADO_IMAGEN not in st.session_state:
        st.session_state[RESULTADO_IMAGEN] = None
    if RESULTADO_COMBINADO not in st.session_state:
        st.session_state[RESULTADO_COMBINADO] = None


def reiniciar_caso() -> None:
    st.session_state[DATOS_FORMULARIO] = {}
    st.session_state[IMAGENES] = []
    st.session_state[PROB_TABULAR] = None
    st.session_state[RESULTADO_IMAGEN] = None
    st.session_state[RESULTADO_COMBINADO] = None
    st.session_state[PASO_ACTUAL] = 0


def limpiar_prediccion() -> None:
    st.session_state[PROB_TABULAR] = None
    st.session_state[RESULTADO_IMAGEN] = None
    st.session_state[RESULTADO_COMBINADO] = None
