"""Flujo simulador: formulario, imagen, revision y resultado."""

from __future__ import annotations

import streamlit as st

from layout import pintar_encabezado, pintar_progreso
from servicio_modelo import buscar_raiz_proyecto, cargar_dataset, obtener_rutas
import state
from views.carga_imagenes import render as vista_carga_imagenes
from views.datos_clinicos import render as vista_datos_clinicos
from views.resultado import render as vista_resultado
from views.revision_caso import render as vista_revision_caso


def render() -> None:
    pintar_encabezado()
    pintar_progreso()

    raiz = buscar_raiz_proyecto()
    rutas = obtener_rutas(raiz)
    datos = cargar_dataset(rutas["csv"])

    paso = int(st.session_state[state.PASO_ACTUAL])
    if paso == 0:
        vista_datos_clinicos(datos)
    elif paso == 1:
        vista_carga_imagenes()
    elif paso == 2:
        vista_revision_caso()
    else:
        vista_resultado()
