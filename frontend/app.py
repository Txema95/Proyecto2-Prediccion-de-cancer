"""
Punto de entrada Streamlit del simulador.

Este fichero concentra la configuracion de pagina, constantes de layout
compartidas y el despacho al paso actual (vistas en `views/`).

Ejecucion:
    uv run streamlit run frontend/app.py
"""

from __future__ import annotations

import streamlit as st

from config import LAYOUT, PAGE_TITLE
from layout import pintar_encabezado, pintar_progreso
from servicio_modelo import buscar_raiz_proyecto, cargar_dataset, obtener_rutas
from state import inicializar_estado
from views.carga_imagenes import render as vista_carga_imagenes
from views.datos_clinicos import render as vista_datos_clinicos
from views.resultado import render as vista_resultado
from views.revision_caso import render as vista_revision_caso
import state


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout=LAYOUT)
    inicializar_estado()
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


if __name__ == "__main__":
    main()
