"""
Punto de entrada Streamlit: simulador, panel ML (tabular) y panel DL (Kvasir).

Configuración de pagina, cabecera con pestañas, vistas en `views/`.

Arranque solo UI:
    uv run streamlit run frontend/app.py

Con API y simulador completo (misma consola, desde la raiz del repo):
    uv run python main.py
"""

from __future__ import annotations

import streamlit as st

from config import CABECERA_TITULO, LAYOUT, PAGE_TITLE, TAB_LABELS
from estilos_clinicos import aplicar_tema_clinico
from paths import asegurar_sys_path_repo
from state import inicializar_estado
from views.explorador_dl import render as render_dl
from views.explorador_ml import render as render_ml
from views.portal_simulador import render as render_simulador

asegurar_sys_path_repo()


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        layout=LAYOUT,
        initial_sidebar_state="collapsed",
    )
    aplicar_tema_clinico()
    inicializar_estado()

    st.markdown(
        f'<p class="app-cabecera-titulo">{CABECERA_TITULO}</p>',
        unsafe_allow_html=True,
    )
    t_consulta, t_datos, t_imagen = st.tabs(TAB_LABELS)

    with t_consulta:
        render_simulador()
    with t_datos:
        render_ml()
    with t_imagen:
        render_dl()


if __name__ == "__main__":
    main()
