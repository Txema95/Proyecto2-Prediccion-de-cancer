"""
Estilos globales para un aspecto de interfaz de salud: azul confianza (eHealth) y
verde agua / menta (asociados a calma y bienestar en muchas guías de UI clínica).

Solo CSS inyectado; el tono concreto también se refuerza en `.streamlit/config.toml`.
"""

from __future__ import annotations

import streamlit as st


def aplicar_tema_clinico() -> None:
    st.markdown(
        """
        <style>
        /* Paleta: #0D4A6E (azul clínico), #1B7B6B (verde agua OMS/estilo hospitalario suave) */
        :root {
            --clin-azul: #0D4A6E;
            --clin-azul-claro: #1A6FA0;
            --clin-mint: #1B7B6B;
            --clin-fondo-1: #F0F5F9;
            --clin-fondo-2: #FAFCFD;
        }
        .stApp,
        [data-testid="stAppViewContainer"] > .main {
            background: linear-gradient(150deg, var(--clin-fondo-1) 0%, var(--clin-fondo-2) 50%, #E9F0F4 100%) !important;
        }
        [data-testid="stAppViewContainer"] {
            background: transparent !important;
        }
        .app-cabecera-titulo {
            font-size: 1.4rem;
            font-weight: 650;
            color: #0A3A55;
            margin: 0 0 0.2rem 0;
            padding: 0.5rem 0.5rem 0.65rem 1rem;
            border-left: 4px solid var(--clin-mint);
            background: linear-gradient(90deg, rgba(27, 123, 107, 0.1) 0%, rgba(13, 74, 110, 0.04) 55%, transparent 100%);
            border-radius: 0 12px 12px 0;
        }
        .clinical-notice {
            background: linear-gradient(90deg, #E3EEF4 0%, #F0F6F8 100%);
            border-left: 4px solid var(--clin-azul);
            padding: 0.85rem 1.1rem;
            border-radius: 0 12px 12px 0;
            color: #1a2330;
            font-size: 0.95rem;
            margin: 0.4rem 0 1rem 0;
            box-shadow: 0 1px 4px rgba(13, 74, 110, 0.07);
        }
        .clinical-notice strong { color: #082c42; }
        /* Pestañas principales */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.4rem;
            background: rgba(255, 255, 255, 0.6);
            padding: 0.25rem 0.25rem 0 0.25rem;
            border-radius: 12px 12px 0 0;
        }
        [data-testid="stTabs"] [data-baseweb="tab"] {
            border-radius: 10px 10px 0 0;
            font-weight: 500;
        }
        [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] p {
            color: #ffffff !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(180deg, #0D4A6E 0%, #0A3550 100%) !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="false"] {
            background: rgba(255, 255, 255, 0.75) !important;
        }
        /* Barra de progreso del simulador */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #1B7B6B, #1A6FA0) !important;
        }
        /* Títulos en el cuerpo */
        .main h1, .main h2, .main h3 { color: #0A3A55; }
        .main a { color: #0D4A6E; }
        </style>
        """,
        unsafe_allow_html=True,
    )
