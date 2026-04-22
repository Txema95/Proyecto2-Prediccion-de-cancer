"""Formulario de variables clinicas tabulares."""

import pandas as pd
import streamlit as st

from config import COLUMNA_ID, OBJETIVO
from labels import etiqueta_columna, etiqueta_valor_columna
import state


def dibujar_formulario_datos(datos: pd.DataFrame) -> dict[str, float]:
    columnas_base = [col for col in datos.columns if col not in {COLUMNA_ID, OBJETIVO}]
    formulario = {}
    columnas_ui = st.columns(3)
    for indice, columna in enumerate(columnas_base):
        serie = datos[columna].dropna()
        minimo = float(serie.min())
        maximo = float(serie.max())
        mediana = float(serie.median())
        if serie.nunique() <= 8 and minimo >= 0 and maximo <= 5:
            opciones = sorted({int(x) for x in serie.unique()})
            valor_inicial = int(st.session_state[state.DATOS_FORMULARIO].get(columna, int(mediana)))
            with columnas_ui[indice % 3]:
                formulario[columna] = st.selectbox(
                    etiqueta_columna(columna),
                    options=opciones,
                    format_func=lambda opcion, col=columna: etiqueta_valor_columna(col, opcion),
                    index=opciones.index(valor_inicial) if valor_inicial in opciones else 0,
                )
        else:
            valor_guardado = float(st.session_state[state.DATOS_FORMULARIO].get(columna, mediana))
            with columnas_ui[indice % 3]:
                if columna == "age":
                    formulario[columna] = st.number_input(
                        etiqueta_columna(columna),
                        value=int(round(valor_guardado)),
                        min_value=int(round(minimo)),
                        max_value=int(round(maximo)),
                        step=1,
                    )
                else:
                    formulario[columna] = st.number_input(
                        etiqueta_columna(columna),
                        value=float(valor_guardado),
                        min_value=minimo,
                        max_value=maximo,
                        step=max(0.1, (maximo - minimo) / 100),
                        format="%.3f",
                    )
    return formulario
