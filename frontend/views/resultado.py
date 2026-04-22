"""Paso 4: resultado de inferencia tabular, imagen y combinado."""

import pandas as pd
import streamlit as st

import state
from servicio_modelo import ejecutar_prediccion, etiqueta_desde_probabilidad, normalizar_valor


def render() -> None:
    if st.session_state[state.PROB_TABULAR] is None:
        with st.spinner("Calculando prediccion (servidor)..."):
            try:
                ejecutar_prediccion(
                    datos_formulario=st.session_state[state.DATOS_FORMULARIO],
                    imagenes=st.session_state[state.IMAGENES],
                )
            except RuntimeError as error:
                st.error(str(error))
                st.info("Arranca el API con: uv run uvicorn app.main:app --reload --app-dir backend")
                if st.button("Volver a revision", width="stretch"):
                    st.session_state[state.PASO_ACTUAL] = 2
                    state.limpiar_prediccion()
                    st.rerun()
                return
            st.rerun()

    st.subheader("4) Resultado")
    prob_tabular = float(st.session_state[state.PROB_TABULAR])
    resultado_imagen = st.session_state[state.RESULTADO_IMAGEN]
    prob_combinada = float(st.session_state[state.RESULTADO_COMBINADO])

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Probabilidad tabular", f"{prob_tabular:.2%}", border=True)
        st.write(etiqueta_desde_probabilidad(prob_tabular))
    with col_b:
        if resultado_imagen["estado"] == "sin_imagen":
            st.info("Sin prediccion por imagen")
        elif resultado_imagen["estado"] == "modelo_no_disponible":
            st.warning("Modelo de imagen no integrado")
        else:
            st.info("Modelo de imagen detectado (integracion pendiente)")
        st.caption(resultado_imagen["mensaje"])
    with col_c:
        st.metric("Probabilidad combinada", f"{prob_combinada:.2%}", border=True)
        st.write(etiqueta_desde_probabilidad(prob_combinada))

    datos = st.session_state[state.DATOS_FORMULARIO]
    if datos:
        edad = float(datos.get("age", 0.0))
        sof = float(datos.get("sof", 0.0))
        tenesmus = float(datos.get("tenesmus", 0.0))
        rectorrhagia = float(datos.get("rectorrhagia", 0.0))
        sintomas = sof + tenesmus + rectorrhagia
        riesgo_familiar = float(datos.get("digestive_family_risk_level", 0.0))
        explicacion = pd.DataFrame(
            {
                "factor": ["edad", "n_sintomas", "riesgo_familiar"],
                "valor_normalizado": [
                    normalizar_valor(edad, 18, 95),
                    normalizar_valor(sintomas, 0, 3),
                    normalizar_valor(riesgo_familiar, 0, 3),
                ],
            }
        )
        st.write("Indicadores clinicos resumidos del caso:")
        st.bar_chart(explicacion.set_index("factor"), width="stretch")

    col_izq, col_der = st.columns([1, 1])
    with col_izq:
        if st.button("Volver a revision", width="stretch"):
            st.session_state[state.PASO_ACTUAL] = 2
            state.limpiar_prediccion()
            st.rerun()
    with col_der:
        if st.button("Nuevo caso", width="stretch"):
            state.reiniciar_caso()
            st.rerun()
