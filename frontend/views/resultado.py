"""Paso 4: resultado de inferencia tabular, imagen y combinado."""

import streamlit as st

import state
from visor_imagen import mostrar_imagen_centrada
from servicio_modelo import (
    ejecutar_prediccion,
    etiqueta_desde_probabilidad,
    tipo_riesgo_terciles,
)


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

    pred_k = st.session_state.get(state.PRED_KVASIR)
    pred_k_validas: list = []
    if pred_k and isinstance(pred_k, list):
        for p in pred_k:
            if p and not p.get("error"):
                pred_k_validas.append(p)
    prob_max_conf: float | None
    if pred_k_validas:
        prob_max_conf = max(float(p.get("confianza", 0) or 0) for p in pred_k_validas)
    else:
        prob_max_conf = None

    col_tab, col_img = st.columns(2, width="stretch")
    with col_tab:
        st.markdown("**Resumen — datos tabulares (API)**")
        st.metric("Probabilidad estimada (cáncer)", f"{prob_tabular:.1%}", border=True)
        st.write("**Tipo de riesgo (terciles):**", tipo_riesgo_terciles(prob_tabular))
        st.caption(etiqueta_desde_probabilidad(prob_tabular))
    with col_img:
        st.markdown("**Resumen — imágenes (Kvasir, local)**")
        if pred_k_validas and prob_max_conf is not None:
            st.metric("Máx. confianza (clase predicha)", f"{prob_max_conf:.1%}", border=True)
            st.write("**Tipo de riesgo (terciles, sobre la confianza):**", tipo_riesgo_terciles(prob_max_conf))
            for p in pred_k_validas:
                nom = p.get("archivo", "?")
                tipo = p.get("clase_presentacion", p.get("clase_tecnica", "")) or "—"
                conf = float(p.get("confianza", 0) or 0)
                riesgo_i = tipo_riesgo_terciles(conf)
                st.markdown(
                    f"- **Archivo:** `{nom}`  \n"
                    f"  **Clase (modelo):** {tipo}  \n"
                    f"  **Confianza:** {conf:.1%}  \n"
                    f"  **Tipo de riesgo:** {riesgo_i}"
                )
        elif pred_k and isinstance(pred_k, list) and all(p and p.get("error") for p in pred_k):
            st.warning("Comprobación Kvasir no disponible; define `KVASIR_MODELO_PESOS` o entrena un run.")
        else:
            st.info("Carga imágenes en el paso 2 para el análisis Kvasir (modelo local).")
        msg_api = resultado_imagen.get("mensaje") or ""
        if msg_api:
            st.caption(f"Nota API: {msg_api}")
        if resultado_imagen.get("probabilidad") is not None:
            st.caption(f"Prob. imagen (API, si aplica): {float(resultado_imagen['probabilidad']):.1%}")

    st.subheader("Vista detallada — visión (Kvasir, local)")
    if pred_k and isinstance(pred_k, list) and any(
        p and not p.get("error") for p in pred_k
    ):
        st.markdown("**Vision (Kvasir, local)**")
        for p in pred_k:
            if not p:
                continue
            if p.get("error"):
                st.warning(f"{p.get('archivo', 'archivo')}: {p['error']}")
            else:
                st.write(
                    f"**{p.get('archivo', '?')}** → {p.get('clase_presentacion', '')} "
                    f"({p.get('confianza', 0):.0%})"
                )
                if p.get("gradcam_superposicion") is not None:
                    mostrar_imagen_centrada(
                        p["gradcam_superposicion"],
                        caption="Grad-CAM (clase predicha)",
                        rellenar_ancho_bloque=True,
                    )
                elif p.get("gradcam_error"):
                    st.caption(f"Grad-CAM: {p['gradcam_error'][:120]}")
    else:
        st.caption("Sin análisis Kvasir en detalle o análisis no disponible.")

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
