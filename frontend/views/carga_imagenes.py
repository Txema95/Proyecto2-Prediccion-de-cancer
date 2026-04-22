"""Paso 2: carga de imagenes adjuntas al caso y comprobacion con el modelo Kvasir (multiclase)."""

import pandas as pd
import streamlit as st

from config import MAX_MB_POR_IMAGEN
import state
from servicio_modelo import buscar_raiz_proyecto
from servicio_vision_kvasir import NOMBRES_CLASE_KVASIR, predecir_fichero_uploader


def _mostrar_prediccion_kvasir(p: dict) -> None:
    nombre = p.get("archivo", "imagen")
    if p.get("error"):
        st.error(f"**{nombre}:** {p['error']}")
        return
    st.markdown(f"**{nombre}**")
    pp = p.get("preprocesado") or {}
    vp = p.get("vista_previa_preprocesado")
    if pp.get("aplicado"):
        st.markdown("##### Preprocesado (alineado con el entrenamiento)")
        st.caption(
            "Mismo flujo que `kvasir_preprocesado_minimo.py`: bordes negros, recorte cuadrado, "
            "salida 512×512; el modelo aplica despues 224×224 + normalizacion ImageNet."
        )
        col_pp1, col_pp2 = st.columns(2)
        with col_pp1:
            if vp is not None:
                st.image(
                    vp,
                    caption="Imagen enviada al modelo (tras preprocesado; ver tamano en JSON)",
                    use_container_width=True,
                )
        with col_pp2:
            st.json(
                {
                    "size_preprocesado_px": pp.get("size_salida"),
                    "umbral_negro": pp.get("umbral_negro"),
                    "padding_fraccion": pp.get("padding_fraccion"),
                    "recorte_borde_negro": pp.get("recorte_borde_negro"),
                    "pixeles_recortados_borde": pp.get("pixeles_recortados_borde"),
                    "original_wh": f"{pp.get('ancho_original')}x{pp.get('alto_original')}",
                    "tras_recorte_borde_wh": f"{pp.get('ancho_tras_borde')}x{pp.get('alto_tras_borde')}",
                }
            )
    else:
        st.info(
            pp.get("motivo", "Preprocesado minimo no aplicado.")
            + " Puedes activarlo quitando `KVASIR_SIN_PREPROCESADO` del entorno."
        )
    st.success(
        f"Clase predicha: **{p['clase_presentacion']}** "
        f"— confianza: {p['confianza']:.1%}"
    )
    if p.get("ruta_pesos"):
        st.caption(f"Checkpoint: `{p['ruta_pesos']}`")
    filas = [
        {
            "Categoria (Kvasir)": NOMBRES_CLASE_KVASIR.get(k, k),
            "Probabilidad": v,
        }
        for k, v in p.get("probabilidades", {}).items()
    ]
    if filas:
        df = pd.DataFrame(filas).sort_values("Probabilidad", ascending=False)
        df["Prob."] = df["Probabilidad"].map(lambda x: f"{x:.1%}")
        st.dataframe(
            df[["Categoria (Kvasir)", "Prob."]],
            width="stretch",
            hide_index=True,
        )
    gc = p.get("gradcam_superposicion")
    ge = p.get("gradcam_error")
    if ge:
        st.warning(f"Grad-CAM no disponible: {ge}")
    elif gc is not None:
        st.markdown("**Grad-CAM** (zonas que mas influyen en la clase predicha arriba)")
        st.caption(
            "Colormap *jet*: rojo = mayor peso en la decision para esa clase. "
            "Imagen internamente a 224 px como en el entrenamiento."
        )
        st.image(gc, caption="Superposicion Grad-CAM + imagen", use_container_width=True)


def render() -> None:
    st.subheader("2) Carga de imagenes")
    st.write(
        "Adjunta una o varias imagenes (PNG/JPG/JPEG). Cada imagen pasa por el **preprocesado minimo** "
        "(`kvasir_preprocesado_minimo`: recorte de viñeta, cuadrado, 512 px) y despues el modelo **ResNet-18** "
        "Kvasir (4 clases). Solo orientacion; no es diagnostico medico."
    )
    try:
        raiz = buscar_raiz_proyecto()
    except FileNotFoundError as err:
        st.error(str(err))
        return

    ficheros = st.file_uploader(
        "Selecciona imagenes",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if not ficheros:
        st.session_state[state.IMAGENES] = []
        st.session_state[state.PRED_KVASIR] = None
    else:
        imagenes_validas: list = []
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

            with st.spinner("Comprobando imagenes con el modelo de vision (Kvasir)..."):
                predicciones = [predecir_fichero_uploader(raiz, f) for f in imagenes_validas]
            st.session_state[state.PRED_KVASIR] = predicciones

            st.divider()
            st.markdown("**Comprobacion con el modelo (baseline Kvasir, ResNet-18)**")
            for pred in predicciones:
                with st.container(border=True):
                    _mostrar_prediccion_kvasir(pred)
        else:
            st.session_state[state.PRED_KVASIR] = None

    col_izq, col_der = st.columns([1, 1])
    with col_izq:
        if st.button("Volver a datos clinicos", width="stretch"):
            st.session_state[state.PASO_ACTUAL] = 0
            st.rerun()
    with col_der:
        if st.button("Continuar a revision", width="stretch"):
            st.session_state[state.PASO_ACTUAL] = 2
            st.rerun()
