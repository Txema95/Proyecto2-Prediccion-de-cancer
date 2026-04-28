"""Paso 2: carga de imagenes adjuntas al caso (inferencia Kvasir al generar el resultado)."""

import streamlit as st

from config import MAX_MB_POR_IMAGEN
import state
from visor_imagen import mostrar_imagen_centrada

# Ancho fijo (px) para previsualizaciones; evita que una sola imagen ocupe toda la ventana.
_ANCHO_VISTA_PREVIA_PX = 360


def _firma_imagenes(ficheros: list) -> tuple[tuple[str, int], ...]:
    return tuple((str(f.name), int(f.size)) for f in ficheros)


def render() -> None:
    st.subheader("2) Carga de imagenes")
    st.write(
        "Adjunta una o varias imagenes (PNG/JPG/JPEG). La comprobacion con el modelo **Kvasir** "
        "(ResNet-18, multiclase) se ejecuta al **obtener el resultado** (mismo flujo que la prediccion tabular vía API). "
        "Ahi veras clases, confianza, Grad-CAM y el resumen clínico. "
        "Solo orientacion; no es diagnostico medico."
    )

    ficheros = st.file_uploader(
        "Selecciona imagenes",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if not ficheros:
        st.session_state[state.IMAGENES] = []
        st.session_state[state.PRED_KVASIR] = None
        st.session_state[state.PRED_KVASIR_FIRMA] = None
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
            firma_actual = _firma_imagenes(imagenes_validas)
            st.success(f"Imagenes validas cargadas: {len(imagenes_validas)}")
            columnas = st.columns(min(3, len(imagenes_validas)))
            for i, fichero in enumerate(imagenes_validas):
                with columnas[i % len(columnas)]:
                    mostrar_imagen_centrada(
                        fichero,
                        caption=fichero.name,
                        ancho_px=_ANCHO_VISTA_PREVIA_PX,
                    )
            firma_prev = st.session_state.get(state.PRED_KVASIR_FIRMA)
            if firma_prev != firma_actual:
                st.session_state[state.PRED_KVASIR] = None
                if firma_prev is not None:
                    st.info(
                        "Has cambiado las imagenes. Al **generar el resultado** se volverá a inferir con Kvasir."
                    )
        else:
            st.session_state[state.PRED_KVASIR] = None
            st.session_state[state.PRED_KVASIR_FIRMA] = None

    col_izq, col_der = st.columns([1, 1])
    with col_izq:
        if st.button("Volver a datos clinicos", width="stretch"):
            st.session_state[state.PASO_ACTUAL] = 0
            st.rerun()
    with col_der:
        if st.button("Continuar a revision", width="stretch"):
            st.session_state[state.PASO_ACTUAL] = 2
            st.rerun()
