"""Panel ML: baselines tabulares (artefactos bajo `ml/`), reentrenamiento bajo demanda."""

from __future__ import annotations

import streamlit as st

from paths import asegurar_sys_path_repo
from servicio_modelo import buscar_raiz_proyecto


@st.cache_resource
def _modulo_ml() -> object:
    asegurar_sys_path_repo()
    from ml import main as ml_mod

    return ml_mod


def render() -> None:
    st.subheader("Señales del historial, en cifras")
    st.caption(
        "Aquí se comparan distintas formas de puntuar el riesgo a partir del mismo registro. "
        "Puedes ver resultados guardados o volver a entrenar con el boton de abajo."
    )

    ml = _modulo_ml()
    raiz = buscar_raiz_proyecto()
    rutas = ml.obtener_rutas(raiz)

    st.subheader("Variables derivadas")
    with st.expander("Definicion de `n_sintomas` y `riesgo_familiar_x_edad`"):
        st.write("Features incorporadas al entrenamiento:")
        st.json(ml.FEATURES_GENERADAS)
        for nombre, texto in ml.EXPLICACION_FEATURES.items():
            st.markdown(f"- `{nombre}`: {texto}")

    st.divider()
    st.subheader("Comparacion de resultados (test)")

    if st.button("Entrenar y evaluar de nuevo los cuatro modelos", type="primary"):
        with st.spinner("Entrenando y evaluando (puede tardar unos minutos)..."):
            try:
                ml.entrenar_todos_los_baselines(rutas)
            except Exception as e:
                st.error(f"Error durante el entrenamiento: {e}")
            else:
                st.success("Entrenamiento y evaluacion completados. Las tablas e imagenes se actualizan abajo.")

    tabla = ml.construir_tabla_comparacion(rutas)
    if tabla.empty:
        st.info("Aun no hay metricas de test en `ml/*/v1/artefactos/`. Pulsa el boton de entrenar.")
    else:
        st.dataframe(tabla, use_container_width=True)
        mejor = tabla.iloc[0]
        st.success(
            f"Mejor PR-AUC en test: {mejor['modelo']} "
            f"(pr_auc={mejor['pr_auc']:.4f}, recall={mejor['recall']:.4f})"
        )

    st.subheader("Matrices de confusion (test)")
    columnas = st.columns(4)
    for i, nombre_modelo in enumerate(["regresion_logistica", "random_forest", "xgboost", "svm"]):
        with columnas[i]:
            st.caption(f"**{nombre_modelo}**")
            artefactos = ml.cargar_artefactos_modelo(rutas[nombre_modelo])
            if artefactos is None:
                st.info("Sin artefactos.")
            elif artefactos["ruta_matriz"].exists():
                st.image(str(artefactos["ruta_matriz"]))
            else:
                st.info("Sin matriz de confusion generada.")
