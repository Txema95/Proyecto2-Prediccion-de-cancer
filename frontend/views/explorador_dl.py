"""Panel DL: runs de vision (Kvasir) bajo `dl/vision_baseline_kvasir/runs/`."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from servicio_modelo import buscar_raiz_proyecto


def _cargar_json(ruta: Path) -> dict | list | None:
    if not ruta.is_file():
        return None
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _listar_runs(raiz: Path) -> list[Path]:
    base = raiz / "dl" / "vision_baseline_kvasir" / "runs"
    if not base.is_dir():
        return []
    return sorted(
        (p for p in base.iterdir() if p.is_dir() and p.name.startswith("resnet18_")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def render() -> None:
    st.subheader("Lo aprendido a partir de imágenes de endoscopía")
    st.caption(
        "Revisa entrenamientos anteriores (configuración, métricas y, si existen, análisis de test). "
        "En la simulación se usa el modelo entrenado más reciente."
    )

    raiz = buscar_raiz_proyecto()
    runs = _listar_runs(raiz)

    with st.expander("Resumen de datos e imagenes (proyecto)"):
        resumen = raiz / "data" / "resumen_diagnostico_imagen.md"
        if resumen.is_file():
            st.markdown(resumen.read_text(encoding="utf-8"))
        else:
            st.info(f"No se encontro `{resumen.name}` (opcional).")

    st.subheader("Entrenamientos recientes")
    if not runs:
        st.warning(
            "No hay carpetas `resnet18_*` bajo `dl/vision_baseline_kvasir/runs/`. "
            "Entrena con `dl/vision_baseline_kvasir/entrenar.py` o copia un run al repositorio."
        )
        return

    nombres = [p.name for p in runs]
    elegido = st.selectbox("Run a inspeccionar", options=nombres, index=0)
    run = next(p for p in runs if p.name == elegido)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**`config.json`**")
        cfg = _cargar_json(run / "config.json")
        if cfg is None:
            st.info("Sin config.json")
        else:
            st.json(cfg)
    with c2:
        st.markdown("**`metricas_test.json` (si existe)**")
        m = _cargar_json(run / "metricas_test.json")
        if m is None:
            st.info("Sin metricas de test en este run.")
        else:
            st.json(m)

    analisis_dir = run / "analisis_test" / "resumen_analisis.json"
    st.markdown("**Analisis de test (si se genero con `evaluar` / `analisis_evaluacion`)**")
    a = _cargar_json(analisis_dir)
    if a is None:
        st.info(f"No hay `{analisis_dir.name}` en este run.")
    else:
        st.json(a)

    pesos = run / "mejor_pesos.pt"
    st.markdown("**Checkpoint**")
    if pesos.is_file():
        st.success(f"`{pesos}` ({pesos.stat().st_size // 1024} KiB aprox.)")
    else:
        st.info("No hay `mejor_pesos.pt` en este run.")
