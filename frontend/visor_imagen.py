"""Mostrar imágenes centradas en Streamlit (evita alinear solo a la izquierda en columnas)."""

from __future__ import annotations

import base64
import html
import io
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image


def _a_data_uri(imagen: Any) -> str:
    if hasattr(imagen, "getvalue") and callable(imagen.getvalue):
        data = imagen.getvalue()
        nombre = str(getattr(imagen, "name", "") or "").lower()
        if nombre.endswith((".jpg", ".jpeg")):
            mime = "image/jpeg"
        else:
            mime = "image/png"
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    if isinstance(imagen, np.ndarray):
        arr = np.asarray(imagen)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255)
            if arr.size and float(np.nanmax(arr)) <= 1.0 + 1e-6:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        pil = Image.fromarray(arr)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    if isinstance(imagen, Image.Image):
        buf = io.BytesIO()
        imagen.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    raise TypeError(f"Tipo no soportado para vista centrada: {type(imagen)!r}")


def mostrar_imagen_centrada(
    imagen: Any,
    *,
    caption: str | None = None,
    ancho_px: int = 360,
    rellenar_ancho_bloque: bool = False,
) -> None:
    uri = _a_data_uri(imagen)
    cap_html = ""
    if caption:
        cap_html = (
            f'<p style="margin:0.35rem 0 0 0;font-size:0.88rem;color:#444;">'
            f"{html.escape(caption)}</p>"
        )
    if rellenar_ancho_bloque:
        img = (
            f'<img src="{uri}" alt="" '
            'style="height:auto;display:block;margin:0 auto;" />'
        )
    else:
        img = f'<img src="{uri}" width="{int(ancho_px)}" alt="" />'
    st.markdown(
        f'<div style="text-align:center;width:100%;">'
        f"{img}{cap_html}</div>",
        unsafe_allow_html=True,
    )
