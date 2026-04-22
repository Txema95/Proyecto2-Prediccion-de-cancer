"""
Inferencia local con el baseline multiclase Kvasir (ResNet-18) para el simulador Streamlit.

Requiere `mejor_pesos.pt` bajo `ml/vision_baseline_kvasir/runs/resnet18_*/` o la ruta
en la variable de entorno `KVASIR_MODELO_PESOS`.
"""

from __future__ import annotations

import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import streamlit as st

# Nombres de presentacion (no son diagnosticos finales; el modelo es multiclase Kvasir v2)
NOMBRES_CLASE_KVASIR: dict[str, str] = {
    "normal-cecum": "Mucosa cecal / normal (Kvasir)",
    "polyps": "Pólipos (Kvasir)",
    "dyed-lifted-polyps": "Pólipos con tinción / levantados (Kvasir)",
    "ulcerative-colitis": "Colitis ulcerosa / inflamación (Kvasir)",
}


def asegurar_path_repo(raiz: Path) -> None:
    s = str(raiz.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


def resolver_ruta_pesos(raiz: Path) -> Path | None:
    env = os.environ.get("KVASIR_MODELO_PESOS", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p
    runs = raiz / "ml" / "vision_baseline_kvasir" / "runs"
    if not runs.is_dir():
        return None
    cands = sorted(
        (p for p in runs.glob("resnet18_*/mejor_pesos.pt") if p.is_file()),
        key=lambda x: x.stat().st_mtime,
    )
    return cands[-1] if cands else None


@st.cache_resource(show_spinner="Cargando modelo de vision (Kvasir, ResNet-18)...")
def _modelo_cargado(raiz_str: str, ruta_pesos_resuelta: str) -> Any:
    """Carga en memoria un solo modelo; clave de cache: raiz + ruta del checkpoint."""
    asegurar_path_repo(Path(raiz_str))
    import torch
    from ml.vision_baseline_kvasir.constantes import CLASES_ORDEN
    from ml.vision_baseline_kvasir.dataset_torch import transformaciones_imagenet_eval
    from ml.vision_baseline_kvasir.modelo_baseline import crear_resnet18

    ruta = Path(ruta_pesos_resuelta)
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(ruta, map_location=d, weights_only=False)
    n_c = int(payload.get("n_clases", len(CLASES_ORDEN)))
    modelo = crear_resnet18(n_c).to(d)
    modelo.load_state_dict(payload["modelo"])
    modelo.eval()
    transform = transformaciones_imagenet_eval(224)
    return {"modelo": modelo, "device": d, "transform": transform, "n_clases": n_c, "ruta": str(ruta)}


def predecir_bytes_imagen(raiz: Path, datos: bytes) -> dict[str, Any]:
    """
    Devuelve prediccion multiclase; si no hay checkpoint, `error` descriptivo.
    """
    ruta = resolver_ruta_pesos(raiz)
    if ruta is None:
        return {
            "error": "No se encontro `mejor_pesos.pt` en `ml/vision_baseline_kvasir/runs/`. "
            "Entrena el modelo o define KVASIR_MODELO_PESOS.",
        }
    try:
        paquete = _modelo_cargado(str(raiz.resolve()), str(ruta.resolve()))
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"No se pudo cargar el modelo: {exc}",
        }
    asegurar_path_repo(raiz)
    from ml.vision_baseline_kvasir.constantes import CLASES_ORDEN, indice_a_clase

    import torch
    from PIL import Image

    modelo = paquete["modelo"]
    d = paquete["device"]
    transform = paquete["transform"]
    n_c = int(paquete["n_clases"])
    im = Image.open(BytesIO(datos)).convert("RGB")
    t = transform(im).unsqueeze(0).to(d)
    with torch.inference_mode():
        logits = modelo(t)
        pr = torch.softmax(logits, dim=1).cpu().numpy().reshape(-1)
    k = int(pr.argmax())
    clave = indice_a_clase(k)
    nombre_largo = NOMBRES_CLASE_KVASIR.get(clave, clave)
    probs = {CLASES_ORDEN[i]: float(pr[i]) for i in range(n_c)}
    salida: dict[str, Any] = {
        "error": None,
        "id_clase": k,
        "clase_tecnica": clave,
        "clase_presentacion": nombre_largo,
        "confianza": float(pr[k]),
        "probabilidades": probs,
        "vector_prob": [float(pr[i]) for i in range(n_c)],
        "etiquetas_orden": list(CLASES_ORDEN[:n_c]),
        "ruta_pesos": str(ruta),
        "gradcam_error": None,
        "gradcam_superposicion": None,
    }
    try:
        from ml.vision_baseline_kvasir.gradcam import (
            grad_cam_resnet18,
            superponer_heatmap_sobre_imagen,
        )

        t_in = transform(im).unsqueeze(0).to(d)
        hw = int(t_in.shape[2])
        mapa = grad_cam_resnet18(modelo, t_in, k)
        sup = superponer_heatmap_sobre_imagen(im, mapa, tam=(hw, hw))
        salida["gradcam_superposicion"] = sup
    except Exception as exc:  # noqa: BLE001
        salida["gradcam_error"] = str(exc)
    return salida


def predecir_fichero_uploader(raiz: Path, fichero: Any) -> dict[str, Any]:
    """`fichero` es un objeto de Streamlit UploadedFile."""
    out = predecir_bytes_imagen(raiz, fichero.getvalue())
    out["archivo"] = getattr(fichero, "name", "imagen")
    return out
