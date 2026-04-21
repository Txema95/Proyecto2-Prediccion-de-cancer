"""Cliente HTTP hacia el backend del simulador."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from config import API_BASE_URL


def serializar_datos_clinicos(datos: dict) -> dict[str, float]:
    return {str(k): float(v) for k, v in datos.items()}


def ejecutar_prediccion_http(datos_clinicos: dict, num_imagenes_adjuntas: int) -> dict:
    url = f"{API_BASE_URL.rstrip('/')}/ejecutar-prediccion"
    cuerpo = json.dumps(
        {
            "datos_clinicos": serializar_datos_clinicos(datos_clinicos),
            "num_imagenes_adjuntas": int(num_imagenes_adjuntas),
        }
    ).encode("utf-8")
    solicitud = urllib.request.Request(
        url,
        data=cuerpo,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(solicitud, timeout=120) as respuesta:
            return json.loads(respuesta.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        texto = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Error del servidor ({error.code}): {texto}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"No se pudo conectar al API en {url}. ¿Esta arrancado el backend? Detalle: {error.reason}"
        ) from error
