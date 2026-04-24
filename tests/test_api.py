"""Pruebas mínimas del API del simulador (health, predicción, validación de entrada).

Para ver trazas en consola: uv run pytest tests/test_api.py -v --log-cli-level=INFO
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app

logger = logging.getLogger(__name__)

OBJETIVO = "cancer_diagnosis"
COLUMNA_ID = "id"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_devuelve_ok(client: TestClient) -> None:
    logger.info("Inicio: GET /api/health")
    respuesta = client.get("/api/health")
    logger.info("Estado HTTP: %s, cuerpo: %s", respuesta.status_code, respuesta.text)
    assert respuesta.status_code == 200
    assert respuesta.json() == {"status": "ok"}
    logger.info("OK: health")


def _datos_desde_fila_muestra(ruta_csv: Path) -> dict[str, float]:
    logger.info("Leyendo primera fila de muestra desde: %s", ruta_csv)
    df = pd.read_csv(ruta_csv, sep=";", encoding="utf-8-sig")
    if df.empty:
        msg = "El CSV de muestra no contiene filas"
        raise AssertionError(msg)
    fila = df.iloc[0]
    casos: dict[str, float] = {}
    for clave in fila.index:
        if clave in (OBJETIVO, COLUMNA_ID):
            continue
        valor = fila[clave]
        if pd.isna(valor):
            continue
        casos[str(clave)] = float(valor)
    logger.info("Claves en datos_clinicos (%d): %s", len(casos), sorted(casos.keys()))
    return casos


def test_ejecutar_prediccion_tabulares_en_rango(client: TestClient, raiz: Path) -> None:
    ruta = raiz / "data" / "processed" / "cancer_final_clean_v2.csv"
    if not ruta.is_file():
        logger.warning("Omitido: no existe %s", ruta)
        pytest.skip("No hay dataset en data/processed/ (entorno mínimo o sin datos locales)")

    logger.info("Inicio: POST /ejecutar-prediccion (num_imagenes=0, datos desde CSV)")
    cuerpo = {
        "datos_clinicos": _datos_desde_fila_muestra(ruta),
        "num_imagenes_adjuntas": 0,
    }
    respuesta = client.post("/ejecutar-prediccion", json=cuerpo)
    logger.info("Estado HTTP: %s", respuesta.status_code)
    if respuesta.status_code != 200:
        logger.error("Cuerpo de error: %s", respuesta.text)
        pytest.fail(f"Respuesta inesperada {respuesta.status_code}: {respuesta.text}")
    carga = respuesta.json()
    prob_tab = float(carga["probabilidad_tabular"])
    prob_comb = float(carga["probabilidad_combinada"])
    logger.info(
        "Respuesta: prob_tabular=%.6f, prob_combinada=%.6f, imagen est=%s, msg=%s, prob_imagen=%s",
        prob_tab,
        prob_comb,
        carga.get("resultado_imagen", {}).get("estado"),
        carga.get("resultado_imagen", {}).get("mensaje", "")[:80],
        carga.get("resultado_imagen", {}).get("probabilidad"),
    )
    assert 0.0 <= prob_tab <= 1.0
    assert 0.0 <= prob_comb <= 1.0
    res_img = carga["resultado_imagen"]
    assert "estado" in res_img and "mensaje" in res_img
    logger.info("OK: prediccion tabular en rango y resultado_imagen coherente")


def test_ejecutar_prediccion_rechaza_valores_no_numericos(client: TestClient) -> None:
    cuerpo = {
        "datos_clinicos": {
            "age": "texto",
        },
        "num_imagenes_adjuntas": 0,
    }
    logger.info("Inicio: POST /ejecutar-prediccion con dato no numerico (age='texto')")
    respuesta = client.post("/ejecutar-prediccion", json=cuerpo)
    logger.info("Estado HTTP: %s, cuerpo: %s", respuesta.status_code, respuesta.text[:500])
    assert respuesta.status_code == 422
    detalle = respuesta.json()
    assert "detail" in detalle
    logger.info("OK: validacion 422 y detail presente")
