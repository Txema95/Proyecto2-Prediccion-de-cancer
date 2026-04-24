# Pruebas del API del simulador

Esta carpeta contiene **pruebas automáticas** del backend FastAPI (`backend/app`) con **pytest**. Comprueban el endpoint de salud, la predicción tabular (si hay dataset local) y la validación de errores (HTTP 422).

## Requisitos

- Entorno con dependencias de desarrollo: en la raíz del repositorio ejecuta `uv sync --extra dev` (instala `pytest` y `httpx` definidos en `pyproject.toml` → extra `dev`).
- Sin el extra `dev`, `pytest` no estará en el venv: un `uv sync` **sin** `--extra dev` deja solo las dependencias de ejecución y **quita** los paquetes del grupo `dev`.

## Cómo ejecutar

Desde la **raíz** del repositorio (no desde `tests/`):

```bash
uv run pytest
```

Solo el fichero del API:

```bash
uv run pytest tests/test_api.py -v
```

**Ver trazas de log** mientras se ejecutan los tests (los `logger.info` de `test_api.py`):

```bash
uv run pytest tests/test_api.py -v --log-cli-level=INFO
```

No ejecutes el test como script directo (`uv run tests/test_api.py`): fallará porque el import `app` exige la ruta `backend` en el `PYTHONPATH`. Eso lo aplica **pytest** mediante la opción en `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["backend"]
```

## Qué hay en cada fichero

| Fichero | Rol |
|--------|-----|
| `conftest.py` | Fichero **común** de pytest. Define el fixture `raiz` (ruta a la raíz del repo) para no repetir cálculo de rutas. |
| `test_api.py` | Casos de prueba: `TestClient` de FastAPI sobre `app` (sin levantar Uvicorn en un puerto), `GET /api/health`, `POST /ejecutar-prediccion` y comprobación de 422. Incluye **logs** informativos para depurar el flujo. |

## Comportamiento de los tests

1. **Salud:** comprueba `GET /api/health` y que la respuesta sea 200 con `{"status": "ok"}`.

2. **Predicción tabular:** si existe `data/processed/cancer_final_clean_v2.csv`, construye el cuerpo a partir de la **primera fila** del CSV (datos consistentes con el entrenamiento) y llama a `POST /ejecutar-prediccion` con `num_imagenes_adjuntas: 0`. Comprueba probabilidades en [0, 1] y la estructura de `resultado_imagen`.  
   Si el CSV no está (p. ej. clon del repo sin datos generados), el test se **omite** (`skip`).

3. **Validación:** envía un valor no numérico y espera **422** (validación del endpoint).

Más detalle de arquitectura y entregables: `../docs/entrega-p2-simulador.md` (sección de pruebas).
