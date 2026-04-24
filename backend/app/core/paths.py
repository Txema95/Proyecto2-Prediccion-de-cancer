"""Resolucion de rutas respecto a la raiz del repositorio."""

from pathlib import Path


def raiz_proyecto() -> Path:
    """Raiz del repo (donde estan `data/`, `ml/`, `dl/`, `frontend/`)."""
    # backend/app/core/paths.py -> parents[3] = raiz del monorepo
    return Path(__file__).resolve().parents[3]
