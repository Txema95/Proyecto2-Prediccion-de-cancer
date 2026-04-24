"""Configuración compartida de tests (rutas al repositorio)."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def raiz() -> Path:
    return Path(__file__).resolve().parent.parent
