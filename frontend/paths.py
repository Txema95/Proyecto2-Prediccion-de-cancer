"""Rutas e importación del repositorio para poder usar paquetes de la raíz (p. ej. `ml`, `dl`)."""

from __future__ import annotations

import sys
from pathlib import Path


def raiz_repositorio() -> Path:
    return Path(__file__).resolve().parent.parent


def asegurar_sys_path_repo() -> None:
    r = str(raiz_repositorio())
    if r not in sys.path:
        sys.path.insert(0, r)
