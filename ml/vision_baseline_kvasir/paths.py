from __future__ import annotations

from pathlib import Path


def raiz_proyecto() -> Path:
    """Sube desde ml/vision_baseline_kvasir/<modulo>.py a la raíz del repositorio."""
    return Path(__file__).resolve().parents[2]
