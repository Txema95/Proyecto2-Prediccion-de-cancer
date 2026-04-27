"""
Evalúa un `mejor_pesos.pt` sobre un split (p. ej. `test`).

Uso (raíz del repo):
    uv run python dl/vision_baseline_kvasir/evaluar.py --run dl/vision_baseline_kvasir/runs/resnet18_XXXX
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

if __name__ == "__main__" and __package__ is None:
    _r = Path(__file__).resolve().parents[2]
    if str(_r) not in sys.path:
        sys.path.insert(0, str(_r))

from dl.vision_baseline_kvasir.constantes import CLASES_ORDEN, indice_a_clase
from dl.vision_baseline_kvasir.dataset_torch import (
    DatasetKvasirMulticlase,
    transformaciones_imagenet_eval,
)
from dl.vision_baseline_kvasir.modelo_baseline import crear_resnet18, evaluar_cargador
from dl.vision_baseline_kvasir.paths import raiz_proyecto


def seleccionar_dispositivo(dispositivo_solicitado: str) -> torch.device:
    if dispositivo_solicitado == "cpu":
        return torch.device("cpu")
    if dispositivo_solicitado == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Se solicito CUDA, pero no esta disponible.")
        return torch.device("cuda")
    if dispositivo_solicitado == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Se solicito MPS, pero no esta disponible.")
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _ultimo_run(base: Path) -> Path:
    cands = sorted(
        (p for p in base.iterdir() if p.is_dir() and p.name.startswith("resnet18_")),
        key=lambda x: x.stat().st_mtime,
    )
    if not cands:
        raise FileNotFoundError(f"No hay carpetas resnet18_* en {base}.")
    return cands[-1]


def main() -> None:
    r = raiz_proyecto()
    p = argparse.ArgumentParser()
    p.add_argument(
        "--splits",
        type=Path,
        default=r / "data" / "processed" / "kvasir_min_clean" / "splits_kvasir_multiclase.csv",
    )
    p.add_argument(
        "--run",
        type=Path,
        default=None,
        help="Carpeta con mejor_pesos.pt (p. ej. resnet18_...).",
    )
    p.add_argument("--ultimo-run", action="store_true", help="Usa el run resnet18_* mas reciente.")
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "val", "test"),
    )
    p.add_argument("--batch", type=int, default=32)
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Hilos del DataLoader; 0 evita bloqueos al evaluar con pocos lotes o en algunos entornos.",
    )
    p.add_argument("--tam-imagen", type=int, default=224)
    p.add_argument("--dispositivo", type=str, default="auto", help="auto | mps | cpu | cuda")
    p.add_argument(
        "--sin-mps-fallback",
        action="store_true",
        help="Desactiva fallback MPS->CPU para operaciones no soportadas.",
    )
    a = p.parse_args()
    if not a.sin_mps_fallback:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    d = seleccionar_dispositivo(a.dispositivo)

    if a.ultimo_run and a.run is not None:
        raise SystemExit("No mezclar --run y --ultimo-run.")
    if a.ultimo_run or a.run is None:
        run = _ultimo_run(r / "dl" / "vision_baseline_kvasir" / "runs")
    else:
        run = Path(a.run)
    wpath = run / "mejor_pesos.pt"
    if not wpath.is_file():
        raise FileNotFoundError(f"No se encontro {wpath}")

    payload = torch.load(wpath, map_location=d, weights_only=False)
    n_c = int(payload.get("n_clases", len(CLASES_ORDEN)))
    modelo = crear_resnet18(n_c).to(d)
    modelo.load_state_dict(payload["modelo"])
    t_ev = transformaciones_imagenet_eval(a.tam_imagen)
    ds = DatasetKvasirMulticlase(a.splits, a.split, r, transform=t_ev)
    carga = DataLoader(
        ds,
        batch_size=a.batch,
        shuffle=False,
        num_workers=a.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=a.workers > 0,
    )
    yv, yp = evaluar_cargador(modelo, carga, d)
    f1m = f1_score(yv, yp, average="macro", zero_division=0)
    acc = accuracy_score(yv, yp)
    target = [indice_a_clase(int(i)) for i in range(n_c)]
    rep = classification_report(
        yv, yp, target_names=target, digits=4, zero_division=0
    )
    res = {
        "run": str(run),
        "split": a.split,
        "accuracy": acc,
        "f1_macro": f1m,
    }
    salida = run / f"metricas_{a.split}.json"
    salida.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    (run / f"reporte_clasificacion_{a.split}.txt").write_text(rep, encoding="utf-8")
    print(res, flush=True)
    print(salida, flush=True)


if __name__ == "__main__":
    main()
