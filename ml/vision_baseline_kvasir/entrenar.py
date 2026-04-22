"""
Entrena un baseline multiclase (ResNet-18) sobre `splits_kvasir_multiclase.csv`.

Uso (raíz del repo):
    uv run python ml/vision_baseline_kvasir/entrenar.py --epocas 10
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.optim import Adam
from torch.utils.data import DataLoader

if __name__ == "__main__" and __package__ is None:
    _r = Path(__file__).resolve().parents[2]
    if str(_r) not in sys.path:
        sys.path.insert(0, str(_r))

from ml.vision_baseline_kvasir.constantes import CLASES_ORDEN, indice_a_clase
from ml.vision_baseline_kvasir.dataset_torch import (
    DatasetKvasirMulticlase,
    transformaciones_imagenet_entrenamiento,
    transformaciones_imagenet_eval,
)
from ml.vision_baseline_kvasir.modelo_baseline import crear_resnet18, evaluar_cargador
from ml.vision_baseline_kvasir.paths import raiz_proyecto


@dataclass
class ConfigEntrenar:
    splits_csv: str
    raiz: str
    salida: str
    epocas: int
    batch: int
    lr: float
    weight_decay: float
    semilla: int
    workers: int
    tam_imagen: int
    dispositivo: str


def fijar_semillas(semilla: int) -> None:
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


def construir_cargador(
    splits: Path,
    particion: str,
    raiz: Path,
    transform: object,
    batch: int,
    workers: int,
    barajar: bool,
) -> DataLoader:
    ds = DatasetKvasirMulticlase(splits, particion, raiz, transform=transform)
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=barajar,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def bucle_epoca(
    modelo: nn.Module,
    cargador: DataLoader,
    criterio: nn.Module,
    optim: Adam,
    dispositivo: torch.device,
    epoca: int,
    epocas_totales: int,
    log_cada_lotes: int,
) -> float:
    modelo.train()
    perdida_acum = 0.0
    n = 0
    n_lotes = len(cargador)
    for ind_lote, lote in enumerate(cargador, start=1):
        x = lote["x"].to(dispositivo, non_blocking=True)
        y = lote["y"].to(dispositivo, non_blocking=True)
        optim.zero_grad()
        salida = modelo(x)
        l = criterio(salida, y)
        l.backward()
        optim.step()
        perdida_acum += float(l.item()) * y.size(0)
        n += y.size(0)
        if log_cada_lotes > 0 and ind_lote % log_cada_lotes == 0:
            print(
                f"    lote {ind_lote}/{n_lotes}  perdida_inst={float(l.item()):.4f}",
                flush=True,
            )
    prom = perdida_acum / max(1, n)
    print(
        f"  Fin entrenamiento epoca {epoca}/{epocas_totales}: "
        f"perdida_media_train={prom:.4f} ({n} imagenes)",
        flush=True,
    )
    return prom


def main() -> None:
    r = raiz_proyecto()
    p = argparse.ArgumentParser(description="Baseline Kvasir multiclase (ResNet-18).")
    p.add_argument(
        "--splits",
        type=Path,
        default=r / "data" / "processed" / "kvasir_min_clean" / "splits_kvasir_multiclase.csv",
    )
    p.add_argument("--output-dir", type=Path, default=r / "ml" / "vision_baseline_kvasir" / "runs")
    p.add_argument("--epocas", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Hilos del DataLoader. 0 suele ser mas estable (sin subprocesos).",
    )
    p.add_argument("--tam-imagen", type=int, default=224, help="Resize de entrada (ImageNet).")
    p.add_argument("--dispositivo", type=str, default="auto", help="auto | cpu | cuda")
    p.add_argument(
        "--log-cada-lotes",
        type=int,
        default=25,
        help="Cada cuantos lotes mostrar progreso en train; 0 lo desactiva.",
    )
    a = p.parse_args()

    if a.dispositivo == "auto":
        dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dispositivo = torch.device(a.dispositivo)
    fijar_semillas(a.semilla)

    print("=== Entrenamiento vision_baseline_kvasir (ResNet-18) ===", flush=True)
    print(f"  splits: {a.splits}", flush=True)
    print(f"  dispositivo: {dispositivo}", flush=True)
    print(f"  batch={a.batch}  workers={a.workers}  epocas={a.epocas}", flush=True)

    tr_t = transformaciones_imagenet_entrenamiento(a.tam_imagen)
    ev_t = transformaciones_imagenet_eval(a.tam_imagen)

    print("Construyendo DataLoaders (lectura de CSV + dataset)...", flush=True)
    car_t = construir_cargador(
        a.splits, "train", r, tr_t, a.batch, a.workers, True
    )
    car_v = construir_cargador(
        a.splits, "val", r, ev_t, a.batch, a.workers, False
    )
    n_train = len(car_t.dataset)
    n_val = len(car_v.dataset)
    print(
        f"  train: {n_train} imagenes -> {len(car_t)} lotes | val: {n_val} -> {len(car_v)} lotes",
        flush=True,
    )

    n_clases = len(CLASES_ORDEN)
    print(
        "Cargando ResNet-18 con pesos ImageNet (la primera vez puede descargar ~45 MB)...",
        flush=True,
    )
    modelo = crear_resnet18(n_clases).to(dispositivo)
    criterio = nn.CrossEntropyLoss()
    optim = Adam(modelo.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    carpe = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = a.output_dir / f"resnet18_{carpe}"
    run.mkdir(parents=True, exist_ok=True)
    cfg = ConfigEntrenar(
        splits_csv=a.splits.as_posix(),
        raiz=r.as_posix(),
        salida=run.as_posix(),
        epocas=a.epocas,
        batch=a.batch,
        lr=a.lr,
        weight_decay=a.weight_decay,
        semilla=a.semilla,
        workers=a.workers,
        tam_imagen=a.tam_imagen,
        dispositivo=str(dispositivo),
    )
    (run / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Run: {run} (config guardado)", flush=True)

    mejor_f1 = -1.0
    historial: list[dict[str, float]] = []
    for ep in range(1, a.epocas + 1):
        print(f"--- Epoca {ep}/{a.epocas} ---", flush=True)
        p_tr = bucle_epoca(
            modelo,
            car_t,
            criterio,
            optim,
            dispositivo,
            epoca=ep,
            epocas_totales=a.epocas,
            log_cada_lotes=a.log_cada_lotes,
        )
        print("  Validando (val)...", flush=True)
        yv, yp = evaluar_cargador(modelo, car_v, dispositivo)
        f1m = f1_score(yv, yp, average="macro", zero_division=0)
        acc = accuracy_score(yv, yp)
        historial.append({"epoca": float(ep), "perdida_train": p_tr, "f1_val_macro": f1m, "acc_val": acc})
        if f1m > mejor_f1:
            mejor_f1 = f1m
            extra = "  (nuevo mejor F1 en val, checkpoint guardado)"
            torch.save(
                {
                    "modelo": modelo.state_dict(),
                    "f1_val_macro": f1m,
                    "n_clases": n_clases,
                },
                run / "mejor_pesos.pt",
            )
        else:
            extra = ""
        print(f"  Val: f1_macro={f1m:.4f}  acc={acc:.4f}{extra}", flush=True)
    (run / "historial.json").write_text(
        json.dumps(historial, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    ultimo = run / "mejor_pesos.pt"
    if not ultimo.is_file():
        torch.save(
            {
                "modelo": modelo.state_dict(),
                "f1_val_macro": float("nan"),
                "n_clases": n_clases,
            },
            ultimo,
        )
    d = torch.load(ultimo, map_location=dispositivo, weights_only=False)
    modelo.load_state_dict(d["modelo"])
    print("Evaluando en test con los mejores pesos (val)...", flush=True)
    car_test = construir_cargador(
        a.splits, "test", r, ev_t, a.batch, a.workers, False
    )
    print(f"  test: {len(car_test.dataset)} imagenes, {len(car_test)} lotes", flush=True)
    yt, yp = evaluar_cargador(modelo, car_test, dispositivo)
    f1m_t = f1_score(yt, yp, average="macro", zero_division=0)
    f1p_t = f1_score(yt, yp, average="micro", zero_division=0)
    acc_t = accuracy_score(yt, yp)
    target = [indice_a_clase(int(i)) for i in range(n_clases)]
    reporte = classification_report(
        yt, yp, target_names=target, digits=4, zero_division=0
    )
    test_out: dict[str, Any] = {
        "accuracy": acc_t,
        "f1_macro": f1m_t,
        "f1_micro": f1p_t,
    }
    (run / "metricas_test.json").write_text(
        json.dumps(test_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run / "reporte_clasificacion_test.txt").write_text(reporte, encoding="utf-8")
    print("Test: ", test_out, flush=True)
    print("Run: ", run, flush=True)


if __name__ == "__main__":
    main()
