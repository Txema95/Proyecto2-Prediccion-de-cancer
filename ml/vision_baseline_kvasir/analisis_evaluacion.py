"""
Analisis extendido: matriz de confusion, curvas ROC (one-vs-rest), probabilidades y
comprobaciones (pares de confusion, confianza, entropia de la salida).

Uso (raiz del repo):
    uv run python ml/vision_baseline_kvasir/analisis_evaluacion.py --ultimo-run --split test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

if __name__ == "__main__" and __package__ is None:
    _r = Path(__file__).resolve().parents[2]
    if str(_r) not in sys.path:
        sys.path.insert(0, str(_r))

from ml.vision_baseline_kvasir.constantes import CLASES_ORDEN, indice_a_clase
from ml.vision_baseline_kvasir.dataset_torch import (
    DatasetKvasirMulticlase,
    transformaciones_imagenet_eval,
)
from ml.vision_baseline_kvasir.modelo_baseline import (
    crear_resnet18,
    inferencia_con_probabilidades,
)
from ml.vision_baseline_kvasir.paths import raiz_proyecto


def _ultimo_run(base: Path) -> Path:
    cands = sorted(
        (p for p in base.iterdir() if p.is_dir() and p.name.startswith("resnet18_")),
        key=lambda x: x.stat().st_mtime,
    )
    if not cands:
        raise FileNotFoundError(f"No hay carpetas resnet18_* en {base}.")
    return cands[-1]


def _entropia(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def _pares_confusion_top(
    y_true: np.ndarray, y_pred: np.ndarray, n_clases: int, k: int = 12
) -> list[dict[str, Any]]:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_clases))
    pares: list[tuple[int, int, int]] = []
    for i in range(n_clases):
        for j in range(n_clases):
            if i != j and cm[i, j] > 0:
                pares.append((i, j, int(cm[i, j])))
    pares.sort(key=lambda t: t[2], reverse=True)
    out: list[dict[str, Any]] = []
    for i, j, c in pares[:k]:
        out.append(
            {
                "verdadero": indice_a_clase(i),
                "predicho": indice_a_clase(j),
                "conteo": c,
            }
        )
    return out


def _fig_matriz_conteos(cm: np.ndarray, etiquetas: list[str], titulo: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=etiquetas,
        yticklabels=etiquetas,
        ylabel="Verdadero",
        xlabel="Predicho",
        title=titulo,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=9,
            )
    fig.tight_layout()
    return fig


def _fig_matriz_por_fila(cm: np.ndarray, etiquetas: list[str], titulo: str) -> plt.Figure:
    s = cm.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1, s)
    p = cm.astype(float) / s
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(p, interpolation="nearest", cmap=plt.cm.Oranges, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fracc. sobre verdadero")
    ax.set(
        xticks=np.arange(p.shape[1]),
        yticks=np.arange(p.shape[0]),
        xticklabels=etiquetas,
        yticklabels=etiquetas,
        ylabel="Verdadero",
        xlabel="Predicho",
        title=titulo,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            ax.text(j, i, f"{p[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")
    fig.tight_layout()
    return fig


def _fig_roc_multiclase(
    y_true: np.ndarray, probs: np.ndarray, n_c: int, etiquetas: list[str]
) -> plt.Figure:
    y_bin = label_binarize(y_true, classes=np.arange(n_c))
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), sharex=True, sharey=True)
    axes_r = axes.ravel()
    for k in range(n_c):
        ax = axes_r[k]
        fpr, tpr, _ = roc_curve(y_bin[:, k], probs[:, k])
        auc_c = float(auc(fpr, tpr)) if len(fpr) > 1 and len(fpr) == len(tpr) else float("nan")
        ax.plot(fpr, tpr, label=f"AUC={auc_c:.3f}", linewidth=1.5)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(etiquetas[k], fontsize=9)
        ax.set_xlabel("FPR", fontsize=8)
        ax.set_ylabel("TPR", fontsize=8)
        ax.legend(loc="lower right", fontsize=7)
    fig.suptitle("Curvas ROC (one-vs-rest)", fontsize=12)
    fig.tight_layout()
    return fig


def _fig_confianza(
    p_max: np.ndarray, acierto: np.ndarray, titulo: str
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        p_max[acierto],
        bins=25,
        alpha=0.6,
        label="aciertos",
        color="tab:blue",
        density=True,
    )
    ax.hist(
        p_max[~acierto],
        bins=25,
        alpha=0.6,
        label="errores",
        color="tab:red",
        density=True,
    )
    ax.set_xlabel("max(prob) en la prediccion elegida")
    ax.set_ylabel("Densidad")
    ax.set_title(titulo)
    ax.legend()
    fig.tight_layout()
    return fig


def _fig_entropia(e: np.ndarray, acierto: np.ndarray, titulo: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(e[acierto], bins=25, alpha=0.6, label="aciertos", color="tab:blue", density=True)
    ax.hist(e[~acierto], bins=25, alpha=0.6, label="errores", color="tab:red", density=True)
    ax.set_xlabel("Entropia de la distribucion predicha (nats)")
    ax.set_ylabel("Densidad")
    ax.set_title(titulo)
    ax.legend()
    fig.tight_layout()
    return fig


def main() -> None:
    r = raiz_proyecto()
    p = argparse.ArgumentParser(
        description="Matriz de confusion, ROC, probabilidades y comprobaciones extra."
    )
    p.add_argument(
        "--splits",
        type=Path,
        default=r / "data" / "processed" / "kvasir_min_clean" / "splits_kvasir_multiclase.csv",
    )
    p.add_argument("--run", type=Path, default=None, help="Carpeta con mejor_pesos.pt.")
    p.add_argument("--ultimo-run", action="store_true")
    p.add_argument(
        "--split", type=str, default="test", choices=("train", "val", "test")
    )
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--tam-imagen", type=int, default=224)
    p.add_argument("--dispositivo", type=str, default="auto")
    p.add_argument(
        "--salida-subdir",
        type=str,
        default="",
        help="Subcarpeta bajo el run; por defecto analisis_<split>.",
    )
    p.add_argument(
        "--no-csv-muestras",
        action="store_true",
        help="No generar CSV largo con probabilidad por imagen.",
    )
    a = p.parse_args()

    if a.dispositivo == "auto":
        d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        d = torch.device(a.dispositivo)

    if a.ultimo_run and a.run is not None:
        raise SystemExit("No mezclar --run y --ultimo-run.")
    if a.ultimo_run or a.run is None:
        run = _ultimo_run(r / "ml" / "vision_baseline_kvasir" / "runs")
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
    )
    y_true, y_pred, probs, image_ids = inferencia_con_probabilidades(modelo, carga, d)
    n = len(y_true)
    if len(image_ids) != n:
        # DataLoader pudo colar otra forma de batch; aseguramos longitud
        if len(image_ids) < n:
            image_ids = (image_ids + [""] * n)[:n]
        else:
            image_ids = list(image_ids)[:n]

    etiquetas = [indice_a_clase(i) for i in range(n_c)]
    p_max = probs[np.arange(n), y_pred]
    acierto = y_true == y_pred
    entro = _entropia(probs)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1m_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc_mc = float(
            roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=np.arange(n_c))
        )
        auc_por_clase: dict[str, float] = {}
        y_bin = label_binarize(y_true, classes=np.arange(n_c))
        for c in range(n_c):
            if y_bin[:, c].sum() == 0 or y_bin[:, c].sum() == len(y_true):
                auc_por_clase[etiquetas[c]] = float("nan")
            else:
                auc_por_clase[etiquetas[c]] = float(roc_auc_score(y_bin[:, c], probs[:, c]))
    except (ValueError, TypeError) as e:
        auc_mc = float("nan")
        auc_por_clase = {"_error_": str(e)}

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_c))
    pares = _pares_confusion_top(y_true, y_pred, n_c, k=16)

    sub = a.salida_subdir.strip() or f"analisis_{a.split}"
    out_dir = run / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        out_dir / "matriz_confusion_conteos.csv", cm, fmt="%d", delimiter=","
    )
    buf = StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([""] + etiquetas)
    for i, fila in enumerate(cm):
        w.writerow([etiquetas[i]] + [int(x) for x in fila.tolist()])
    (out_dir / "matriz_confusion_conteos_legible.csv").write_text(
        buf.getvalue(), encoding="utf-8-sig"
    )

    resumen: dict[str, Any] = {
        "run": str(run),
        "split": a.split,
        "n_muestras": n,
        "accuracy": acc,
        "f1_macro": f1m,
        "f1_ponderada": f1m_w,
        "roc_auc_macro_ovr": auc_mc,
        "roc_auc_por_clase_ovr": auc_por_clase,
        "p_max_aciertos_media": float(p_max[acierto].mean()) if acierto.any() else None,
        "p_max_errores_media": float(p_max[~acierto].mean()) if (~acierto).any() else None,
        "entropia_media": float(entro.mean()),
        "entropia_aciertos_media": float(entro[acierto].mean()) if acierto.any() else None,
        "entropia_errores_media": float(entro[~acierto].mean()) if (~acierto).any() else None,
        "n_errores": int((~acierto).sum()),
        "pares_confusion_principales": pares,
    }
    (out_dir / "resumen_analisis.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if not a.no_csv_muestras:
        lineas = ["image_id,y_true,y_true_nombre,y_pred,y_pred_nombre,acierto," + ",".join(f"prob_c{k}" for k in range(n_c)) + ",p_max,entropia\n"]
        for t in range(n):
            yv, pr = int(y_true[t]), int(y_pred[t])
            row = [
                str(image_ids[t] if t < len(image_ids) else ""),
                str(yv),
                indice_a_clase(yv),
                str(pr),
                indice_a_clase(pr),
                str(bool(acierto[t])),
            ] + [f"{probs[t, k]:.6f}" for k in range(n_c)]
            row += [f"{p_max[t]:.6f}", f"{entro[t]:.6f}"]
            lineas.append(",".join(row) + "\n")
        (out_dir / "probabilidades_por_muestra.csv").write_text("".join(lineas), encoding="utf-8")

    _fig_matriz_conteos(cm, etiquetas, f"Matriz de confusion (conteos) — {a.split} (n={n})").savefig(
        out_dir / "matriz_confusion.png", dpi=150
    )
    plt.close("all")
    _fig_matriz_por_fila(
        cm, etiquetas, f"Matriz normalizada por fila (verdadero) — {a.split}"
    ).savefig(out_dir / "matriz_confusion_normalizada_fila.png", dpi=150)
    plt.close("all")
    _fig_roc_multiclase(y_true, probs, n_c, etiquetas).savefig(
        out_dir / "curvas_roc_uno_contra_resto.png", dpi=150
    )
    plt.close("all")
    _fig_confianza(
        p_max, acierto, f"Distrib. confianza max(prob) — {a.split}"
    ).savefig(out_dir / "confianza_maxima_aciertos_vs_errores.png", dpi=150)
    plt.close("all")
    _fig_entropia(
        entro, acierto, f"Entropia de la prediccion — {a.split}"
    ).savefig(out_dir / "entropia_aciertos_vs_errores.png", dpi=150)
    plt.close("all")

    print(json.dumps({k: resumen[k] for k in ("split", "n_muestras", "accuracy", "f1_macro", "roc_auc_macro_ovr", "n_errores")}, ensure_ascii=False), flush=True)
    print("Salida:", out_dir, flush=True)


if __name__ == "__main__":
    main()
