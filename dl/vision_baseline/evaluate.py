from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

try:
    from dl.vision_baseline.dataset import DatasetColonoscopiaBinario
except ModuleNotFoundError:
    # Permite ejecutar el script desde dl/vision_baseline sin depender del paquete "dl".
    from dataset import DatasetColonoscopiaBinario


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evalua checkpoint en split test y genera reporte completo. "
            "Usa --ultimo-run para tomar el best_checkpoint.pt del entrenamiento mas reciente."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Ruta a best_checkpoint.pt (relativa a la raiz del proyecto o absoluta).",
    )
    parser.add_argument(
        "--ultimo-run",
        action="store_true",
        help="Usa dl/vision_baseline/runs/<carpeta_mas_reciente>/best_checkpoint.pt",
    )
    parser.add_argument("--splits-csv", type=Path, default=Path("data/processed/splits.csv"))
    parser.add_argument("--modelo", type=str, default="resnet50", choices=["resnet50", "mobilenet_v2"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--umbral", type=float, default=0.5)
    args = parser.parse_args()
    if not args.ultimo_run and args.checkpoint is None:
        parser.error("Indica --checkpoint RUTA o bien --ultimo-run.")
    if args.ultimo_run and args.checkpoint is not None:
        parser.error("No combines --ultimo-run con --checkpoint.")
    return args


def obtener_raiz_proyecto() -> Path:
    return Path(__file__).resolve().parents[2]


def resolver_ruta_checkpoint(raiz: Path, checkpoint: Path | None, ultimo_run: bool) -> Path:
    if ultimo_run:
        carpeta_runs = raiz / "dl" / "vision_baseline" / "runs"
        if not carpeta_runs.is_dir():
            raise FileNotFoundError(
                f"No existe la carpeta de runs: {carpeta_runs}. Entrena primero con train.py."
            )
        candidatos: list[tuple[float, Path]] = []
        for sub in carpeta_runs.iterdir():
            if not sub.is_dir():
                continue
            ruta_pt = sub / "best_checkpoint.pt"
            if ruta_pt.is_file():
                candidatos.append((ruta_pt.stat().st_mtime, ruta_pt))
        if not candidatos:
            raise FileNotFoundError(
                f"No hay ningun best_checkpoint.pt bajo {carpeta_runs}. "
                f"Ejecuta antes: uv run python dl/vision_baseline/train.py ..."
            )
        return max(candidatos, key=lambda par: par[0])[1]

    assert checkpoint is not None
    ruta = (raiz / checkpoint).resolve() if not checkpoint.is_absolute() else checkpoint
    texto_ruta = str(ruta)
    texto_arg = checkpoint.as_posix()
    if "<run>" in texto_ruta or "<run>" in texto_arg:
        raise FileNotFoundError(
            f"La ruta parece un ejemplo sin sustituir: {ruta}\n"
            f"Sustituye <run> por el nombre real de la carpeta bajo dl/vision_baseline/runs/ "
            f"(p. ej. resnet50_20260420_123456) o ejecuta:\n"
            f"  uv run python dl/vision_baseline/evaluate.py --ultimo-run --modelo resnet50"
        )
    return ruta


def crear_modelo_binario(nombre_modelo: str, device: torch.device) -> nn.Module:
    if nombre_modelo == "resnet50":
        modelo = models.resnet50(weights=None)
        modelo.fc = nn.Linear(modelo.fc.in_features, 1)
    else:
        modelo = models.mobilenet_v2(weights=None)
        modelo.classifier[1] = nn.Linear(modelo.classifier[1].in_features, 1)
    return modelo.to(device)


def transformar_eval():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@torch.no_grad()
def predecir_probabilidades(modelo: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    modelo.eval()
    probabilidades = []
    etiquetas = []
    for imagenes, y in loader:
        imagenes = imagenes.to(device)
        logits = modelo(imagenes)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        probabilidades.extend(probs.tolist())
        etiquetas.extend(y.numpy().tolist())
    return np.array(etiquetas, dtype=int), np.array(probabilidades, dtype=float)


def guardar_matriz_confusion(y_true: np.ndarray, y_pred: np.ndarray, ruta_salida: Path) -> None:
    matriz = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5.6, 4.6))
    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["Real 0", "Real 1"],
    )
    plt.title("Matriz de confusion - Test")
    plt.xlabel("Prediccion")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=140)
    plt.close()


def main() -> None:
    args = parsear_argumentos()
    raiz = obtener_raiz_proyecto()
    ruta_splits = (raiz / args.splits_csv).resolve() if not args.splits_csv.is_absolute() else args.splits_csv
    ruta_checkpoint = resolver_ruta_checkpoint(
        raiz, checkpoint=args.checkpoint, ultimo_run=args.ultimo_run
    )
    if not ruta_checkpoint.exists():
        raise FileNotFoundError(f"No existe checkpoint: {ruta_checkpoint}")

    if args.ultimo_run:
        print(f"Checkpoint elegido (--ultimo-run): {ruta_checkpoint}")

    carpeta_salida = ruta_checkpoint.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = crear_modelo_binario(args.modelo, device)
    payload = torch.load(ruta_checkpoint, map_location=device)
    modelo.load_state_dict(payload["model_state_dict"])

    dataset_test = DatasetColonoscopiaBinario(
        ruta_proyecto=raiz,
        ruta_splits_csv=ruta_splits,
        split="test",
        transform=transformar_eval(),
    )
    loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    y_true, y_prob = predecir_probabilidades(modelo, loader_test, device)
    y_pred = (y_prob >= args.umbral).astype(int)

    metricas = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "umbral": args.umbral,
        "falsos_negativos": int(np.sum((y_true == 1) & (y_pred == 0))),
        "total_positivos_reales": int(np.sum(y_true == 1)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }

    (carpeta_salida / "metricas_test_detalladas.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    reporte = classification_report(y_true, y_pred, digits=4, zero_division=0)
    (carpeta_salida / "reporte_test.txt").write_text(reporte, encoding="utf-8")
    guardar_matriz_confusion(y_true, y_pred, carpeta_salida / "matriz_confusion_test.png")

    print("Evaluacion completada.")
    print(f"- Carpeta artefactos: {carpeta_salida}")
    print(f"- Recall test: {metricas['recall']:.4f}")
    print(f"- F1 test: {metricas['f1']:.4f}")
    print(f"- ROC-AUC test: {metricas['roc_auc']:.4f}")
    print(f"- Falsos negativos: {metricas['falsos_negativos']} / {metricas['total_positivos_reales']}")


if __name__ == "__main__":
    main()
