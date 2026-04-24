from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import models, transforms

try:
    from dl.vision_baseline.dataset import DatasetColonoscopiaBinario
except ModuleNotFoundError:
    # Permite ejecutar el script desde dl/vision_baseline sin depender del paquete "dl".
    from dataset import DatasetColonoscopiaBinario


@dataclass
class ConfigEntrenamiento:
    modelo: str
    ruta_splits: str
    salida_dir: str
    epocas: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    semilla: int
    workers: int
    metrica_checkpoint: str


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena baseline de clasificacion de polipos con splits.csv.")
    parser.add_argument("--splits-csv", type=Path, default=Path("data/processed/splits.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("dl/vision_baseline/runs"))
    parser.add_argument("--modelo", type=str, default="resnet50", choices=["resnet50", "mobilenet_v2"])
    parser.add_argument("--epocas", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metrica-checkpoint",
        type=str,
        default="val_recall",
        choices=["val_recall", "val_f1"],
        help="Metrica para decidir el mejor checkpoint.",
    )
    return parser.parse_args()


def establecer_semilla_global(semilla: int) -> None:
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


def obtener_raiz_proyecto() -> Path:
    return Path(__file__).resolve().parents[2]


def construir_transformaciones():
    normalizacion = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
            transforms.ToTensor(),
            normalizacion,
        ]
    )
    transform_eval = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalizacion,
        ]
    )
    return transform_train, transform_eval


def crear_modelo_binario(nombre_modelo: str, device: torch.device) -> nn.Module:
    if nombre_modelo == "resnet50":
        pesos = models.ResNet50_Weights.DEFAULT
        modelo = models.resnet50(weights=pesos)
        in_features = modelo.fc.in_features
        modelo.fc = nn.Linear(in_features, 1)
    else:
        pesos = models.MobileNet_V2_Weights.DEFAULT
        modelo = models.mobilenet_v2(weights=pesos)
        in_features = modelo.classifier[1].in_features
        modelo.classifier[1] = nn.Linear(in_features, 1)
    return modelo.to(device)


def construir_dataloaders(
    ruta_proyecto: Path,
    ruta_splits_csv: Path,
    batch_size: int,
    workers: int,
) -> dict[str, DataLoader]:
    transform_train, transform_eval = construir_transformaciones()
    dataset_train = DatasetColonoscopiaBinario(
        ruta_proyecto=ruta_proyecto,
        ruta_splits_csv=ruta_splits_csv,
        split="train",
        transform=transform_train,
    )
    dataset_val = DatasetColonoscopiaBinario(
        ruta_proyecto=ruta_proyecto,
        ruta_splits_csv=ruta_splits_csv,
        split="val",
        transform=transform_eval,
    )
    dataset_test = DatasetColonoscopiaBinario(
        ruta_proyecto=ruta_proyecto,
        ruta_splits_csv=ruta_splits_csv,
        split="test",
        transform=transform_eval,
    )

    loaders = {
        "train": DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers),
        "val": DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=workers),
        "test": DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=workers),
    }
    return loaders


def calcular_metricas_binarias(etiquetas: np.ndarray, probabilidades: np.ndarray, umbral: float = 0.5) -> dict:
    predicciones = (probabilidades >= umbral).astype(int)
    return {
        "accuracy": float(accuracy_score(etiquetas, predicciones)),
        "precision": float(precision_score(etiquetas, predicciones, zero_division=0)),
        "recall": float(recall_score(etiquetas, predicciones, zero_division=0)),
        "f1": float(f1_score(etiquetas, predicciones, zero_division=0)),
        "roc_auc": float(roc_auc_score(etiquetas, probabilidades)),
        "confusion_matrix": confusion_matrix(etiquetas, predicciones, labels=[0, 1]).tolist(),
    }


def ejecutar_epoca_entrenamiento(
    modelo: nn.Module,
    loader: DataLoader,
    criterio: nn.Module,
    optimizador: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    modelo.train()
    perdidas = []
    for imagenes, etiquetas in loader:
        imagenes = imagenes.to(device)
        etiquetas = etiquetas.to(device).unsqueeze(1)
        optimizador.zero_grad()
        logits = modelo(imagenes)
        perdida = criterio(logits, etiquetas)
        perdida.backward()
        optimizador.step()
        perdidas.append(float(perdida.item()))
    return float(np.mean(perdidas)) if perdidas else 0.0


@torch.no_grad()
def evaluar_modelo(modelo: nn.Module, loader: DataLoader, criterio: nn.Module, device: torch.device) -> tuple[float, dict]:
    modelo.eval()
    perdidas = []
    probabilidades = []
    etiquetas_reales = []
    for imagenes, etiquetas in loader:
        imagenes = imagenes.to(device)
        etiquetas = etiquetas.to(device).unsqueeze(1)
        logits = modelo(imagenes)
        perdida = criterio(logits, etiquetas)
        probs = torch.sigmoid(logits).squeeze(1)
        perdidas.append(float(perdida.item()))
        probabilidades.extend(probs.detach().cpu().numpy().tolist())
        etiquetas_reales.extend(etiquetas.squeeze(1).detach().cpu().numpy().tolist())

    y_true = np.array(etiquetas_reales, dtype=int)
    y_prob = np.array(probabilidades, dtype=float)
    metricas = calcular_metricas_binarias(y_true, y_prob)
    return float(np.mean(perdidas)) if perdidas else 0.0, metricas


def main() -> None:
    args = parsear_argumentos()
    raiz = obtener_raiz_proyecto()
    ruta_splits = (raiz / args.splits_csv).resolve() if not args.splits_csv.is_absolute() else args.splits_csv
    salida_base = (raiz / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir

    establecer_semilla_global(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sello = datetime.now().strftime("%Y%m%d_%H%M%S")
    salida_run = salida_base / f"{args.modelo}_{sello}"
    salida_run.mkdir(parents=True, exist_ok=True)

    config = ConfigEntrenamiento(
        modelo=args.modelo,
        ruta_splits=str(ruta_splits),
        salida_dir=str(salida_run),
        epocas=args.epocas,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        semilla=args.seed,
        workers=args.workers,
        metrica_checkpoint=args.metrica_checkpoint,
    )
    (salida_run / "config.json").write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8")

    loaders = construir_dataloaders(
        ruta_proyecto=raiz,
        ruta_splits_csv=ruta_splits,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    modelo = crear_modelo_binario(args.modelo, device)
    criterio = nn.BCEWithLogitsLoss()
    optimizador = Adam(modelo.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    historico = []
    mejor_valor = -1.0
    mejor_epoca = -1
    ruta_mejor_checkpoint = salida_run / "best_checkpoint.pt"

    for epoca in range(1, args.epocas + 1):
        perdida_train = ejecutar_epoca_entrenamiento(modelo, loaders["train"], criterio, optimizador, device)
        perdida_val, metricas_val = evaluar_modelo(modelo, loaders["val"], criterio, device)
        registro_epoca = {
            "epoca": epoca,
            "train_loss": perdida_train,
            "val_loss": perdida_val,
            "val_metrics": metricas_val,
        }
        historico.append(registro_epoca)

        valor_checkpoint = metricas_val["recall"] if args.metrica_checkpoint == "val_recall" else metricas_val["f1"]
        if valor_checkpoint > mejor_valor:
            mejor_valor = valor_checkpoint
            mejor_epoca = epoca
            torch.save(
                {
                    "model_state_dict": modelo.state_dict(),
                    "epoch": epoca,
                    "modelo": args.modelo,
                    "metrica_checkpoint": args.metrica_checkpoint,
                    "valor_checkpoint": mejor_valor,
                },
                ruta_mejor_checkpoint,
            )

        print(
            f"[Epoca {epoca}/{args.epocas}] "
            f"train_loss={perdida_train:.4f} val_loss={perdida_val:.4f} "
            f"val_recall={metricas_val['recall']:.4f} val_f1={metricas_val['f1']:.4f}"
        )

    if not ruta_mejor_checkpoint.exists():
        raise RuntimeError("No se pudo guardar un checkpoint valido.")

    checkpoint = torch.load(ruta_mejor_checkpoint, map_location=device)
    modelo.load_state_dict(checkpoint["model_state_dict"])
    perdida_test, metricas_test = evaluar_modelo(modelo, loaders["test"], criterio, device)

    resumen = {
        "mejor_epoca": mejor_epoca,
        "mejor_valor_checkpoint": mejor_valor,
        "metrica_checkpoint": args.metrica_checkpoint,
        "test_loss": perdida_test,
        "metricas_test": metricas_test,
        "ruta_checkpoint": str(ruta_mejor_checkpoint),
    }
    (salida_run / "historial_entrenamiento.json").write_text(
        json.dumps(historico, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (salida_run / "resumen_final.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nEntrenamiento finalizado.")
    print(f"- Run: {salida_run}")
    print(f"- Mejor epoca: {mejor_epoca} ({args.metrica_checkpoint}={mejor_valor:.4f})")
    print(f"- Test recall: {metricas_test['recall']:.4f}")
    print(f"- Test f1: {metricas_test['f1']:.4f}")
    print(f"- Test roc_auc: {metricas_test['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
