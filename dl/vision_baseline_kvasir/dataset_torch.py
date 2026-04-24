from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _cargar_filas_csv(ruta: Path) -> list[dict[str, str]]:
    with ruta.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def transformaciones_imagenet_entrenamiento(
    tam_entrada: int,
    *,
    rotacion_grados: float = 15.0,
    brillo: float = 0.2,
    contraste: float = 0.2,
    saturacion: float = 0.15,
    matiz: float = 0.04,
) -> transforms.Compose:
    """Augmentations solo en train: rotación leve (endoscopía), espejo y photometrica más exigente que el baseline mínimo."""
    return transforms.Compose(
        [
            transforms.Resize((tam_entrada, tam_entrada)),
            transforms.RandomRotation(degrees=rotacion_grados, fill=0),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=brillo, contrast=contraste, saturation=saturacion, hue=matiz),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def transformaciones_imagenet_eval(tam_entrada: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((tam_entrada, tam_entrada)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class DatasetKvasirMulticlase(Dataset[dict[str, object]]):
    """Lee `splits_kvasir_multiclase.csv` (o similar) y carga jpg/png desde `filepath` relativo al repo."""

    def __init__(
        self,
        ruta_splits: Path,
        particion: str,
        raiz: Path,
        transform: transforms.Compose | None = None,
    ) -> None:
        ruta_splits = Path(ruta_splits)
        if not ruta_splits.is_file():
            raise FileNotFoundError(f"No existe CSV de particiones: {ruta_splits}")
        self.raiz = raiz.resolve()
        filas = _cargar_filas_csv(ruta_splits)
        self.filas = [r for r in filas if r.get("split") == particion]
        if not self.filas:
            raise ValueError(
                f"No hay filas con split={particion!r} en {ruta_splits}. Revisa el fichero."
            )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filas)

    def __getitem__(self, ind: int) -> dict[str, object]:
        f = self.filas[ind]
        ruta = (self.raiz / f["filepath"]).resolve()
        if not ruta.is_file():
            raise FileNotFoundError(f"Falta imagen: {ruta}")
        y = int(f["label"])
        with Image.open(ruta) as im:
            im = im.convert("RGB")
        if self.transform is not None:
            im = self.transform(im)
        return {"x": im, "y": torch.tensor(y, dtype=torch.long), "image_id": f.get("image_id", "")}
