from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class RegistroSplit:
    filepath: str
    label: int
    source: str
    group_id: str
    image_id: str
    split: str


def cargar_registros_desde_csv(ruta_splits_csv: Path, split_objetivo: str) -> list[RegistroSplit]:
    if split_objetivo not in {"train", "val", "test"}:
        raise ValueError(f"Split no valido: {split_objetivo}. Use train, val o test.")
    if not ruta_splits_csv.exists():
        raise FileNotFoundError(f"No existe splits.csv en: {ruta_splits_csv}")

    registros: list[RegistroSplit] = []
    with ruta_splits_csv.open("r", newline="", encoding="utf-8") as archivo:
        lector = csv.DictReader(archivo)
        columnas = {"filepath", "label", "source", "group_id", "image_id", "split"}
        if lector.fieldnames is None or not columnas.issubset(set(lector.fieldnames)):
            raise ValueError(
                "splits.csv no contiene las columnas esperadas: "
                "filepath,label,source,group_id,image_id,split"
            )
        for fila in lector:
            if fila["split"] != split_objetivo:
                continue
            registros.append(
                RegistroSplit(
                    filepath=fila["filepath"],
                    label=int(fila["label"]),
                    source=fila["source"],
                    group_id=fila["group_id"],
                    image_id=fila["image_id"],
                    split=fila["split"],
                )
            )
    if not registros:
        raise ValueError(f"No se encontraron registros para split='{split_objetivo}'.")
    return registros


class DatasetColonoscopiaBinario(Dataset):
    """Dataset para clasificacion binaria de polipo/no-polipo."""

    def __init__(self, ruta_proyecto: Path, ruta_splits_csv: Path, split: str, transform=None) -> None:
        self.ruta_proyecto = ruta_proyecto
        self.transform = transform
        self.registros = cargar_registros_desde_csv(ruta_splits_csv=ruta_splits_csv, split_objetivo=split)

    def __len__(self) -> int:
        return len(self.registros)

    def __getitem__(self, indice: int):
        registro = self.registros[indice]
        ruta_imagen = self.ruta_proyecto / registro.filepath
        if not ruta_imagen.exists():
            raise FileNotFoundError(f"No se encontro imagen referenciada en splits.csv: {ruta_imagen}")

        imagen = Image.open(ruta_imagen).convert("RGB")
        if self.transform is not None:
            imagen = self.transform(imagen)

        etiqueta = float(registro.label)
        return imagen, etiqueta
