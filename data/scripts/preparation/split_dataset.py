from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


RAIZ_PROYECTO = Path(__file__).resolve().parents[3]
MANIFEST_POR_DEFECTO = RAIZ_PROYECTO / "data" / "processed" / "manifest.csv"
SPLITS_POR_DEFECTO = RAIZ_PROYECTO / "data" / "processed" / "splits.csv"


def parsear_argumentos() -> argparse.Namespace:
    analizador = argparse.ArgumentParser(
        description=(
            "Genera splits train/val/test desde manifest.csv respetando group_id "
            "y validando que no haya fuga entre conjuntos."
        )
    )
    analizador.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_POR_DEFECTO,
        help="Ruta al manifest.csv de entrada.",
    )
    analizador.add_argument(
        "--output",
        type=Path,
        default=SPLITS_POR_DEFECTO,
        help="Ruta del splits.csv de salida.",
    )
    analizador.add_argument("--train-ratio", type=float, default=0.70)
    analizador.add_argument("--val-ratio", type=float, default=0.15)
    analizador.add_argument("--test-ratio", type=float, default=0.15)
    analizador.add_argument("--seed", type=int, default=42)
    return analizador.parse_args()


def leer_manifest(ruta: Path) -> list[dict[str, str]]:
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontro manifest: {ruta}")
    with ruta.open("r", newline="", encoding="utf-8") as archivo:
        filas = list(csv.DictReader(archivo))
    columnas_requeridas = {"filepath", "label", "source", "group_id", "image_id"}
    if not filas:
        raise ValueError("Manifest vacio.")
    columnas_faltantes = columnas_requeridas - set(filas[0].keys())
    if columnas_faltantes:
        raise ValueError(f"Faltan columnas en manifest: {sorted(columnas_faltantes)}")
    return filas


def validar_proporciones(train: float, val: float, test: float) -> None:
    total = train + val + test
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Las proporciones deben sumar 1.0; recibido={total:.8f}")


def asignar_grupos_a_splits(
    filas: list[dict[str, str]], train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> dict[str, str]:
    grupos_a_filas: dict[str, list[dict[str, str]]] = defaultdict(list)
    for fila in filas:
        grupos_a_filas[fila["group_id"]].append(fila)

    nombres_split = ["train", "val", "test"]
    proporciones_objetivo = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    total_filas = len(filas)
    conteos_objetivo = {
        split: total_filas * ratio for split, ratio in proporciones_objetivo.items()
    }
    conteos_actuales = {split: 0 for split in nombres_split}
    asignaciones: dict[str, str] = {}

    grupos = list(grupos_a_filas.items())
    generador_aleatorio = random.Random(seed)
    generador_aleatorio.shuffle(grupos)
    grupos.sort(key=lambda par: len(par[1]), reverse=True)

    for group_id, filas_grupo in grupos:
        tamano_grupo = len(filas_grupo)
        deficit = {
            split: conteos_objetivo[split] - conteos_actuales[split] for split in nombres_split
        }
        splits_candidatos = [split for split in nombres_split if deficit[split] >= tamano_grupo]
        if splits_candidatos:
            # Priorizamos el split con mayor deficit real que aun puede absorber el grupo.
            split_elegido = max(splits_candidatos, key=lambda split: deficit[split])
        else:
            # Si todos los splits ya estan "llenos", elegimos el menos sobrecargado.
            split_elegido = max(nombres_split, key=lambda split: deficit[split])
        asignaciones[group_id] = split_elegido
        conteos_actuales[split_elegido] += tamano_grupo

    return asignaciones


def dividir_por_etiqueta(
    filas: list[dict[str, str]], train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> dict[str, str]:
    por_etiqueta: dict[str, list[dict[str, str]]] = defaultdict(list)
    for fila in filas:
        por_etiqueta[fila["label"]].append(fila)

    asignaciones_combinadas: dict[str, str] = {}
    for i, (etiqueta, filas_etiqueta) in enumerate(
        sorted(por_etiqueta.items(), key=lambda item: item[0])
    ):
        asignaciones_etiqueta = asignar_grupos_a_splits(
            filas=filas_etiqueta,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed + i,
        )
        superposicion = set(asignaciones_combinadas).intersection(asignaciones_etiqueta)
        if superposicion:
            raise ValueError(
                f"Hay group_id repetidos entre etiquetas distintas. Revisar manifest. "
                f"Ejemplo: {next(iter(superposicion))} (etiqueta: {etiqueta})"
            )
        asignaciones_combinadas.update(asignaciones_etiqueta)
    return asignaciones_combinadas


def validar_sin_fugas(filas: list[dict[str, str]], asignaciones: dict[str, str]) -> None:
    splits_a_grupos: dict[str, set[str]] = defaultdict(set)
    for fila in filas:
        split = asignaciones[fila["group_id"]]
        splits_a_grupos[split].add(fila["group_id"])

    inter_train_val = splits_a_grupos["train"] & splits_a_grupos["val"]
    inter_train_test = splits_a_grupos["train"] & splits_a_grupos["test"]
    inter_val_test = splits_a_grupos["val"] & splits_a_grupos["test"]
    if inter_train_val or inter_train_test or inter_val_test:
        raise ValueError("Se detecto fuga de grupos entre splits.")


def guardar_splits(
    filas: list[dict[str, str]], asignaciones: dict[str, str], ruta_salida: Path
) -> None:
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    columnas = ["filepath", "label", "source", "group_id", "image_id", "split"]
    with ruta_salida.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        for fila in filas:
            fila_salida = dict(fila)
            fila_salida["split"] = asignaciones[fila["group_id"]]
            escritor.writerow(fila_salida)


def imprimir_resumen(
    filas: list[dict[str, str]], asignaciones: dict[str, str], ruta_salida: Path, semilla: int
) -> None:
    conteos_split: Counter[str] = Counter()
    conteos_split_etiqueta: dict[str, Counter[str]] = defaultdict(Counter)
    for fila in filas:
        split = asignaciones[fila["group_id"]]
        conteos_split[split] += 1
        conteos_split_etiqueta[split][fila["label"]] += 1

    print(f"Splits guardados en: {ruta_salida}")
    print(f"Semilla: {semilla}")
    for split in ("train", "val", "test"):
        total = conteos_split[split]
        positivos = conteos_split_etiqueta[split]["1"]
        negativos = conteos_split_etiqueta[split]["0"]
        print(f"- {split}: total={total}, polipo(1)={positivos}, sano(0)={negativos}")


def main() -> None:
    argumentos = parsear_argumentos()
    validar_proporciones(argumentos.train_ratio, argumentos.val_ratio, argumentos.test_ratio)

    filas = leer_manifest(argumentos.manifest)
    asignaciones = dividir_por_etiqueta(
        filas=filas,
        train_ratio=argumentos.train_ratio,
        val_ratio=argumentos.val_ratio,
        test_ratio=argumentos.test_ratio,
        seed=argumentos.seed,
    )
    validar_sin_fugas(filas, asignaciones)
    guardar_splits(filas, asignaciones, argumentos.output)
    imprimir_resumen(filas, asignaciones, argumentos.output, argumentos.seed)


if __name__ == "__main__":
    main()
