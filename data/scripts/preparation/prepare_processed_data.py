from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


RAIZ_PROYECTO = Path(__file__).resolve().parents[3]
DIR_CRUDO_POLIPOS = RAIZ_PROYECTO / "data" / "raw" / "polipos"
DIR_CRUDO_KVASIR = RAIZ_PROYECTO / "data" / "raw" / "kvasir-dataset-v2"
DIR_PROCESADO = RAIZ_PROYECTO / "data" / "processed"
DIR_PROCESADO_POLIPO = DIR_PROCESADO / "polipo"
DIR_PROCESADO_SANO = DIR_PROCESADO / "sano"

CATEGORIAS_NORMALES_KVASIR = ("normal-cecum", "normal-pylorus", "normal-z-line")
EXTENSIONES_IMAGEN = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parsear_argumentos() -> argparse.Namespace:
    analizador = argparse.ArgumentParser(
        description=(
            "Prepara data/processed para clasificacion binaria de polipos "
            "(CVC positivo + Kvasir normal negativo)."
        )
    )
    analizador.add_argument(
        "--kvasir-target",
        type=int,
        default=612,
        help="Total de imagenes sanas a copiar desde Kvasir.",
    )
    analizador.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para muestreo reproducible de Kvasir.",
    )
    analizador.add_argument(
        "--overwrite",
        action="store_true",
        help="Si existe data/processed/polipo o data/processed/sano, la reemplaza.",
    )
    return analizador.parse_args()


def asegurar_directorios(sobrescribir: bool) -> None:
    DIR_PROCESADO.mkdir(parents=True, exist_ok=True)
    for directorio_destino in (DIR_PROCESADO_POLIPO, DIR_PROCESADO_SANO):
        if directorio_destino.exists() and sobrescribir:
            shutil.rmtree(directorio_destino)
        directorio_destino.mkdir(parents=True, exist_ok=True)


def leer_rutas_polipos() -> list[Path]:
    ruta_metadata = DIR_CRUDO_POLIPOS / "metadata.csv"
    if not ruta_metadata.exists():
        raise FileNotFoundError(f"No se encontro metadata de CVC: {ruta_metadata}")

    rutas: list[Path] = []
    with ruta_metadata.open("r", newline="", encoding="utf-8") as archivo:
        lector = csv.DictReader(archivo)
        for fila in lector:
            ruta_relativa = fila["png_image_path"]
            ruta_imagen = DIR_CRUDO_POLIPOS / ruta_relativa
            if not ruta_imagen.exists():
                raise FileNotFoundError(f"No se encontro imagen CVC referenciada: {ruta_imagen}")
            rutas.append(ruta_imagen)
    return rutas


def copiar_imagenes_polipos() -> int:
    total_copiadas = 0
    for origen in leer_rutas_polipos():
        destino = DIR_PROCESADO_POLIPO / origen.name
        shutil.copy2(origen, destino)
        total_copiadas += 1
    return total_copiadas


def listar_imagenes_kvasir(categoria: str) -> list[Path]:
    directorio_categoria = DIR_CRUDO_KVASIR / categoria
    if not directorio_categoria.exists():
        raise FileNotFoundError(f"No se encontro categoria de Kvasir: {directorio_categoria}")

    archivos = [
        ruta
        for ruta in directorio_categoria.iterdir()
        if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_IMAGEN
    ]
    return sorted(archivos)


def repartir_objetivo_por_categoria(total_objetivo: int, categorias: tuple[str, ...]) -> dict[str, int]:
    base = total_objetivo // len(categorias)
    residuo = total_objetivo % len(categorias)
    asignacion: dict[str, int] = {}
    for i, categoria in enumerate(categorias):
        asignacion[categoria] = base + (1 if i < residuo else 0)
    return asignacion


def muestrear_imagenes_kvasir(total_objetivo: int, semilla: int) -> list[tuple[str, Path]]:
    generador_aleatorio = random.Random(semilla)
    asignacion = repartir_objetivo_por_categoria(total_objetivo, CATEGORIAS_NORMALES_KVASIR)
    seleccionadas: list[tuple[str, Path]] = []

    for categoria in CATEGORIAS_NORMALES_KVASIR:
        disponibles = listar_imagenes_kvasir(categoria)
        necesarias = asignacion[categoria]
        if necesarias > len(disponibles):
            raise ValueError(
                f"No hay suficientes imagenes en {categoria}. "
                f"Solicitadas={necesarias}, disponibles={len(disponibles)}."
            )
        elegidas = generador_aleatorio.sample(disponibles, necesarias)
        seleccionadas.extend((categoria, ruta) for ruta in elegidas)

    return seleccionadas


def copiar_imagenes_kvasir(total_objetivo: int, semilla: int) -> int:
    total_copiadas = 0
    for categoria, origen in muestrear_imagenes_kvasir(total_objetivo, semilla):
        # Prefijo de categoria para evitar posibles colisiones entre carpetas.
        nombre_destino = f"{categoria}__{origen.name}"
        destino = DIR_PROCESADO_SANO / nombre_destino
        shutil.copy2(origen, destino)
        total_copiadas += 1
    return total_copiadas


def main() -> None:
    argumentos = parsear_argumentos()
    asegurar_directorios(sobrescribir=argumentos.overwrite)

    total_polipos = copiar_imagenes_polipos()
    total_sanos = copiar_imagenes_kvasir(
        total_objetivo=argumentos.kvasir_target,
        semilla=argumentos.seed,
    )

    print("Preparacion completada.")
    print(f"- Carpeta positiva: {DIR_PROCESADO_POLIPO} ({total_polipos} imagenes)")
    print(f"- Carpeta negativa: {DIR_PROCESADO_SANO} ({total_sanos} imagenes)")
    print(f"- Semilla: {argumentos.seed}")


if __name__ == "__main__":
    main()
