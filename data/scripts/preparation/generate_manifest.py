from __future__ import annotations

import argparse
import csv
from pathlib import Path


RAIZ_PROYECTO = Path(__file__).resolve().parents[3]
DIR_CRUDO_POLIPOS = RAIZ_PROYECTO / "data" / "raw" / "polipos"
DIR_PROCESADO = RAIZ_PROYECTO / "data" / "processed"
DIR_PROCESADO_POLIPO = DIR_PROCESADO / "polipo"
DIR_PROCESADO_SANO = DIR_PROCESADO / "sano"
CSV_SALIDA_POR_DEFECTO = DIR_PROCESADO / "manifest.csv"


def parsear_argumentos() -> argparse.Namespace:
    analizador = argparse.ArgumentParser(
        description=(
            "Genera manifest.csv con columnas: filepath,label,source,group_id,image_id."
        )
    )
    analizador.add_argument(
        "--output",
        type=Path,
        default=CSV_SALIDA_POR_DEFECTO,
        help="Ruta de salida del manifest CSV.",
    )
    return analizador.parse_args()


def construir_mapa_secuencia_cvc() -> dict[str, str]:
    ruta_metadata = DIR_CRUDO_POLIPOS / "metadata.csv"
    if not ruta_metadata.exists():
        raise FileNotFoundError(f"No se encontro metadata de CVC: {ruta_metadata}")

    mapa_secuencias: dict[str, str] = {}
    with ruta_metadata.open("r", newline="", encoding="utf-8") as archivo:
        lector = csv.DictReader(archivo)
        for fila in lector:
            ruta_png_relativa = fila["png_image_path"]
            nombre_imagen = Path(ruta_png_relativa).name
            mapa_secuencias[nombre_imagen] = fila["sequence_id"]
    return mapa_secuencias


def iterar_imagenes_procesadas(carpeta: Path) -> list[Path]:
    if not carpeta.exists():
        raise FileNotFoundError(f"No existe carpeta esperada: {carpeta}")
    return sorted(ruta for ruta in carpeta.iterdir() if ruta.is_file())


def construir_filas() -> list[dict[str, str]]:
    filas: list[dict[str, str]] = []
    secuencia_cvc_por_nombre = construir_mapa_secuencia_cvc()

    for ruta in iterar_imagenes_procesadas(DIR_PROCESADO_POLIPO):
        id_secuencia = secuencia_cvc_por_nombre.get(ruta.name)
        if id_secuencia is None:
            raise ValueError(
                f"No se encontro sequence_id para imagen polipo en metadata.csv: {ruta.name}"
            )
        filas.append(
            {
                "filepath": ruta.relative_to(RAIZ_PROYECTO).as_posix(),
                "label": "1",
                "source": "cvc_clinicdb",
                "group_id": f"cvc_seq_{id_secuencia}",
                "image_id": f"cvc_{ruta.stem}",
            }
        )

    for ruta in iterar_imagenes_procesadas(DIR_PROCESADO_SANO):
        # Formato esperado: <categoria>__<archivo_original>.
        if "__" in ruta.name:
            categoria, nombre_original = ruta.name.split("__", 1)
        else:
            categoria, nombre_original = "kvasir-normal", ruta.name

        filas.append(
            {
                "filepath": ruta.relative_to(RAIZ_PROYECTO).as_posix(),
                "label": "0",
                "source": f"kvasir_{categoria}",
                "group_id": f"kvasir_{categoria}_{Path(nombre_original).stem}",
                "image_id": f"kvasir_{Path(ruta.name).stem}",
            }
        )

    return filas


def guardar_manifest(filas: list[dict[str, str]], ruta_salida: Path) -> None:
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    columnas = ["filepath", "label", "source", "group_id", "image_id"]
    with ruta_salida.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)


def main() -> None:
    argumentos = parsear_argumentos()
    filas = construir_filas()
    guardar_manifest(filas, argumentos.output)

    total_polipo = sum(1 for fila in filas if fila["label"] == "1")
    total_sano = sum(1 for fila in filas if fila["label"] == "0")
    print(f"Manifest generado: {argumentos.output}")
    print(f"- Registros totales: {len(filas)}")
    print(f"- Polipo (1): {total_polipo}")
    print(f"- Sano (0): {total_sano}")


if __name__ == "__main__":
    main()
