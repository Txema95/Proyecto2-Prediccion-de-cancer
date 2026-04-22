"""Etiquetas visuales para columnas y valores codificados."""

from config import ETIQUETAS_POR_COLUMNA, NOMBRES_VISUALES_VARIABLES


def etiqueta_columna(columna: str) -> str:
    return NOMBRES_VISUALES_VARIABLES.get(columna, columna)


def etiqueta_valor_columna(columna: str, valor: int | float) -> str:
    mapa = ETIQUETAS_POR_COLUMNA.get(columna, {})
    valor_entero = int(valor)
    if valor_entero in mapa:
        return mapa[valor_entero]
    return str(valor_entero)
