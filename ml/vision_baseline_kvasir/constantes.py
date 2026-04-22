from __future__ import annotations

# Nombres de carpetas (Kvasir v2) e índice de clase para el modelo.
CLASES_ORDEN: tuple[str, ...] = (
    "normal-cecum",
    "polyps",
    "dyed-lifted-polyps",
    "ulcerative-colitis",
)

_CLASE_A_INDICE: dict[str, int] = {c: i for i, c in enumerate(CLASES_ORDEN)}


def clase_a_indice(nombre: str) -> int:
    if nombre not in _CLASE_A_INDICE:
        raise KeyError(f"Clase desconocida: {nombre!r}; esperada una de {CLASES_ORDEN}.")
    return _CLASE_A_INDICE[nombre]


def indice_a_clase(ind: int) -> str:
    if not 0 <= ind < len(CLASES_ORDEN):
        raise IndexError(f"Indice de clase invalido: {ind}")
    return CLASES_ORDEN[ind]
