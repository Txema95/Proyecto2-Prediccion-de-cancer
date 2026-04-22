"""Esquemas de entrada y salida para /ejecutar-prediccion."""

from typing import Any

from pydantic import BaseModel, Field


class ResultadoImagen(BaseModel):
    estado: str
    mensaje: str
    probabilidad: float | None = None


class PrediccionEntrada(BaseModel):
    datos_clinicos: dict[str, Any] = Field(
        ...,
        description="Valores numericos de las variables predictoras (mismas claves que el CSV limpio).",
    )
    num_imagenes_adjuntas: int = Field(0, ge=0, description="Numero de imagenes adjuntas al caso (sin bytes en esta version).")


class PrediccionSalida(BaseModel):
    probabilidad_tabular: float
    probabilidad_combinada: float
    resultado_imagen: ResultadoImagen
