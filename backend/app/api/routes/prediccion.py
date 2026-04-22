"""Endpoint de inferencia para el simulador."""

from fastapi import APIRouter, HTTPException

from app.schemas.prediccion import PrediccionEntrada, PrediccionSalida, ResultadoImagen
from app.services.prediccion_tabular import ejecutar_inferencia

router = APIRouter()


@router.post("/ejecutar-prediccion", response_model=PrediccionSalida)
def ejecutar_prediccion(cuerpo: PrediccionEntrada) -> PrediccionSalida:
    try:
        datos = {k: float(v) for k, v in cuerpo.datos_clinicos.items()}
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="datos_clinicos deben ser valores numericos") from exc

    try:
        prob_tabular, prob_combinada, resultado_imagen = ejecutar_inferencia(
            datos, cuerpo.num_imagenes_adjuntas
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en inferencia: {exc!s}") from exc

    return PrediccionSalida(
        probabilidad_tabular=prob_tabular,
        probabilidad_combinada=prob_combinada,
        resultado_imagen=ResultadoImagen(
            estado=str(resultado_imagen["estado"]),
            mensaje=str(resultado_imagen["mensaje"]),
            probabilidad=resultado_imagen.get("probabilidad"),
        ),
    )
