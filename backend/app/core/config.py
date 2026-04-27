import os
from dataclasses import dataclass, field


def _split_origins(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _parsear_umbral_decision() -> float:
    valor_bruto = os.environ.get("SIMULATOR_DECISION_THRESHOLD", "0.5").strip()
    try:
        valor = float(valor_bruto)
    except ValueError as exc:
        raise ValueError(
            "SIMULATOR_DECISION_THRESHOLD debe ser un numero entre 0.0 y 1.0"
        ) from exc
    if not 0.0 <= valor <= 1.0:
        raise ValueError("SIMULATOR_DECISION_THRESHOLD debe estar entre 0.0 y 1.0")
    return valor


@dataclass
class Settings:
    app_name: str = "Detección cáncer colon"
    cors_origins: list[str] = field(
        default_factory=lambda: _split_origins(
            os.environ.get(
                "CORS_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173,"
                "http://localhost:8501,http://127.0.0.1:8501",
            )
        )
    )
    umbral_decision: float = field(default_factory=_parsear_umbral_decision)


settings = Settings()
