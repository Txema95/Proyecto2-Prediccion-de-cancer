import os
from dataclasses import dataclass, field


def _split_origins(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


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


settings = Settings()
