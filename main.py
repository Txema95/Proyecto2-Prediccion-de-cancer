"""
Raíz del monorepo: API en `backend/`, front en `frontend/`, datos en `data/` y `assets/`.
Arranque del API: uv run uvicorn app.main:app --reload --app-dir backend
Documentación interactiva: http://127.0.0.1:8000/docs
"""


def main() -> None:
    print("Proyecto deteccion-cancer. Arranca el API con:")
    print("  uv run uvicorn app.main:app --reload --app-dir backend")


if __name__ == "__main__":
    main()
