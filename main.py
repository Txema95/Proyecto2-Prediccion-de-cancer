"""
Arranque unificado: API FastAPI (puerto 8000) y Streamlit (puerto 8501).

Uso (desde la raíz del repositorio):
    uv run python main.py
    uv run python main.py --api-only
    uv run python main.py --ui-only
    uv run python main.py --no-reload

Solo API (como hasta ahora):
    uv run uvicorn app.main:app --reload --app-dir backend
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
BACKEND = REPO / "backend"
FRONT = REPO / "frontend" / "app.py"
def _popen(cmd: list[str], *, env: dict | None = None) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        cmd,
        cwd=REPO,
        env=env,
        start_new_session=True,
    )


def _terminar_grupo(proceso: subprocess.Popen[bytes] | None) -> None:
    if proceso is None or proceso.poll() is not None:
        return
    try:
        os.killpg(proceso.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proceso.terminate()
    try:
        proceso.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proceso.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proceso.kill()


def main() -> int:
    ap = argparse.ArgumentParser(description="API + Streamlit en el mismo arranque.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--api-only", action="store_true", help="Solo Uvicorn (puerto 8000).")
    g.add_argument("--ui-only", action="store_true", help="Solo Streamlit (puerto 8501).")
    ap.add_argument(
        "--no-reload",
        action="store_true",
        help="Uvicorn sin --reload (mas estable para demos).",
    )
    ap.add_argument("--port-api", type=int, default=8000)
    ap.add_argument("--port-ui", type=int, default=8501)
    args = ap.parse_args()

    if not FRONT.is_file():
        print(f"No se encontro {FRONT}", file=sys.stderr)
        return 1

    env_base = os.environ.copy()
    env_base.setdefault("SIMULATOR_API_BASE_URL", f"http://127.0.0.1:{args.port_api}".rstrip("/"))

    procs: list[subprocess.Popen[bytes]] = []

    def on_sigint(_sig: int, _frame: object) -> None:
        for p in procs:
            _terminar_grupo(p)
        sys.exit(130)

    signal.signal(signal.SIGINT, on_sigint)
    signal.signal(signal.SIGTERM, on_sigint)

    if not args.ui_only:
        cmd_api = [sys.executable, "-m", "uvicorn", "app.main:app", "--app-dir", str(BACKEND), "--host", "127.0.0.1", "--port", str(args.port_api)]
        if not args.no_reload:
            cmd_api.append("--reload")
        p_api = _popen(cmd_api, env=env_base)
        procs.append(p_api)
        print(f"API:   http://127.0.0.1:{args.port_api}  (docs: /docs)")
    if not args.api_only:
        env_ui = env_base.copy()
        # Streamlit port
        cmd_ui = [sys.executable, "-m", "streamlit", "run", str(FRONT), "--server.port", str(args.port_ui), "--server.address", "127.0.0.1"]
        p_ui = _popen(cmd_ui, env=env_ui)
        procs.append(p_ui)
        print(f"Streamlit: http://127.0.0.1:{args.port_ui}")
        if not args.ui_only:
            print("Pulsa Ctrl+C para detener los procesos.")

    if args.api_only and not args.ui_only:
        p = procs[0]
        return p.wait() or 0
    if args.ui_only and not args.api_only:
        p = procs[0]
        return p.wait() or 0

    # Both: wait on first process; if it exits, stop the other
    if len(procs) == 2:
        while True:
            r0 = procs[0].poll()
            r1 = procs[1].poll()
            if r0 is not None:
                _terminar_grupo(procs[1])
                return r0
            if r1 is not None:
                _terminar_grupo(procs[0])
                return r1
            time.sleep(0.3)
    if procs:
        return procs[0].wait() or 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
