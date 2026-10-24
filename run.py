"""SignBridge One-Command Application Launcher.

Launches both the FastAPI backend and the Vite frontend concurrently in a single process,
handles logging output, and ensures clean termination of all child processes upon Ctrl+C.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def find_python_executable() -> str:
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def main():
    print("=" * 60)
    print("  Starting SignBridge Application (Backend + Frontend)")
    print("=" * 60)

    python_bin = find_python_executable()
    frontend_dir = PROJECT_ROOT / "frontend"

    # Start FastAPI backend server
    print("\n-> Launching Backend API (http://localhost:8000)...")
    backend_proc = subprocess.Popen(
        [
            python_bin, "-m", "uvicorn",
            "backend.app.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
        ],
        cwd=str(PROJECT_ROOT)
    )

    # Allow backend to bind socket
    time.sleep(1.5)

    # Start Vite frontend
    print("-> Launching Frontend Dashboard (http://localhost:5173)...")
    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(frontend_dir)
    )

    print("\n" + "=" * 60)
    print("  SignBridge is running!")
    print("    Frontend Dashboard: http://localhost:5173")
    print("    Backend API Docs:   http://localhost:8000/docs")
    print("  Press Ctrl+C to terminate both servers.")
    print("=" * 60 + "\n")

    def shutdown(signum, frame):
        print("\nStopping SignBridge servers...")
        try:
            frontend_proc.terminate()
            backend_proc.terminate()
            frontend_proc.wait(timeout=3)
            backend_proc.wait(timeout=3)
        except Exception:
            frontend_proc.kill()
            backend_proc.kill()
        print("SignBridge stopped cleanly.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Keep main process alive until interrupted or child exits
    try:
        while True:
            if backend_proc.poll() is not None:
                print(f"Backend exited with code {backend_proc.returncode}")
                shutdown(None, None)
            if frontend_proc.poll() is not None:
                print(f"Frontend exited with code {frontend_proc.returncode}")
                shutdown(None, None)
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()
