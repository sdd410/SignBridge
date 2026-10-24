#!/usr/bin/env bash
set -e

# SignBridge One-Command Startup Script
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================================="
echo " Starting SignBridge Application (Backend + Frontend)"
echo "=========================================================="

# Check Python environment
if [ -f "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
    UVICORN_BIN="$PROJECT_ROOT/.venv/bin/uvicorn"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
    UVICORN_BIN="uvicorn"
else
    echo "Error: Python 3 was not found."
    exit 1
fi

# Cleanup handler on exit or Ctrl+C
cleanup() {
    echo ""
    echo "Stopping SignBridge servers..."
    if [ -n "$BACKEND_PID" ]; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    echo "SignBridge stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# 1. Start Backend in background
echo "-> Starting Backend API server on http://localhost:8000 ..."
"$UVICORN_BIN" backend.app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to initialize
sleep 1.5

# 2. Start Frontend in background
echo "-> Starting Frontend development server on http://localhost:5173 ..."
(cd "$PROJECT_ROOT/frontend" && npm run dev) &
FRONTEND_PID=$!

echo ""
echo "=========================================================="
echo " SignBridge is running!"
echo "   Dashboard: http://localhost:5173"
echo "   API Docs:  http://localhost:8000/docs"
echo " Press Ctrl+C to stop both servers."
echo "=========================================================="
echo ""

# Wait on background processes
wait
