#!/usr/bin/env bash
# Start the real_simple_stats web app (backend + frontend).
# Run from the app/ directory: bash start.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting backend on :8020..."
cd "$SCRIPT_DIR"
PYTHONPATH=".:..":$PYTHONPATH uvicorn backend.main:app --port 8020 --reload &
BACKEND_PID=$!

echo "Starting frontend..."
cd "$SCRIPT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "Backend:  http://localhost:8020"
echo "Frontend: http://localhost:5173  (or next available port)"
echo ""
echo "Press Ctrl-C to stop both."

cleanup() {
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
  exit 0
}
trap cleanup SIGINT SIGTERM
wait
