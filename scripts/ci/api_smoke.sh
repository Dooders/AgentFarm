#!/usr/bin/env bash
# Start the FastAPI server under uvicorn as a real process, wait for it to
# accept connections, hit a few read-only endpoints, and shut it down.
# Catches import-time and lifespan failures that fastapi.testclient masks.
set -euo pipefail

PORT="${API_SMOKE_PORT:-5055}"
BASE="http://127.0.0.1:${PORT}"
LOG="${RUNNER_TEMP:-/tmp}/api-smoke-uvicorn.log"

uvicorn farm.api.server:app --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
PID=$!
trap 'kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  if curl -sf "$BASE/openapi.json" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "::error::uvicorn exited before accepting connections"
    cat "$LOG"
    exit 1
  fi
  sleep 1
done

check() {
  local path="$1"
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' "$BASE$path")
  if [ "$code" != "200" ]; then
    echo "::error::GET $path returned HTTP $code"
    cat "$LOG"
    exit 1
  fi
  echo "GET $path -> $code"
}

check /openapi.json
check /docs
check /api/simulations
check /api/analysis/modules

echo "API smoke test passed"
