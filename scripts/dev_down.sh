#!/usr/bin/env bash
# dev_down.sh — stop everything started by dev_up.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDFILE="$HERE/.dev_pids"
[[ -f "$PIDFILE" ]] || { echo "[dev_down] no $PIDFILE"; exit 0; }
while read -r pid name; do
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null && echo "[dev_down] stopped $name ($pid)"
  else
    echo "[dev_down] $name ($pid) already gone"
  fi
done <"$PIDFILE"
sleep 1
pkill -f "bench/mock_server.py" 2>/dev/null || true
rm -f "$PIDFILE"
echo "[dev_down] done"
