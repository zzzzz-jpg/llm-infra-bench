#!/usr/bin/env bash
# wait_ready.sh — block until an OpenAI-compatible server can actually serve.
#
# /health returning 200 is NOT enough for vLLM: the process answers health
# checks before weights are on the GPU and CUDA graphs are captured. This probe
# sends one real (tiny) completion and only returns success when that works,
# which is what makes back-to-back A/B runs comparable — otherwise the first
# variant silently eats the model-load cost in its TTFT numbers.
#
#   ./scripts/wait_ready.sh http://127.0.0.1:8000 [timeout_sec] [model]
set -uo pipefail

URL="${1:-http://127.0.0.1:8000}"
TIMEOUT="${2:-900}"
MODEL="${3:-}"
INTERVAL="${INTERVAL:-5}"
start=$(date +%s)

echo "[wait_ready] waiting for ${URL} (timeout ${TIMEOUT}s)"

while true; do
  now=$(date +%s); elapsed=$((now - start))
  if (( elapsed > TIMEOUT )); then
    echo "[wait_ready] TIMEOUT after ${elapsed}s" >&2
    exit 1
  fi

  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "${URL}/health" 2>/dev/null || echo 000)
  if [[ "$code" != "200" ]]; then
    printf '[wait_ready] %4ss  /health=%s\n' "$elapsed" "$code"
    sleep "$INTERVAL"; continue
  fi

  if [[ -z "$MODEL" ]]; then
    MODEL=$(curl -s -m 5 "${URL}/v1/models" \
      | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo "")
  fi
  if [[ -z "$MODEL" ]]; then
    printf '[wait_ready] %4ss  health ok, model list not ready\n' "$elapsed"
    sleep "$INTERVAL"; continue
  fi

  body=$(curl -s -m 60 "${URL}/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"hi\",\"max_tokens\":4,\"temperature\":0,\"ignore_eos\":true}")
  if python3 -c 'import sys,json;d=json.load(sys.stdin);sys.exit(0 if d.get("choices") else 1)' <<<"$body" 2>/dev/null; then
    echo "[wait_ready] READY in ${elapsed}s — model=${MODEL}"
    echo "[wait_ready] probe response: ${body:0:200}"
    exit 0
  fi
  printf '[wait_ready] %4ss  health ok, inference not ready: %s\n' "$elapsed" "${body:0:120}"
  sleep "$INTERVAL"
done
