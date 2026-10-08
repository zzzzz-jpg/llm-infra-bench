#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# dev_up.sh — bring up the CPU-only MOCK topology so the whole pipeline
# (loadgen -> scrape -> compare -> capacity -> reports) can be developed and
# CI-tested without a GPU.
#
#   ./scripts/dev_up.sh          # starts 4 processes, writes .dev_pids
#   ./scripts/dev_down.sh
#
# Ports
#   8000  "vLLM"  engine, prefix caching ON,  capacity 12
#   8001  "SGLang" engine, prefix caching ON, capacity 14, sglang:* metrics
#   8002  "vLLM"  engine, prefix caching OFF, capacity 12   <- prefix-cache A/B twin
#   8003  "vLLM"  engine, "AWQ int4" profile, capacity 18   <- quantization A/B twin
#   9400  dcgm-exporter (DCGM_FI_* only)
#
# The synthetic latencies are NOT a performance reference. They only exercise
# the code paths: SSE framing, usage chunk, ignore_eos, queue growth, KV-cache
# pressure, prefix-hit accounting and Prometheus scraping.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
mkdir -p .devlogs
PIDFILE="$HERE/.dev_pids"
: >"$PIDFILE"

up() { # up <name> <args...>
  local name="$1"; shift
  python3 bench/mock_server.py "$@" >".devlogs/${name}.log" 2>&1 &
  echo "$! $name" >>"$PIDFILE"
  echo "[dev_up] $name pid=$! $*"
}

up vllm-pc-on   --port 8000 --capacity 12 --ttft-ms 90 --tok-ms 8   --model mock-qwen2.5-7b-instruct
up sglang-pc-on --port 8001 --capacity 14 --ttft-ms 78 --tok-ms 7.4 --model mock-qwen2.5-7b-instruct --sglang-style
up vllm-pc-off  --port 8002 --capacity 12 --ttft-ms 90 --tok-ms 8   --model mock-qwen2.5-7b-instruct --no-prefix-cache
up vllm-awq     --port 8003 --capacity 18 --ttft-ms 62 --tok-ms 5.6 --model mock-qwen2.5-7b-instruct-awq
up dcgm         --port 9400 --capacity 32 --ttft-ms 5  --tok-ms 1   --only-dcgm

sleep 2
for P in 8000 8001 8002 8003 9400; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://127.0.0.1:${P}/metrics" || echo 000)
  echo "[dev_up]   :${P} /metrics -> ${code}"
done
echo "[dev_up] logs in .devlogs/   stop with ./scripts/dev_down.sh"
