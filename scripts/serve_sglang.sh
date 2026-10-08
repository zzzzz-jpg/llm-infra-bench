#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# serve_sglang.sh — the same fixed-workload contract as serve_vllm.sh, on
# SGLang, so the vLLM-vs-SGLang comparison differs in *engine only*.
#
#   MODEL=Qwen/Qwen2.5-7B-Instruct ./scripts/serve_sglang.sh
#   RADIX_CACHE=0 ./scripts/serve_sglang.sh     # disables prefix caching
#
# Flag mapping used to keep the two engines comparable:
#   vLLM                              SGLang
#   --gpu-memory-utilization          --mem-fraction-static
#   --max-model-len                   --context-length
#   --max-num-seqs                    --max-running-requests
#   --enable-prefix-caching           (default on) / --disable-radix-cache
#   --enable-chunked-prefill          --chunked-prefill-size
#   --enforce-eager                   --disable-cuda-graph
#   --quantization                    --quantization
#   --kv-cache-dtype fp8              --kv-cache-dtype fp8_e5m2
#   /metrics (always on)              --enable-metrics
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SERVED_NAME="${SERVED_NAME:-$MODEL}"
HOST="${HOST:-0.0.0.0}"; PORT="${PORT:-30000}"
TP="${TP:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"
CONTEXT_LEN="${CONTEXT_LEN:-8192}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-256}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"   # -1 disables
QUANT="${QUANT:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
RADIX_CACHE="${RADIX_CACHE:-1}"        # 1 = prefix caching on (SGLang default)
CUDA_GRAPHS="${CUDA_GRAPHS:-1}"
DTYPE="${DTYPE:-auto}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
ENABLE_METRICS="${ENABLE_METRICS:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
LOG_DIR="${LOG_DIR:-runs}"

mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
CMD_FILE="$LOG_DIR/${STAMP}-sglang-cmd.txt"
LOG_FILE="$LOG_DIR/${STAMP}-sglang.log"

ARGS=(
  --model-path "$MODEL"
  --served-model-name "$SERVED_NAME"
  --host "$HOST" --port "$PORT"
  --tp "$TP"
  --mem-fraction-static "$MEM_FRACTION"
  --context-length "$CONTEXT_LEN"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --dtype "$DTYPE"
)

[[ "$CHUNKED_PREFILL_SIZE" != "-1" ]] && ARGS+=(--chunked-prefill-size "$CHUNKED_PREFILL_SIZE")
[[ -n "$QUANT" ]] && ARGS+=(--quantization "$QUANT")
[[ "$KV_CACHE_DTYPE" != "auto" ]] && ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
[[ "$RADIX_CACHE" == "0" ]] && ARGS+=(--disable-radix-cache)
[[ "$CUDA_GRAPHS" == "0" ]] && ARGS+=(--disable-cuda-graph)
[[ "$TRUST_REMOTE_CODE" == "1" ]] && ARGS+=(--trust-remote-code)
[[ "$ENABLE_METRICS" == "1" ]] && ARGS+=(--enable-metrics)
if [[ -n "$EXTRA_ARGS" ]]; then
  read -r -a _extra <<<"$EXTRA_ARGS"
  ARGS+=("${_extra[@]}")
fi

FULL=(python3 -m sglang.launch_server "${ARGS[@]}")
printf '%s\n' "${FULL[@]}" >"$CMD_FILE"
{
  echo "# resolved at $(date -Is)"
  echo "# sglang: $(python3 -c 'import sglang;print(sglang.__version__)' 2>/dev/null || echo unknown)"
  echo "# gpus:"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    | sed 's/^/#   /' || echo "#   (nvidia-smi unavailable)"
} >>"$CMD_FILE"

echo "[serve_sglang] cmd -> $CMD_FILE"
echo "[serve_sglang] log -> $LOG_FILE"
echo "[serve_sglang] OpenAI-compatible base url: http://${HOST}:${PORT}/v1"
exec "${FULL[@]}" >>"$LOG_FILE" 2>&1
