#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# serve_vllm.sh — launch vLLM with an OpenAI-compatible API. Every knob the
# A/B experiments touch is an environment variable, and the resolved command
# line is written to disk so any benchmark run can be reproduced exactly.
#
#   MODEL=Qwen/Qwen2.5-7B-Instruct ./scripts/serve_vllm.sh
#   QUANT=awq MODEL=/models/Qwen2.5-7B-Instruct-AWQ ./scripts/serve_vllm.sh
#   PREFIX_CACHING=0 ./scripts/serve_vllm.sh
#
# Env knobs (defaults in brackets):
#   MODEL            [Qwen/Qwen2.5-7B-Instruct]   HF id or local path
#   SERVED_NAME      [$MODEL]                     name reported by /v1/models
#   HOST [0.0.0.0]  PORT [8000]
#   TP               [1]        tensor-parallel size (GPUs)
#   PP               [1]        pipeline-parallel size
#   GPU_MEM_UTIL     [0.90]     fraction of VRAM for weights + KV cache
#   MAX_MODEL_LEN    [8192]
#   MAX_NUM_SEQS     [256]      engine batch cap == the real concurrency ceiling
#   MAX_NUM_BATCHED_TOKENS [8192]
#   QUANT            [""]       awq | gptq | fp8 | marlin | bitsandbytes | ""
#   KV_CACHE_DTYPE   [auto]     auto | fp8 | fp8_e5m2
#   PREFIX_CACHING   [1]        1 = on, 0 = off
#   CHUNKED_PREFILL  [1]
#   CUDA_GRAPHS      [1]        0 = --enforce-eager (debuggable, slower)
#   SWAP_SPACE       [4]        GiB CPU swap
#   DTYPE            [auto]
#   TRUST_REMOTE_CODE[0]
#   EXTRA_ARGS       [""]       appended verbatim
#   LOG_DIR          [runs]
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SERVED_NAME="${SERVED_NAME:-$MODEL}"
HOST="${HOST:-0.0.0.0}"; PORT="${PORT:-8000}"
TP="${TP:-1}"; PP="${PP:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
QUANT="${QUANT:-}"; KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
PREFIX_CACHING="${PREFIX_CACHING:-1}"; CHUNKED_PREFILL="${CHUNKED_PREFILL:-1}"
CUDA_GRAPHS="${CUDA_GRAPHS:-1}"; SWAP_SPACE="${SWAP_SPACE:-4}"
DTYPE="${DTYPE:-auto}"; TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"; LOG_DIR="${LOG_DIR:-runs}"
ENTRYPOINT="${ENTRYPOINT:-vllm-serve}"   # vllm-serve | api-server

mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
CMD_FILE="$LOG_DIR/${STAMP}-serve-cmd.txt"
LOG_FILE="$LOG_DIR/${STAMP}-serve.log"

if [[ "$ENTRYPOINT" == "vllm-serve" ]]; then
  BASE=(vllm serve)
else
  BASE=(python3 -m vllm.entrypoints.openai.api_server)
fi

# Flag spelling for prefix caching / chunked prefill moved across vLLM releases
# (--enable-X  ->  --X / --no-X). Probe --help once and use what exists.
HELP_TEXT="$("${BASE[@]}" --help 2>/dev/null || true)"
has_flag() { grep -qE -- "(^|[[:space:]])$1([[:space:]=]|$)" <<<"$HELP_TEXT"; }

ARGS=(
  --model "$MODEL"
  --served-model-name "$SERVED_NAME"
  --host "$HOST" --port "$PORT"
  --tensor-parallel-size "$TP"
  --pipeline-parallel-size "$PP"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --dtype "$DTYPE"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --swap-space "$SWAP_SPACE"
  --disable-log-requests
)

[[ -n "$QUANT" ]] && ARGS+=(--quantization "$QUANT")
[[ "$TRUST_REMOTE_CODE" == "1" ]] && ARGS+=(--trust-remote-code)
[[ "$CUDA_GRAPHS" == "0" ]] && ARGS+=(--enforce-eager)

if [[ "$PREFIX_CACHING" == "1" ]]; then
  has_flag --enable-prefix-caching && ARGS+=(--enable-prefix-caching)
else
  if has_flag --no-enable-prefix-caching; then ARGS+=(--no-enable-prefix-caching)
  else ARGS+=(--no-prefix-caching); fi   # legacy spelling
fi
if [[ "$CHUNKED_PREFILL" == "1" ]]; then
  has_flag --enable-chunked-prefill && ARGS+=(--enable-chunked-prefill)
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  read -r -a _extra <<<"$EXTRA_ARGS"
  ARGS+=("${_extra[@]}")
fi

FULL=("${BASE[@]}" "${ARGS[@]}")
printf '%s\n' "${FULL[@]}" >"$CMD_FILE"
{
  echo "# resolved at $(date -Is)"
  echo "# vllm: $(python3 -c 'import vllm;print(vllm.__version__)' 2>/dev/null || echo unknown)"
  echo "# torch: $(python3 -c 'import torch;print(torch.__version__, torch.version.cuda)' 2>/dev/null || echo unknown)"
  echo "# gpus:"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    | sed 's/^/#   /' || echo "#   (nvidia-smi unavailable)"
} >>"$CMD_FILE"

echo "[serve_vllm] cmd -> $CMD_FILE"
echo "[serve_vllm] log -> $LOG_FILE"
exec "${FULL[@]}" >>"$LOG_FILE" 2>&1
