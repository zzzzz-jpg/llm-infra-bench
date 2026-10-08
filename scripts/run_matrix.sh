#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_matrix.sh — drive a whole experiment suite against ONE already-running
# server, then emit the comparison + capacity reports.
#
#   ./scripts/run_matrix.sh --tag vllm-fp16 --suite all
#   ./scripts/run_matrix.sh --tag vllm-awq  --suite conc --url http://127.0.0.1:8000
#
# Suites
#   baseline : the single fixed reference load
#   rate     : open-loop Poisson sweep  -> "what rps can I promise?"
#   conc     : closed-loop sweep        -> "peak throughput + the knee"
#   prefix   : same load, shared prefix 0 vs N -> isolates prefix-cache effect
#   all      : rate + conc + prefix
#
# Every run directory is self-describing (meta.json holds the full config and
# the env.json snapshot), so `compare.py` can later prove two runs are
# actually comparable instead of assuming it.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE/bench"

TAG="run"; SUITE="baseline"; URL="http://127.0.0.1:8000"; MODEL=""
GPU_METRICS="http://127.0.0.1:9400/metrics"
RATES="${RATES:-1 2 4 6 8 12}"; CONCS="${CONCS:-1 2 4 8 16 32 64}"
N_REQ="${N_REQ:-200}"; PREFIX_TOKENS="${PREFIX_TOKENS:-512}"; PREFIX_CONC="${PREFIX_CONC:-4}"
OUT_DIR="${OUT_DIR:-results}"; ENDPOINT="${ENDPOINT:-chat}"
# The engine's prefix cache is process-global and survives across runs; with a
# fixed seed every run replays the same prompts, so run N+1 would start with a
# warm cache. Flush before each run (or set FLUSH_CACHE=0 and restart the
# engine yourself between variants).
FLUSH_CACHE="${FLUSH_CACHE:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2;;
    --suite) SUITE="$2"; shift 2;;
    --url) URL="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --gpu-metrics) GPU_METRICS="$2"; shift 2;;
    --rates) RATES="$2"; shift 2;;
    --concs) CONCS="$2"; shift 2;;
    --num-requests) N_REQ="$2"; shift 2;;
    --endpoint) ENDPOINT="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    -h|--help) sed -n '2,25p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# --out-dir is interpreted relative to the REPO ROOT by callers, but we already
# cd'd into bench/ above — a naive resolve would land results in bench/<path>.
# Force repo-root semantics for relative paths, absolute paths pass through.
case "$OUT_DIR" in
  /*) ;;
  *)  OUT_DIR="$HERE/$OUT_DIR" ;;
esac
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
COMMON=(--base-url "$URL" --endpoint "$ENDPOINT" --out-dir "$OUT_DIR"
        --metrics-endpoint "${URL%/}/metrics" --gpu-metrics-endpoint "$GPU_METRICS")
[[ -n "$MODEL" ]] && COMMON+=(--model "$MODEL")

# record the environment once per matrix, not per run
python3 ../scripts/record_env.py --out "$OUT_DIR/env.json" --probe-url "$URL" || true

run() {  # run <tag> <extra args...>
  local t="$1"; shift
  echo "==================================================================="
  echo "[matrix ${TAG}] run ${t}"
  echo "==================================================================="
  if [[ "$FLUSH_CACHE" == "1" ]]; then
    "$HERE/scripts/flush_cache.sh" "$URL" >/dev/null 2>&1 || \
      echo "[matrix ${TAG}] note: cache flush failed for $URL (see flush_cache.sh)" >&2
  fi
  python3 loadgen.py "${COMMON[@]}" --tag "${TAG}-${t}" "$@" || \
    echo "[matrix ${TAG}] run ${t} FAILED (continuing)" >&2
  sleep "${COOLDOWN:-10}"      # let the KV cache drain between runs
}

case "$SUITE" in
  baseline)
    run baseline --profile profiles/baseline.yaml --num-requests "$N_REQ"
    ;;
  rate)
    for R in $RATES; do
      run "rate${R}" --profile profiles/baseline.yaml --mode open \
          --request-rate "$R" --num-requests "$N_REQ" \
          --notes "open-loop sweep, rate=${R} rps"
    done
    ;;
  conc)
    for C in $CONCS; do
      run "conc${C}" --profile profiles/saturation.yaml --mode closed \
          --max-concurrency "$C" --num-requests "$N_REQ" \
          --notes "closed-loop sweep, concurrency=${C}"
    done
    ;;
  prefix)
    # Closed-loop at LOW concurrency on purpose. Prefix caching saves prefill
    # work, i.e. tens of milliseconds of TTFT. Run this at saturation and the
    # queueing component of TTFT (seconds) swallows the effect and you will
    # "measure" a null result. See docs/04-experiments.md.
    run "prefix0" --profile profiles/baseline.yaml --num-requests "$N_REQ" \
        --mode closed --max-concurrency "$PREFIX_CONC" \
        --shared-prefix-tokens 0 --prompt-style shared_prefix \
        --notes "no shared prefix, closed-loop conc=${PREFIX_CONC} (unsaturated)"
    run "prefix${PREFIX_TOKENS}" --profile profiles/baseline.yaml \
        --num-requests "$N_REQ" --mode closed --max-concurrency "$PREFIX_CONC" \
        --shared-prefix-tokens "$PREFIX_TOKENS" --prompt-style shared_prefix \
        --notes "shared prefix ${PREFIX_TOKENS} tokens, closed-loop conc=${PREFIX_CONC} (unsaturated)"
    ;;
  all)
    # re-invoke by absolute path: we cd'd into bench/, so $0 is no longer valid
    SELF="$HERE/scripts/run_matrix.sh"
    for S in rate conc prefix; do
      "$SELF" --tag "$TAG" --suite "$S" --url "$URL" --model "$MODEL" \
              --num-requests "$N_REQ" --out-dir "$OUT_DIR" --endpoint "$ENDPOINT" \
              --gpu-metrics "$GPU_METRICS"
    done
    ;;
  *) echo "unknown suite: $SUITE" >&2; exit 2;;
esac

echo
echo "[matrix ${TAG}] done. Reports:"
python3 compare.py "$OUT_DIR" --baseline-tag "${TAG}-rate${RATES%% *}" \
  --title "Comparison — ${TAG}" --out "$OUT_DIR/report-${TAG}.md" || true
python3 capacity.py "$OUT_DIR" --tag-filter "$TAG" \
  --out "$OUT_DIR/capacity-${TAG}.md" || true
echo "  - $OUT_DIR/report-${TAG}.md"
echo "  - $OUT_DIR/capacity-${TAG}.md"
