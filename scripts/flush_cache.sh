#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# flush_cache.sh — clear the engine's prefix cache before a benchmark run.
#
# WHY THIS EXISTS (found by the smoke test, not by theory):
#   The prefix cache is process-global and persists ACROSS benchmark runs.
#   Because loadgen uses a fixed seed, run N and run N+1 generate the *same*
#   prompt sequence — so run N+1 starts with an almost fully warm cache and
#   reports a ~100% hit ratio even when shared_prefix_tokens is 0.
#   We measured exactly this: a control arm that should have shown 0% hits
#   showed 100%, which would have produced a completely wrong conclusion
#   about prefix caching.
#
#   Two ways to get a clean measurement:
#     (a) flush the cache before every run  <- this script
#     (b) restart the engine between variants
#   Do (a) between runs of the same variant, and (b) between variants.
#
#   ./scripts/flush_cache.sh http://127.0.0.1:8000
# ---------------------------------------------------------------------------
set -uo pipefail

URL="${1:-http://127.0.0.1:8000}"
URL="${URL%/}"

try() { # try <method> <path> <label>
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 10 -X "$1" "${URL}$2" 2>/dev/null || echo 000)
  if [[ "$code" =~ ^2 ]]; then
    echo "[flush] $3 -> $2 = HTTP $code  ✅"
    return 0
  fi
  echo "[flush] $3 -> $2 = HTTP $code"
  return 1
}

ok=1
# vLLM: POST /reset_prefix_cache (available when prefix caching is enabled)
try POST /reset_prefix_cache "vLLM prefix cache" || ok=0
# SGLang: POST /flush_cache (radix cache + KV pool)
try POST /flush_cache "SGLang radix cache" || true

if [[ $ok -ne 0 ]]; then
  echo "[flush] WARN could not flush via API."
  echo "[flush]      vLLM: needs --enable-prefix-caching and a version exposing"
  echo "[flush]            POST /reset_prefix_cache (0.6.x+). Otherwise restart."
  echo "[flush]      If you cannot flush, you MUST restart the engine between runs,"
  echo "[flush]      or every run after the first inherits a warm cache."
fi

# Report the post-flush state so the run record shows the cache really is cold.
sleep 1
echo "[flush] post-flush engine state:"
curl -s -m 10 "${URL}/metrics" 2>/dev/null \
  | grep -E "prefix_cache_(hit_rate|hits_total|queries_total)|cache_hit_rate|token_usage" \
  | grep -v '^#' | sed 's/^/          /' || echo "          (no /metrics)"
exit 0
