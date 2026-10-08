#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# demo_mock_matrix.sh — run the FULL experiment matrix against the mock
# topology and produce every report the project generates.
#
# This is the "does the whole pipeline work" check: no GPU, no cluster, ~4 min.
# It exercises all three A/B variables from requirement 4
#   A. quantization      fp16 (:8000)  vs  int4/AWQ (:8003)
#   B. prefix caching    ON   (:8000)  vs  OFF      (:8002)
#   C. engine            vLLM (:8000)  vs  SGLang   (:8001)
# and then the rate/concurrency sweeps that feed the capacity plan.
#
# The numbers are SYNTHETIC (mock server), the pipeline is real. Use this to
# validate your fork of the harness before spending GPU hours.
#
#   ./scripts/dev_up.sh && ./scripts/demo_mock_matrix.sh
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

OUT="${OUT:-$HERE/bench/results/demo}"
N_REQ="${N_REQ:-24}"
CONCS="${CONCS:-4 8 12 16 24}"
RATES="${RATES:-2 4 6 8}"
COOLDOWN="${COOLDOWN:-2}"
export COOLDOWN N_REQ CONCS RATES

GPU_M="http://127.0.0.1:9400/metrics"
VLLM="http://127.0.0.1:8000"
SGLANG="http://127.0.0.1:8001"
VLLM_NPC="http://127.0.0.1:8002"
VLLM_AWQ="http://127.0.0.1:8003"

for u in "$VLLM" "$SGLANG" "$VLLM_NPC" "$VLLM_AWQ"; do
  curl -sf -m 3 "$u/health" >/dev/null || {
    echo "ERROR: nothing listening at $u — run ./scripts/dev_up.sh first" >&2; exit 1; }
done

mkdir -p "$OUT"
echo "### environment snapshot"
python3 scripts/record_env.py --out "$OUT/env.json" --probe-url "$VLLM" || true

M="./scripts/run_matrix.sh --num-requests $N_REQ --out-dir $OUT --gpu-metrics $GPU_M"

echo; echo "### A · quantization: FP16 vs AWQ int4 (concurrency sweep)"
$M --tag fp16 --suite conc --url "$VLLM"
$M --tag awq  --suite conc --url "$VLLM_AWQ"

echo; echo "### B · prefix caching: ON vs OFF (shared-prefix workload)"
$M --tag pc-on  --suite prefix --url "$VLLM"
$M --tag pc-off --suite prefix --url "$VLLM_NPC"

echo; echo "### C · engine: vLLM vs SGLang (concurrency sweep)"
$M --tag sglang --suite conc --url "$SGLANG"

echo; echo "### D · open-loop rate sweep on the baseline variant"
$M --tag ratesweep --suite rate --url "$VLLM"

cd bench
pick() { ls -d "$OUT"/*"$1"* 2>/dev/null | tail -1; }

echo; echo "### reports"
python3 compare.py "$(pick fp16-conc12)" "$(pick awq-conc12)" \
  --baseline-tag fp16 --title "Experiment A — Quantization: FP16 vs AWQ int4 @ concurrency 12" \
  --out "$OUT/report-A-quantization.md"

python3 compare.py "$(pick pc-on-prefix512)" "$(pick pc-off-prefix512)" \
  --baseline-tag pc-off --title "Experiment B — Prefix Cache ON vs OFF (512-token shared prefix)" \
  --out "$OUT/report-B-prefixcache.md"

python3 compare.py "$(pick fp16-conc12)" "$(pick sglang-conc12)" \
  --baseline-tag fp16 --title "Experiment C — vLLM vs SGLang @ concurrency 12" \
  --out "$OUT/report-C-engine.md"

python3 capacity.py "$OUT" --tag-filter fp16-conc --target-rps 20 \
  --slo-ttft-p99-ms 1500 --slo-tpot-p99-ms 50 --headroom 0.3 \
  --gpu-hours-cost 2.5 --gpus-per-replica 1 --out "$OUT/capacity-plan.md"

python3 ../tests/verify_run.py "$OUT"

echo
echo "### done"
for f in "$OUT"/report-*.md "$OUT"/capacity-plan.md; do echo "  - $f"; done
