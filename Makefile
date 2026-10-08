# ---------------------------------------------------------------------------
# Makefile — the whole project driven by named entry points.
#   make help
# ---------------------------------------------------------------------------
SHELL := /usr/bin/env bash
PY    ?= python3

# --- where your real service lives (override on the command line) ----------
URL          ?= http://127.0.0.1:8000
MODEL        ?=
TOKENIZER    ?=
GPU_METRICS  ?= http://127.0.0.1:9400/metrics
RESULTS      ?= $(CURDIR)/bench/results
TAG          ?= run
SUITE        ?= baseline
N_REQ        ?= 200
RATES        ?= 1 2 4 6 8 12
CONCS        ?= 1 2 4 8 16 32 64
TARGET_RPS   ?= 20
SLO_TTFT     ?= 2000
SLO_TPOT     ?= 100
HEADROOM     ?= 0.3
GPU_COST     ?= 0
GPUS_PER_REPLICA ?= 1

export COOLDOWN N_REQ RATES CONCS

.PHONY: help install dev-up dev-down smoke test lint bench bench-rate bench-conc \
        bench-prefix report capacity all clean serve-vllm serve-sglang env \
        config-hash deploy deploy-prod bench-incluster demo

help:
	@echo "Setup / dev loop"
	@echo "  make install          install python deps"
	@echo "  make dev-up           start the CPU-only MOCK topology (no GPU needed)"
	@echo "  make smoke            run a tiny benchmark against the mock + verify outputs"
	@echo "  make test             pytest suite (harness correctness)"
	@echo "  make lint             kustomize build + kubeconform on every manifest"
	@echo
	@echo "Real service"
	@echo "  make env              snapshot hardware/software -> \$$RESULTS/env.json"
	@echo "  make serve-vllm       launch vLLM  (MODEL=... QUANT=... PREFIX_CACHING=...)"
	@echo "  make serve-sglang     launch SGLang (MODEL=... RADIX_CACHE=...)"
	@echo "  make bench            run one suite: TAG=vllm-fp16 SUITE=all URL=http://..."
	@echo "  make bench-rate       open-loop rate sweep"
	@echo "  make bench-conc       closed-loop concurrency sweep"
	@echo "  make bench-prefix     prefix-cache A/B"
	@echo
	@echo "Analysis"
	@echo "  make report           compare every run under \$$RESULTS"
	@echo "  make capacity         capacity plan + HPA numbers"
	@echo "  make all              lint + test + smoke"

install:
	$(PY) -m pip install -r requirements.txt

dev-up:
	./scripts/dev_up.sh

dev-down:
	./scripts/dev_down.sh

# full mock matrix: 3 A/B experiments + sweeps + all reports (~6 min, no GPU)
demo: dev-up-check
	./scripts/demo_mock_matrix.sh

dev-up-check:
	@curl -sf -m 3 http://127.0.0.1:8000/health >/dev/null || \
	  { echo "mock topology not running: make dev-up"; exit 1; }

env:
	$(PY) scripts/record_env.py --out $(RESULTS)/env.json --probe-url $(URL)

# ---- the mock smoke test: proves the pipeline end to end on any machine ----
smoke:
	@echo "[smoke] generating load against $(URL)"
	cd bench && $(PY) loadgen.py --base-url $(URL) --endpoint chat \
	  --mode closed --max-concurrency 8 --num-requests 24 --warmup-requests 4 \
	  --input-len 512 --output-len 64 --force-output-len \
	  --metrics-endpoint $(URL)/metrics --gpu-metrics-endpoint $(GPU_METRICS) \
	  --scrape-interval 1 --tag smoke --out-dir $(RESULTS)
	@echo "[smoke] asserting the artefacts exist and are non-degenerate"
	cd bench && $(PY) ../tests/verify_run.py $(RESULTS) --tag smoke

test:
	$(PY) -m pytest tests/ -q

lint:
	./scripts/lint_k8s.sh

# ---- real benchmarks ----
serve-vllm:
	./scripts/serve_vllm.sh

serve-sglang:
	./scripts/serve_sglang.sh

bench:
	./scripts/run_matrix.sh --tag $(TAG) --suite $(SUITE) --url $(URL) \
	  --model "$(MODEL)" --gpu-metrics $(GPU_METRICS) --out-dir $(RESULTS)

bench-rate:
	$(MAKE) bench SUITE=rate

bench-conc:
	$(MAKE) bench SUITE=conc

bench-prefix:
	$(MAKE) bench SUITE=prefix

report:
	cd bench && $(PY) compare.py $(RESULTS) --title "Comparison — $(TAG)" \
	  --baseline-tag "$(TAG)" --out $(RESULTS)/report-$(TAG).md
	@echo "  -> $(RESULTS)/report-$(TAG).md"

capacity:
	cd bench && $(PY) capacity.py $(RESULTS) --tag-filter "$(TAG)" \
	  --target-rps $(TARGET_RPS) --slo-ttft-p99-ms $(SLO_TTFT) \
	  --slo-tpot-p99-ms $(SLO_TPOT) --headroom $(HEADROOM) \
	  --gpu-hours-cost $(GPU_COST) --gpus-per-replica $(GPUS_PER_REPLICA) \
	  --out $(RESULTS)/capacity-$(TAG).md
	@echo "  -> $(RESULTS)/capacity-$(TAG).md"

# ConfigMap changes do NOT restart pods. This target re-annotates the
# Deployment with a hash of the engine args so the rollout actually happens,
# and the same hash lands in Prometheus as a label (see ServiceMonitor
# relabelings) so Grafana can split panels by deployed configuration.
config-hash:
	@H=$$(kubectl -n llm-inference get cm vllm-engine-args \
	      -o jsonpath='{.data.args\.txt}' | sha256sum | cut -d' ' -f1); \
	 echo "[config-hash] $$H"; \
	 kubectl -n llm-inference annotate deploy/vllm \
	   llm-inference/config-hash="$$H" --overwrite; \
	 kubectl -n llm-inference patch deploy/vllm --type=json \
	   -p "[{\"op\":\"replace\",\"path\":\"/spec/template/metadata/annotations/llm-inference~1config-hash\",\"value\":\"\"$$H\"\"}]"
	@echo "[config-hash] run 'kubectl -n llm-inference rollout status deploy/vllm' to watch"

deploy:
	kubectl apply -k k8s/base
	kubectl -n llm-inference rollout status deploy/vllm --timeout=20m

deploy-prod:
	kubectl apply -k k8s/overlays/prod
	kubectl -n llm-inference rollout status deploy/vllm --timeout=20m

bench-incluster:
	kubectl apply -f k8s/base/60-bench-job.yaml
	kubectl -n llm-inference wait --for=condition=complete job/inference-bench --timeout=30m

all: lint test smoke
	@echo "[all] PASS"

clean:
	rm -rf bench/results/2026* .tools .devlogs .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
