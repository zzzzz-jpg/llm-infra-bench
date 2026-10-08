# Capacity plan (derived from measured runs)

SLO: TTFT p99 ≤ 1500 ms · TPOT p99 ≤ 50 ms · error rate ≤ 0.100%  
Headroom: 30% · GPUs/replica: 1

## All runs

| run | mode | conc | rate | RPS | out tok/s | TTFT p99 | TPOT p99 | err | KV util | queue p99 | GPU util | SLO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `20261008-045818-fp16-conc4` | closed | 4 | 2 | 1.51 | 386 | 149 | 10.0 | 0.00% | 0.27 | 0.0 | 83 | ✅ |
| `20261008-045846-fp16-conc8` | closed | 8 | 2 | 2.75 | 703 | 154 | 10.9 | 0.00% | 0.52 | 0.0 | 76 | ✅ |
| `20261008-045904-fp16-conc12` | closed | 12 | 2 | 3.45 | 883 | 150 | 12.1 | 0.00% | 0.68 | 0.0 | 32 | ✅ |
| `20261008-045920-fp16-conc16` | closed | 16 | 2 | 3.48 | 891 | 3266 | 11.9 | 0.00% | 0.68 | 4.0 | 37 | ❌ |
| `20261008-045936-fp16-conc24` | closed | 24 | 2 | 3.48 | 891 | 3288 | 11.9 | 0.00% | 0.68 | 12.0 | 42 | ❌ |

## closed-loop sweep (concurrency is the independent variable)

* **Knee (max SLO-compliant concurrency)**: `20261008-045904-fp16-conc12` → concurrency=12, **3.45 rps**, **883 out tok/s**, TTFT p99 150 ms, TPOT p99 12.1 ms
* First SLO-violating point after the knee: `20261008-045920-fp16-conc16` → concurrency=16, TTFT p99 3266 ms (22x the knee), throughput 891 out tok/s (+0.9% vs knee)
* **Why it breaks there**: **Admission bound** — requests are waiting for a slot. The engine's max-num-seqs is the ceiling; raising it past the knee buys throughput with a TPOT cliff, so scale out instead.
* State at the knee: **Prefill/batching bound** — GPU util low while long prompts are served: the batch is not filling. Raise the offered rate or max-num-seqs, enable chunked prefill, or check that the load generator is not the bottleneck.
* **Peak decode throughput**: `20261008-045920-fp16-conc16` → 891 out tok/s at concurrency=16 (TTFT p99 3266 ms — SLO violated, throughput-only number)
* **Cost**: ≈ $0.78 per 1M output tokens at peak ($2.5/GPU-hour × 1 GPUs)

## Replica sizing

| item | value |
|---|---|
| per-replica SLO-compliant RPS (measured) | **3.45** |
| target peak RPS | 20 |
| + headroom | 26.00 |
| **replicas required** | **8** |
| **GPUs required** | **8** |
| minAvailable (PDB) | 7 |
| per-replica in-flight cap (`--max-num-seqs`) | 12 |

### HPA settings derived from the same numbers

```yaml
minReplicas: 7
maxReplicas: 16
# scale on the measured knee, not on CPU:
targetAverageUtilization_or_value: 2.59  # rps per pod, 75% of knee
# or, with the Prometheus adapter, on the engine's own signal:
#   vllm:num_requests_waiting > 1  (sustained 60s) -> scale up
#   vllm:gpu_cache_usage_perc > 0.85 (sustained 120s) -> scale up
behavior:
  scaleUp:   { stabilizationWindowSeconds: 60,  policies: [{type: Percent, value: 100, periodSeconds: 60}] }
  scaleDown: { stabilizationWindowSeconds: 600, policies: [{type: Pods, value: 1, periodSeconds: 120}] }
```

> Cold-start caveat: a 7B replica needs model load + CUDA graph capture before it can serve. Measure `pod_ready_seconds` in your cluster and set `scaleUp.stabilizationWindowSeconds` and the readiness probe `initialDelaySeconds` from that number, otherwise the HPA will scale up too late for traffic spikes.
