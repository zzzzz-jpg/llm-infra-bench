# Experiment B — Prefix Cache ON vs OFF (512-token shared prefix)

Baseline: **pc-off-prefix512**  ·  Runs compared: 2

## Headline metrics

| Metric | pc-on-prefix512 | pc-off-prefix512 |
|---|---|---|
| RPS (achieved) | 1.54<br>`+2.6% ✅` | 1.50 |
| Out tok/s | 394.18<br>`+2.6% ✅` | 384.11 |
| Total tok/s | 2,500<br>`+2.6% ✅` | 2,436 |
| TTFT mean (ms) | 66.95<br>`-50.7% ✅` | 135.84 |
| TTFT p50 (ms) | 65.31<br>`-52.3% ✅` | 137.00 |
| TTFT p99 (ms) | 77.00<br>`-50.7% ✅` | 156.25 |
| TPOT mean (ms) | 9.87<br>`-0.3% =` | 9.90 |
| TPOT p99 (ms) | 10.03<br>`-0.1% =` | 10.05 |
| ITL p99 (ms) | 12.17<br>`+0.0% =` | 12.17 |
| E2E p99 (ms) | 2,631<br>`-2.8% ✅` | 2,706 |
| Goodput RPS | 1.54<br>`+2.6% ✅` | 1.50 |
| SLO attainment | 1.00<br>`+0.0% =` | 1.00 |
| Error rate | 0.00<br>`` | 0.00 |
| Out tok/req | 256.00<br>`+0.0% →` | 256.00 |
| In tok/req | 1,368<br>`+0.0% →` | 1,368 |

## Server-side signals during the run (mean, and p99 in brackets)

| Signal | pc-on-prefix512 | pc-off-prefix512 |
|---|---|---|
| Queue depth (waiting) | 0.00 [0.00] | 0.00 [0.00] |
| Running requests | 3.81 [4.00] | 3.77 [4.00] |
| KV cache util | 0.27 [0.28] | 0.27 [0.28] |
| Prefix-cache hit rate | 0.11 [0.16] | 0.00 [0.00] |
| Preemptions /s | 0.00 [0.00] | 0.00 [0.00] |
| Engine decode tok/s (SGLang gauge) | — | — |
| GPU util % | 36.81 [49.20] | 26.70 [43.37] |
| GPU mem used MiB | 30,000 [30,000] | 30,000 [30,000] |
| SM active | 0.05 [0.05] | 0.05 [0.05] |
| Tensor pipe active | 0.02 [0.02] | 0.02 [0.02] |
| Power W | 212.51 [245.28] | 186.11 [227.94] |
| GPU temp C | 51.00 [53.40] | 49.34 [52.40] |

## Per-run counter deltas (window-scoped — use these for A/B)

> Engine gauges like `prefix_cache_hit_rate` are cumulative since process start, so two runs on the same server report nearly identical values. These rows are differenced over the measurement window instead.

| Signal | pc-on-prefix512 | pc-off-prefix512 |
|---|---|---|
| Prefix-cache hit ratio (this run only) | 0.99 | 0.00 |
| Prefix-cache hits tok/s | 2,176 | 0.00 |
| Engine prefill tok/s | 1,913 | 1,822 |
| Engine decode tok/s | 358.17 | 341.12 |
| Requests seen by engine in window | 32.00 | 32.00 |
| Engine-side TTFT mean (ms) | 56.70 | 124.84 |
| Engine-side TPOT mean (ms) | 9.87 | 9.91 |
| Preemptions in window | 0.00 | 0.00 |
| Window length (s) | 20.01 | 21.01 |

## Fixed workload & environment (as recorded)

| Field | pc-on-prefix512 | pc-off-prefix512 |
|---|---|---|
| full run id | `20261008-050130-pc-on-prefix512` | `20261008-050230-pc-off-prefix512` |
| path | `results/demo/20261008-050130-pc-on-prefix512` | `results/demo/20261008-050230-pc-off-prefix512` |
| `config.input_len` | 1024 | 1024 |
| `config.output_len` | 256 | 256 |
| `config.mode` | closed | closed |
| `config.request_rate` | 4.00 | 4.00 |
| `config.max_concurrency` | 4 | 4 |
| `config.num_requests` | 32 | 32 |
| `config.endpoint` | chat | chat |
| `config.force_output_len` | True | True |
| `config.tokenizer` |  |  |
| `prompt_builder_method` | heuristic | heuristic |
| `config.notes` | shared prefix 512 tokens, closed-loop conc=4 (unsaturated) | shared prefix 512 tokens, closed-loop conc=4 (unsaturated) |
| loadgen platform | Linux-6.1.158+-x86_64-with-glibc2.41 | Linux-6.1.158+-x86_64-with-glibc2.41 |
| loadgen vCPU | 2 | 2 |
| repo commit |  |  |
| started at | 2026-10-08T05:01:35+0000 | 2026-10-08T05:02:36+0000 |
| GPU | — | — |

## Verdict

_Fill in after reading the tables. Required shape:_

```text
Variable under test : <e.g. AWQ int4 vs FP16>
Throughput effect   : <out tok/s delta, with the CI/σ if repeats exist>
Latency effect      : <TTFT p99 / TPOT p99 delta>
Quality caveat      : <what the number does NOT tell you>
Decision            : <adopt / reject / adopt-with-condition>
```
