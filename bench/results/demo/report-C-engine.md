# Experiment C — vLLM vs SGLang @ concurrency 12

Baseline: **fp16-conc12**  ·  Runs compared: 2

## Headline metrics

| Metric | fp16-conc12 | sglang-conc12 |
|---|---|---|
| RPS (achieved) | 3.45 | 3.88<br>`+12.5% ✅` |
| Out tok/s | 883.29 | 993.34<br>`+12.5% ✅` |
| Total tok/s | 5,095 | 5,729<br>`+12.5% ✅` |
| TTFT mean (ms) | 131.39 | 118.87<br>`-9.5% ✅` |
| TTFT p50 (ms) | 129.93 | 120.36<br>`-7.4% ✅` |
| TTFT p99 (ms) | 150.00 | 133.26<br>`-11.2% ✅` |
| TPOT mean (ms) | 11.62 | 10.37<br>`-10.8% ✅` |
| TPOT p99 (ms) | 12.14 | 10.72<br>`-11.7% ✅` |
| ITL p99 (ms) | 14.47 | 12.97<br>`-10.4% ✅` |
| E2E p99 (ms) | 3,230 | 2,863<br>`-11.4% ✅` |
| Goodput RPS | 3.45 | 3.88<br>`+12.5% ✅` |
| SLO attainment | 1.00 | 1.00<br>`+0.0% =` |
| Error rate | 0.00 | 0.00<br>`` |
| Out tok/req | 256.00 | 256.00<br>`+0.0% →` |
| In tok/req | 1,221 | 1,221<br>`+0.0% →` |

## Server-side signals during the run (mean, and p99 in brackets)

| Signal | fp16-conc12 | sglang-conc12 |
|---|---|---|
| Queue depth (waiting) | 0.00 [0.00] | 0.00 [0.00] |
| Running requests | 9.60 [12.00] | 9.33 [12.00] |
| KV cache util | 0.68 [0.85] | 0.57 [0.73] |
| Prefix-cache hit rate | 0.00 [0.00] | 0.00 [0.00] |
| Preemptions /s | 0.00 [0.00] | — |
| Engine decode tok/s (SGLang gauge) | — | 2,133 [2,743] |
| GPU util % | 32.15 [51.37] | 33.00 [46.88] |
| GPU mem used MiB | 30,000 [30,000] | 30,000 [30,000] |
| SM active | 0.05 [0.05] | 0.05 [0.05] |
| Tensor pipe active | 0.02 [0.02] | 0.02 [0.02] |
| Power W | 201.24 [254.52] | 201.47 [238.13] |
| GPU temp C | 50.10 [53.82] | 50.33 [52.50] |

## Per-run counter deltas (window-scoped — use these for A/B)

> Engine gauges like `prefix_cache_hit_rate` are cumulative since process start, so two runs on the same server report nearly identical values. These rows are differenced over the measurement window instead.

| Signal | fp16-conc12 | sglang-conc12 |
|---|---|---|
| Prefix-cache hit ratio (this run only) | 0.00 | — |
| Prefix-cache hits tok/s | 0.00 | — |
| Engine prefill tok/s | 3,252 | 3,657 |
| Engine decode tok/s | 682.36 | 767.42 |
| Requests seen by engine in window | 32.00 | 32.00 |
| Engine-side TTFT mean (ms) | 118.60 | 107.27 |
| Engine-side TPOT mean (ms) | 11.82 | 10.55 |
| Preemptions in window | 0.00 | — |
| Window length (s) | 9.00 | 8.01 |

## Fixed workload & environment (as recorded)

| Field | fp16-conc12 | sglang-conc12 |
|---|---|---|
| full run id | `20261008-045904-fp16-conc12` | `20261008-050954-sglang-conc12` |
| path | `results/demo/20261008-045904-fp16-conc12` | `results/demo/20261008-050954-sglang-conc12` |
| `config.input_len` | 1024 | 1024 |
| `config.output_len` | 256 | 256 |
| `config.mode` | closed | closed |
| `config.request_rate` | 2.00 | 2.00 |
| `config.max_concurrency` | 12 | 12 |
| `config.num_requests` | 32 | 32 |
| `config.endpoint` | chat | chat |
| `config.force_output_len` | True | True |
| `config.tokenizer` |  |  |
| `prompt_builder_method` | heuristic | heuristic |
| `config.notes` | closed-loop sweep, concurrency=12 | closed-loop sweep, concurrency=12 |
| loadgen platform | Linux-6.1.158+-x86_64-with-glibc2.41 | Linux-6.1.158+-x86_64-with-glibc2.41 |
| loadgen vCPU | 2 | 2 |
| repo commit |  |  |
| started at | 2026-10-08T04:59:06+0000 | 2026-10-08T05:09:56+0000 |
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
