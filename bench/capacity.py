#!/usr/bin/env python3
"""
capacity.py — Turn benchmark runs into a K8s capacity plan.

It reads every run under a results dir, groups them by the *swept* variable
(concurrency for closed-loop, request_rate for open-loop) and answers:

  1. Where is the knee?        (max load that still meets the SLO)
  2. What is peak throughput?  (and at what concurrency / cost per 1k tokens)
  3. How many replicas do I need for a target RPS with headroom?
  4. What should the HPA target be?
  5. Is the bottleneck prefill (compute) or decode (memory bandwidth)?

Usage
-----
    python3 capacity.py results \
        --target-rps 40 --slo-ttft-p99-ms 2000 --slo-tpot-p99-ms 100 \
        --headroom 0.3 --gpu-hours-cost 2.5 --out capacity-plan.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Run:
    tag: str
    dir: str
    concurrency: float
    rate: Optional[float]
    mode: str
    rps: float
    out_tok_s: float
    tot_tok_s: float
    ttft_p99: float
    ttft_mean: float
    tpot_p99: float
    tpot_mean: float
    e2e_p99: float
    err: float
    in_tok: float
    out_tok: float
    goodput: Optional[float]
    kv_util: Optional[float]
    queue: Optional[float]
    gpu_util: Optional[float]
    preemptions: Optional[float]

    @classmethod
    def from_dir(cls, d: Path) -> Optional["Run"]:
        try:
            s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        except Exception:
            return None
        m = {}
        mp = d / "meta.json"
        if mp.exists():
            try:
                m = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:
                m = {}
        cfg = m.get("config", {})
        ser = (s.get("server") or {}).get("series") or {}

        def sv(k: str, stat: str = "mean") -> Optional[float]:
            v = (ser.get(k) or {}).get(stat)
            return float(v) if v is not None else None

        lat = s.get("latency_ms", {})
        th = s.get("throughput", {})
        return cls(
            tag=s.get("run_id", d.name), dir=str(d),
            concurrency=float(cfg.get("max_concurrency") or 0),
            rate=cfg.get("request_rate"), mode=cfg.get("mode", "?"),
            rps=float(th.get("requests_per_sec") or 0),
            out_tok_s=float(th.get("output_tokens_per_sec") or 0),
            tot_tok_s=float(th.get("total_tokens_per_sec") or 0),
            ttft_p99=float((lat.get("ttft") or {}).get("p99") or float("nan")),
            ttft_mean=float((lat.get("ttft") or {}).get("mean") or float("nan")),
            tpot_p99=float((lat.get("tpot") or {}).get("p99") or float("nan")),
            tpot_mean=float((lat.get("tpot") or {}).get("mean") or float("nan")),
            e2e_p99=float((lat.get("e2e") or {}).get("p99") or float("nan")),
            err=float(s.get("error_rate") or 0),
            in_tok=float((s.get("tokens") or {}).get("prompt_tokens_mean") or 0),
            out_tok=float((s.get("tokens") or {}).get("completion_tokens_mean") or 0),
            goodput=(s.get("slo") or {}).get("goodput_rps"),
            kv_util=sv("kv_cache_util"), queue=sv("requests_waiting", "p99"),
            gpu_util=sv("gpu_util"), preemptions=sv("preemptions_total_rate"),
        )


def meets_slo(r: Run, ttft_ms: float, tpot_ms: float, max_err: float) -> bool:
    ok = lambda v: v == v and v <= math.inf
    return (r.err <= max_err
            and (r.ttft_p99 == r.ttft_p99 and r.ttft_p99 <= ttft_ms)
            and (r.tpot_p99 == r.tpot_p99 and r.tpot_p99 <= tpot_ms)
            and ok(r.rps) and r.rps > 0)


def knee(runs: List[Run], key: str, ttft_ms: float, tpot_ms: float,
         max_err: float) -> Optional[Run]:
    good = [r for r in runs if meets_slo(r, ttft_ms, tpot_ms, max_err)]
    if not good:
        return None
    return max(good, key=lambda r: getattr(r, key))


def bottleneck(r: Run) -> str:
    """Cheap, honest diagnosis from the signals we actually collected.

    Order matters: preemption and KV exhaustion are hard resource walls, so
    they are checked before the softer statistical signals. Every branch names
    the action to take, because a diagnosis without an action is just trivia.
    """
    tail_ratio = (r.ttft_p99 / r.ttft_mean) if (r.ttft_mean and r.ttft_mean == r.ttft_mean
                                                and r.ttft_mean > 0) else float("nan")
    if r.preemptions and r.preemptions > 0.01:
        return ("**KV cache pressure** — sequences are being preempted and "
                "recomputed. Lower max-num-seqs, raise gpu-memory-utilization, "
                "shrink the KV dtype, or add replicas.")
    if r.kv_util is not None and r.kv_util > 0.9:
        return ("**KV cache bound** — >90% cache utilisation: concurrency is "
                "capped by memory, not compute. More replicas or shorter "
                "max-model-len; quantizing weights frees VRAM for KV blocks.")
    if r.queue and r.queue > 1:
        return ("**Admission bound** — requests are waiting for a slot. The "
                "engine's max-num-seqs is the ceiling; raising it past the knee "
                "buys throughput with a TPOT cliff, so scale out instead.")
    if r.gpu_util is not None and r.gpu_util > 92:
        return ("**Compute bound** — SM saturating. Scale out, quantize, or add "
                "tensor parallelism (which trades latency for GPU count).")
    if r.gpu_util is not None and r.gpu_util < 55 and r.in_tok and r.in_tok > 512:
        return ("**Prefill/batching bound** — GPU util low while long prompts "
                "are served: the batch is not filling. Raise the offered rate or "
                "max-num-seqs, enable chunked prefill, or check that the load "
                "generator is not the bottleneck.")
    if r.kv_util is not None and r.kv_util > 0.75:
        return ("**Approaching KV cache bound** — utilisation >75% with no "
                "preemptions yet. Headroom is thin; the next traffic step will "
                "likely cross it.")
    if tail_ratio == tail_ratio and tail_ratio > 3:
        return (f"**Latency knee without a resource wall** — TTFT p99 is "
                f"{tail_ratio:.1f}x the mean while KV/GPU look healthy. That is "
                "queueing variance: arrivals are bursty relative to service "
                "time. Verify with an open-loop run at the same mean rate, and "
                "check for a noisy neighbour or a slow replica in the fleet.")
    return ("Not saturated in this run — no resource signal is near its limit. "
            "Increase the load before drawing conclusions.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", help="results dir (or a single run dir)")
    p.add_argument("--target-rps", type=float, default=0,
                   help="production peak request rate to plan for")
    p.add_argument("--slo-ttft-p99-ms", type=float, default=2000)
    p.add_argument("--slo-tpot-p99-ms", type=float, default=100)
    p.add_argument("--max-error-rate", type=float, default=0.001)
    p.add_argument("--headroom", type=float, default=0.3,
                   help="fraction of extra capacity to keep (N+1 / burst)")
    p.add_argument("--gpu-hours-cost", type=float, default=0.0,
                   help="cost per GPU-hour, enables $/1M tokens")
    p.add_argument("--gpus-per-replica", type=int, default=1)
    p.add_argument("--tag-filter", default="", help="only runs whose id contains this")
    p.add_argument("--out", default="-")
    a = p.parse_args()

    root = Path(a.results)
    dirs = [root] if (root / "summary.json").exists() else sorted(
        d for d in root.rglob("*") if d.is_dir() and (d / "summary.json").exists())
    runs = [r for r in (Run.from_dir(d) for d in dirs) if r is not None]
    if a.tag_filter:
        runs = [r for r in runs if a.tag_filter in r.tag]
    if not runs:
        print(f"[capacity] no runs found under {root}", file=sys.stderr)
        return 1
    runs.sort(key=lambda r: (r.mode, r.concurrency, r.rate or 0))

    L: List[str] = []
    L.append("# Capacity plan (derived from measured runs)\n")
    L.append(f"SLO: TTFT p99 ≤ {a.slo_ttft_p99_ms:.0f} ms · TPOT p99 ≤ "
             f"{a.slo_tpot_p99_ms:.0f} ms · error rate ≤ {a.max_error_rate:.3%}  ")
    L.append(f"Headroom: {a.headroom:.0%} · GPUs/replica: {a.gpus_per_replica}\n")

    L.append("## All runs\n")
    L.append("| run | mode | conc | rate | RPS | out tok/s | TTFT p99 | TPOT p99 | err | KV util | queue p99 | GPU util | SLO |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    def opt(v, fmt="{:.2f}"):
        return "—" if v is None else fmt.format(v)

    for r in runs:
        rate = opt(r.rate, "{:g}")
        kv = opt(r.kv_util)
        q = opt(r.queue, "{:.1f}")
        gu = opt(r.gpu_util, "{:.0f}")
        slo = "✅" if meets_slo(r, a.slo_ttft_p99_ms, a.slo_tpot_p99_ms,
                                a.max_error_rate) else "❌"
        L.append(f"| `{r.tag}` | {r.mode} | {r.concurrency:.0f} | {rate} | "
                 f"{r.rps:.2f} | {r.out_tok_s:.0f} | {r.ttft_p99:.0f} | "
                 f"{r.tpot_p99:.1f} | {r.err:.2%} | {kv} | {q} | {gu} | {slo} |")
    L.append("")

    for mode, key, xlab in (("closed", "concurrency", "concurrency"),
                            ("open", "rate", "offered rate (rps)")):
        sub = [r for r in runs if r.mode == mode]
        if not sub:
            continue
        k = knee(sub, key, a.slo_ttft_p99_ms, a.slo_tpot_p99_ms, a.max_error_rate)
        peak = max(sub, key=lambda r: r.out_tok_s)
        L.append(f"## {mode}-loop sweep ({xlab} is the independent variable)\n")
        if k:
            L.append(f"* **Knee (max SLO-compliant {xlab})**: "
                     f"`{k.tag}` → {xlab}={getattr(k, key):g}, "
                     f"**{k.rps:.2f} rps**, **{k.out_tok_s:.0f} out tok/s**, "
                     f"TTFT p99 {k.ttft_p99:.0f} ms, TPOT p99 {k.tpot_p99:.1f} ms")
            # the knee itself usually looks healthy — the *next* point is what
            # tells you which resource ran out first
            nxt = next((r for r in sorted(sub, key=lambda r: getattr(r, key))
                        if getattr(r, key) > getattr(k, key)), None)
            if nxt:
                L.append(f"* First SLO-violating point after the knee: "
                         f"`{nxt.tag}` → {xlab}={getattr(nxt, key):g}, "
                         f"TTFT p99 {nxt.ttft_p99:.0f} ms "
                         f"({nxt.ttft_p99 / max(1e-9, k.ttft_p99):.0f}x the knee), "
                         f"throughput {nxt.out_tok_s:.0f} out tok/s "
                         f"({(nxt.out_tok_s / max(1e-9, k.out_tok_s) - 1) * 100:+.1f}% vs knee)")
                L.append(f"* **Why it breaks there**: {bottleneck(nxt)}")
                L.append(f"* State at the knee: {bottleneck(k)}")
            else:
                L.append(f"* Diagnosis at the knee: {bottleneck(k)}")
                L.append("* The sweep never crossed the knee — extend it to higher "
                         "load before trusting the capacity number.")
        else:
            L.append("* **No run met the SLO.** Either the SLO is too tight for "
                     "this workload/hardware, or the load never got low enough. "
                     "Add runs before sizing the cluster.")
        L.append(f"* **Peak decode throughput**: `{peak.tag}` → "
                 f"{peak.out_tok_s:.0f} out tok/s at {xlab}={getattr(peak, key):g} "
                 f"(TTFT p99 {peak.ttft_p99:.0f} ms — "
                 f"{'SLO OK' if meets_slo(peak, a.slo_ttft_p99_ms, a.slo_tpot_p99_ms, a.max_error_rate) else 'SLO violated, throughput-only number'})")
        if a.gpu_hours_cost > 0 and peak.out_tok_s > 0:
            cost = a.gpu_hours_cost * a.gpus_per_replica / (peak.out_tok_s * 3600) * 1e6
            L.append(f"* **Cost**: ≈ ${cost:,.2f} per 1M output tokens at peak "
                     f"(${a.gpu_hours_cost}/GPU-hour × {a.gpus_per_replica} GPUs)")
        L.append("")

    if a.target_rps > 0:
        closed = [r for r in runs if r.mode == "closed"]
        base = knee(closed or runs, "concurrency", a.slo_ttft_p99_ms,
                    a.slo_tpot_p99_ms, a.max_error_rate)
        if not base:
            L.append("## Replica sizing\n\n*Cannot size: no SLO-compliant run.*")
        else:
            need = a.target_rps * (1 + a.headroom)
            replicas = max(1, math.ceil(need / base.rps))
            L.append("## Replica sizing\n")
            L.append(f"| item | value |")
            L.append(f"|---|---|")
            L.append(f"| per-replica SLO-compliant RPS (measured) | **{base.rps:.2f}** |")
            L.append(f"| target peak RPS | {a.target_rps:g} |")
            L.append(f"| + headroom | {need:.2f} |")
            L.append(f"| **replicas required** | **{replicas}** |")
            L.append(f"| **GPUs required** | **{replicas * a.gpus_per_replica}** |")
            L.append(f"| minAvailable (PDB) | {max(1, replicas - (1 if replicas > 2 else 0))} |")
            L.append(f"| per-replica in-flight cap (`--max-num-seqs`) | {base.concurrency:.0f} |")
            L.append("")
            L.append("### HPA settings derived from the same numbers\n")
            L.append("```yaml")
            L.append("minReplicas: " + str(max(1, replicas - 1)))
            L.append("maxReplicas: " + str(replicas * 2))
            L.append("# scale on the measured knee, not on CPU:")
            L.append(f"targetAverageUtilization_or_value: {base.rps * 0.75:.2f}  # rps per pod, 75% of knee")
            L.append("# or, with the Prometheus adapter, on the engine's own signal:")
            L.append("#   vllm:num_requests_waiting > 1  (sustained 60s) -> scale up")
            L.append("#   vllm:gpu_cache_usage_perc > 0.85 (sustained 120s) -> scale up")
            L.append("behavior:")
            L.append("  scaleUp:   { stabilizationWindowSeconds: 60,  policies: [{type: Percent, value: 100, periodSeconds: 60}] }")
            L.append("  scaleDown: { stabilizationWindowSeconds: 600, policies: [{type: Pods, value: 1, periodSeconds: 120}] }")
            L.append("```")
            L.append("")
            L.append("> Cold-start caveat: a 7B replica needs model load + CUDA graph "
                     "capture before it can serve. Measure `pod_ready_seconds` in "
                     "your cluster and set `scaleUp.stabilizationWindowSeconds` and "
                     "the readiness probe `initialDelaySeconds` from that number, "
                     "otherwise the HPA will scale up too late for traffic spikes.")

    txt = "\n".join(L) + "\n"
    if a.out == "-":
        print(txt)
    else:
        Path(a.out).write_text(txt, encoding="utf-8")
        print(f"[capacity] wrote {a.out} from {len(runs)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
