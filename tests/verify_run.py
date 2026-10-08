#!/usr/bin/env python3
"""
verify_run.py — assert that a completed loadgen run is structurally sound and
not silently degenerate. Used by `make smoke` and in CI, so a broken harness
fails the pipeline instead of producing plausible-looking nonsense.

    python3 tests/verify_run.py bench/results --tag smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def fail(msgs: List[str], m: str) -> None:
    msgs.append(m)


def check_run(d: Path):
    errs: List[str] = []
    for f in ("summary.json", "meta.json", "requests.csv"):
        if not (d / f).exists():
            fail(errs, f"{d}: missing {f}")
    if errs:
        return errs, []

    s: Dict[str, Any] = json.loads((d / "summary.json").read_text())
    m: Dict[str, Any] = json.loads((d / "meta.json").read_text())

    # --- the run actually happened ---------------------------------------
    if s["requests_total"] == 0:
        fail(errs, "zero requests recorded")
    if s["requests_ok"] == 0:
        fail(errs, f"zero successful requests; errors={s['errors_by_type']}")
    if s["wall_clock_sec"] <= 0:
        fail(errs, "non-positive wall clock")

    # --- the workload was what was asked for ------------------------------
    cfg = m.get("config", {})
    if cfg.get("force_output_len"):
        want = cfg.get("output_len")
        got = s["tokens"]["completion_tokens_mean"]
        if want and abs(got - want) > max(1, want * 0.02):
            fail(errs, f"output length not fixed: want {want}, mean {got}")
    if s["tokens"]["token_source"] in ([], ["chunks"]):
        fail(errs, "completion tokens were counted from SSE chunks, not from the "
                   "usage object — token throughput is unreliable")

    # --- latency is internally consistent --------------------------------
    lat = s["latency_ms"]
    for k in ("ttft", "tpot", "e2e"):
        st = lat.get(k) or {}
        if not st or st.get("count", 0) == 0:
            fail(errs, f"{k} has no samples")
            continue
        if not (st["min"] <= st["p50"] <= st["p99"] <= st["max"]):
            fail(errs, f"{k} percentiles not monotonic: {st}")
        if st["mean"] != st["mean"] or st["mean"] <= 0:
            fail(errs, f"{k} mean is NaN/non-positive")
    if lat["ttft"].get("p99", 0) > lat["e2e"].get("p99", 0) * 1.5 + 1:
        fail(errs, "TTFT p99 implausibly larger than E2E p99")

    # --- the load model did what it claimed ------------------------------
    rows = list(csv.DictReader((d / "requests.csv").open(encoding="utf-8")))
    bench = [r for r in rows if r["mode"] == "bench"]
    warm = [r for r in rows if r["mode"] == "warmup"]
    if len(bench) != s["requests_total"]:
        fail(errs, f"requests.csv bench rows {len(bench)} != summary total "
                   f"{s['requests_total']}")
    if cfg.get("warmup_requests") and len(warm) != cfg["warmup_requests"]:
        fail(errs, f"expected {cfg['warmup_requests']} warmup rows, got {len(warm)}")
    if cfg.get("mode") == "closed":
        ev = sorted([(float(r["start_ts"]), 1) for r in bench if r["start_ts"]]
                    + [(float(r["end_ts"]), -1) for r in bench if r["end_ts"]])
        cur = peak = 0
        for _, delta in ev:
            cur += delta
            peak = max(peak, cur)
        if peak > int(cfg["max_concurrency"]):
            fail(errs, f"closed-loop exceeded concurrency: peak {peak} > "
                       f"{cfg['max_concurrency']}")

    # --- provenance -------------------------------------------------------
    for k in ("config", "host", "started_at"):
        if k not in m:
            fail(errs, f"meta.json missing {k}")
    if not m.get("environment") and not (m.get("host") or {}).get("gpus"):
        fail(errs, "no GPU inventory recorded (env.json missing and host.gpus "
                   "empty) — hardware information is a required part of a run")
    if s["server"]["samples"] == 0:
        fail(errs, "no server-side metric samples: queue/KV/GPU columns will be "
                   "empty and the run cannot be diagnosed")
    if s["server"]["samples"] and not s["server"]["series"]:
        fail(errs, "samples collected but no series parsed — metric names do not "
                   "match the vLLM/SGLang/DCGM patterns in loadgen.INTERESTING")
    if s["server"]["scrape_errors"] > s["server"]["samples"]:
        fail(errs, f"scrape_errors {s['server']['scrape_errors']} exceeds samples "
                   f"{s['server']['samples']}")
    if s["error_rate"] > 0.05:
        fail(errs, f"error rate {s['error_rate']:.2%} — fix the service before "
                   "treating this run as data")

    # loadgen's own validity checks are advisory, not fatal: surface them so a
    # human reads them, but do not fail the run for them.
    return errs, s.get("warnings", [])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("results")
    p.add_argument("--tag", default="", help="only verify runs whose id contains this")
    a = p.parse_args()

    root = Path(a.results)
    dirs = [root] if (root / "summary.json").exists() else sorted(
        d for d in root.iterdir() if d.is_dir() and (d / "summary.json").exists()
        and (not a.tag or a.tag in d.name))
    if not dirs:
        print(f"[verify] no runs under {root} matching tag={a.tag!r}", file=sys.stderr)
        return 1
    bad = 0
    warn_total = 0
    for d in dirs:
        errs, warns = check_run(d)
        warn_total += len(warns)
        if errs:
            bad += 1
            print(f"[verify] FAIL {d.name}")
            for e in errs:
                print(f"    - {e}")
        else:
            s = json.loads((d / "summary.json").read_text())
            print(f"[verify] OK   {d.name}  "
                  f"rps={s['throughput']['requests_per_sec']} "
                  f"ttft_p99={s['latency_ms']['ttft']['p99']:.0f}ms "
                  f"tpot_p99={s['latency_ms']['tpot']['p99']:.1f}ms "
                  f"err={s['error_rate']:.2%} "
                  f"samples={s['server']['samples']}"
                  + (f"  ⚠ {len(warns)} warning(s)" if warns else ""))
        for w in warns:
            print(f"    ⚠ {w}")
    print(f"[verify] {len(dirs) - bad}/{len(dirs)} runs valid, "
          f"{warn_total} advisory warning(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
