#!/usr/bin/env python3
"""
compare.py — Turn a set of loadgen runs into an A/B comparison report.

Usage
-----
    # compare everything under results/, baseline = the run whose tag matches
    python3 compare.py results --baseline-tag vllm-fp16 --out report.md

    # or name the runs explicitly
    python3 compare.py results/A results/B results/C --baseline results/A

Output: a markdown file with
  * an environment-diff guard (fails loudly if two runs are not comparable)
  * the headline metric table with deltas vs baseline
  * server-side (queue / KV cache / GPU) table
  * a verdict block per variable under test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HEADLINE: List[Tuple[str, str, str]] = [
    # (json path, column label, better direction: lower|higher|na)
    ("throughput.requests_per_sec",       "RPS (achieved)",     "higher"),
    ("throughput.output_tokens_per_sec",  "Out tok/s",          "higher"),
    ("throughput.total_tokens_per_sec",   "Total tok/s",        "higher"),
    ("latency_ms.ttft.mean",              "TTFT mean (ms)",     "lower"),
    ("latency_ms.ttft.p50",               "TTFT p50 (ms)",      "lower"),
    ("latency_ms.ttft.p99",               "TTFT p99 (ms)",      "lower"),
    ("latency_ms.tpot.mean",              "TPOT mean (ms)",     "lower"),
    ("latency_ms.tpot.p99",               "TPOT p99 (ms)",      "lower"),
    ("latency_ms.itl.p99",                "ITL p99 (ms)",       "lower"),
    ("latency_ms.e2e.p99",                "E2E p99 (ms)",       "lower"),
    ("slo.goodput_rps",                   "Goodput RPS",        "higher"),
    ("slo.slo_attainment",                "SLO attainment",     "higher"),
    ("error_rate",                        "Error rate",         "lower"),
    ("tokens.completion_tokens_mean",     "Out tok/req",        "na"),
    ("tokens.prompt_tokens_mean",         "In tok/req",         "na"),
]

SERVER: List[Tuple[str, str]] = [
    ("requests_waiting",  "Queue depth (waiting)"),
    ("requests_running",  "Running requests"),
    ("kv_cache_util",     "KV cache util"),
    ("prefix_cache_hit_rate", "Prefix-cache hit rate"),
    ("preemptions_total_rate", "Preemptions /s"),
    ("gen_throughput",      "Engine decode tok/s (SGLang gauge)"),
    ("gpu_util",          "GPU util %"),
    ("gpu_mem_used_mib",  "GPU mem used MiB"),
    ("sm_active",         "SM active"),
    ("tensor_active",     "Tensor pipe active"),
    ("power_w",           "Power W"),
    ("gpu_temp_c",        "GPU temp C"),
]


def dig(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


_TS_PREFIX = re.compile(r"^\d{8}-\d{6}-")


def short_label(run_id: str) -> str:
    """`20261008-043226-fp16-conc12` -> `fp16-conc12`.

    Full run ids make comparison tables unreadable; the timestamp is provenance,
    not identity, and it is preserved in the appendix and in --full-labels.
    """
    return _TS_PREFIX.sub("", run_id or "")


def load_run(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        meta_p = path / "meta.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    else:                                   # a bare summary.json
        summary = json.loads(path.read_text(encoding="utf-8"))
        meta = {}
    run_id = summary.get("run_id", Path(path).name)
    return {"dir": str(path), "summary": summary, "meta": meta,
            "run_id": run_id, "tag": short_label(run_id)}


def fmt(v: Any, label: str = "") -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != v:
            return "—"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) < 0.01 and v != 0:
            return f"{v:.4f}"
        return f"{v:,.2f}"
    return str(v)


def delta(cur: Any, base: Any, direction: str) -> str:
    if cur is None or base is None:
        return ""
    try:
        c, b = float(cur), float(base)
    except (TypeError, ValueError):
        return ""
    if b == 0 or c != c or b != b:
        return ""
    pct_ = (c - b) / abs(b) * 100.0
    arrow = "→"
    if direction in ("lower", "higher"):
        improved = (c < b) if direction == "lower" else (c > b)
        if abs(pct_) < 0.5:
            arrow = "="
        else:
            arrow = "✅" if improved else "❌"
    sign = "+" if pct_ >= 0 else ""
    return f"{sign}{pct_:.1f}% {arrow}"


def env_guard(runs: List[Dict[str, Any]]) -> List[str]:
    """Warn when two runs differ on a dimension that invalidates comparison."""
    warnings: List[str] = []
    keys = [("config.input_len", "input_len"), ("config.output_len", "output_len"),
            ("config.mode", "mode"), ("config.request_rate", "request_rate"),
            ("config.max_concurrency", "max_concurrency"),
            ("config.num_requests", "num_requests"),
            ("config.endpoint", "endpoint"),
            ("config.force_output_len", "force_output_len"),
            ("prompt_builder_method", "prompt method")]
    ref = runs[0]
    for path, label in keys:
        vals = {r["tag"]: dig(r["summary"], path) if path.startswith(("throughput", "latency", "error", "slo", "tokens")) else dig(r["meta"], path) for r in runs}
        uniq = {json.dumps(v, sort_keys=True, default=str) for v in vals.values()}
        if len(uniq) > 1:
            warnings.append(f"`{label}` differs across runs: "
                            + ", ".join(f"{k}={fmt(v)}" for k, v in vals.items()))
    gpus = {}
    for r in runs:
        g = dig(r["meta"], "host.gpus") or dig(r["meta"], "environment.gpus")
        gpus[r["tag"]] = json.dumps(g, sort_keys=True, default=str) if g else "unknown"
    if len(set(gpus.values())) > 1 and "unknown" not in set(gpus.values()):
        warnings.append("GPU inventory differs across runs — do not compare directly.")
    return warnings


WINDOW_ROWS: List[Tuple[str, str]] = [
    ("prefix_cache_hit_ratio_window", "Prefix-cache hit ratio (this run only)"),
    ("prefix_cache_hits_per_sec",     "Prefix-cache hits tok/s"),
    ("prompt_tokens_per_sec",         "Engine prefill tok/s"),
    ("gen_tokens_per_sec",            "Engine decode tok/s"),
    ("ttft_seconds_count",            "Requests seen by engine in window"),
    ("engine_ttft_mean_ms_window",    "Engine-side TTFT mean (ms)"),
    ("engine_tpot_mean_ms_window",    "Engine-side TPOT mean (ms)"),
    ("preemptions_total",             "Preemptions in window"),
]


def build_report(runs: List[Dict[str, Any]], baseline_idx: int,
                 title: str, full_labels: bool = False) -> str:
    base = runs[baseline_idx]
    if full_labels:
        for r in runs:
            r["tag"] = r["run_id"]
    out: List[str] = []
    out.append(f"# {title}\n")
    out.append(f"Baseline: **{base['tag']}**  ·  Runs compared: {len(runs)}\n")

    warns = env_guard(runs)
    if warns:
        out.append("> ⚠️ **Comparability warnings**\n>")
        for w in warns:
            out.append(f"> - {w}")
        out.append("")

    # ---- headline table
    out.append("## Headline metrics\n")
    hdr = "| Metric | " + " | ".join(r["tag"] for r in runs) + " |"
    sep = "|---" * (len(runs) + 1) + "|"
    out.append(hdr)
    out.append(sep)
    for path, label, direction in HEADLINE:
        cells = []
        b = dig(base["summary"], path)
        for r in runs:
            v = dig(r["summary"], path)
            if r is base:
                cells.append(fmt(v))
            else:
                cells.append(f"{fmt(v)}<br>`{delta(v, b, direction)}`")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out.append("")

    # ---- server-side table (mean [max])
    out.append("## Server-side signals during the run (mean, and p99 in brackets)\n")
    out.append("| Signal | " + " | ".join(r["tag"] for r in runs) + " |")
    out.append(sep)
    for key, label in SERVER:
        cells = []
        for r in runs:
            s = dig(r["summary"], f"server.series.{key}")
            if not s:
                cells.append("—")
            else:
                cells.append(f"{fmt(s.get('mean'))} [{fmt(s.get('p99'))}]")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out.append("")

    # ---- window-scoped counter deltas (A/B-safe) ----
    out.append("## Per-run counter deltas (window-scoped — use these for A/B)\n")
    out.append("> Engine gauges like `prefix_cache_hit_rate` are cumulative since "
               "process start, so two runs on the same server report nearly "
               "identical values. These rows are differenced over the "
               "measurement window instead.\n")
    out.append("| Signal | " + " | ".join(r["tag"] for r in runs) + " |")
    out.append(sep)
    for key, label in WINDOW_ROWS:
        cells = []
        for r in runs:
            v = (r["summary"].get("server") or {}).get("window", {}).get(key)
            cells.append(fmt(v))
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    cells = [fmt((r["summary"].get("server") or {}).get("window", {})
                 .get("window_seconds")) for r in runs]
    out.append(f"| Window length (s) | " + " | ".join(cells) + " |")
    out.append("")

    # ---- workload / env appendix
    out.append("## Fixed workload & environment (as recorded)\n")
    fields = ["config.input_len", "config.output_len", "config.mode",
              "config.request_rate", "config.max_concurrency",
              "config.num_requests", "config.endpoint", "config.force_output_len",
              "config.tokenizer", "prompt_builder_method", "config.notes"]
    out.append("| Field | " + " | ".join(r["tag"] for r in runs) + " |")
    out.append(sep)
    out.append(f"| full run id | " + " | ".join(f"`{r['run_id']}`" for r in runs) + " |")
    out.append(f"| path | " + " | ".join(f"`{r['dir']}`" for r in runs) + " |")
    for f in fields:
        cells = [fmt(dig(r["meta"], f) if not f.startswith("prompt") else r["meta"].get("prompt_builder_method")) for r in runs]
        out.append(f"| `{f}` | " + " | ".join(cells) + " |")
    for f, label in [("host.platform", "loadgen platform"),
                     ("host.cpu_count", "loadgen vCPU"),
                     ("git_commit", "repo commit"),
                     ("started_at", "started at")]:
        out.append(f"| {label} | " + " | ".join(fmt(dig(r['meta'], f)) for r in runs) + " |")
    gpu_rows = []
    for r in runs:
        g = dig(r["meta"], "environment.gpus") or dig(r["meta"], "host.gpus") or []
        gpu_rows.append("; ".join(f"{x.get('name', '?')} {x.get('memory_total', '')}"
                                  for x in g) if g else "—")
    out.append(f"| GPU | " + " | ".join(gpu_rows) + " |")
    out.append("")

    # ---- verdict scaffold
    out.append("## Verdict\n")
    out.append("_Fill in after reading the tables. Required shape:_\n")
    out.append("```text")
    out.append("Variable under test : <e.g. AWQ int4 vs FP16>")
    out.append("Throughput effect   : <out tok/s delta, with the CI/σ if repeats exist>")
    out.append("Latency effect      : <TTFT p99 / TPOT p99 delta>")
    out.append("Quality caveat      : <what the number does NOT tell you>")
    out.append("Decision            : <adopt / reject / adopt-with-condition>")
    out.append("```")
    return "\n".join(out) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run directories (or summary.json files)")
    p.add_argument("--baseline", help="run dir/tag to use as baseline")
    p.add_argument("--baseline-tag", help="pick baseline by substring of run id")
    p.add_argument("--title", default="Inference A/B comparison")
    p.add_argument("--out", default="-", help="output md path, '-' for stdout")
    p.add_argument("--sort", action="store_true", help="sort runs by mtime")
    p.add_argument("--full-labels", action="store_true",
                   help="use full run ids as column headers instead of short tags")
    a = p.parse_args()

    paths: List[Path] = []
    for arg in a.runs:
        pth = Path(arg)
        if pth.is_dir() and (pth / "summary.json").exists():
            paths.append(pth)
        elif pth.is_dir():
            paths += sorted(d for d in pth.iterdir()
                            if d.is_dir() and (d / "summary.json").exists())
        elif pth.exists():
            paths.append(pth)
        else:
            print(f"[compare] skip missing: {arg}", file=sys.stderr)
    if a.sort:
        paths.sort(key=lambda x: x.stat().st_mtime)
    if not paths:
        print("[compare] no runs found", file=sys.stderr)
        return 1

    runs = [load_run(x) for x in paths]
    idx = 0
    if a.baseline:
        for i, r in enumerate(runs):
            if a.baseline in r["dir"] or a.baseline == r["tag"]:
                idx = i
                break
    elif a.baseline_tag:
        for i, r in enumerate(runs):
            if a.baseline_tag in r["tag"]:
                idx = i
                break
    report = build_report(runs, idx, a.title, full_labels=a.full_labels)
    if a.out == "-":
        print(report)
    else:
        Path(a.out).write_text(report, encoding="utf-8")
        print(f"[compare] wrote {a.out} ({len(runs)} runs, baseline={runs[idx]['tag']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
