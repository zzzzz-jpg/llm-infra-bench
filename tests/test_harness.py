"""
Correctness tests for the benchmark harness.

A benchmark that is subtly wrong is worse than no benchmark: it produces
confident numbers. These tests pin the semantics that the whole project rests
on — how TTFT/TPOT are defined, that the load model actually delivers the load
it claims, that output length is really fixed, and that the Prometheus parser
understands both vLLM and SGLang naming.

    python3 -m pytest tests/ -q
"""
from __future__ import annotations

import asyncio
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import pytest

import loadgen
import compare
import capacity

ROOT = Path(__file__).resolve().parent.parent
MOCK = ROOT / "bench" / "mock_server.py"


# --------------------------------------------------------------------------- #
# fixtures: a real mock engine on a random port
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def server():
    port = 18000 + random.randint(0, 999)
    proc = subprocess.Popen(
        [sys.executable, str(MOCK), "--port", str(port), "--capacity", "6",
         "--ttft-ms", "60", "--tok-ms", "4", "--model", "test-model"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30
    import urllib.request
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/health", timeout=2):
                break
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError(f"mock died:\n{proc.stdout.read()}")
            time.sleep(0.25)
    else:
        proc.kill()
        raise RuntimeError("mock server did not start")
    yield base
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_loadgen(base: str, **over) -> dict:
    defaults = dict(
        base_url=base, model="test-model", endpoint="chat",
        input_len=64, output_len=16, force_output_len=True,
        num_requests=12, warmup_requests=2, mode="closed", max_concurrency=4,
        request_rate=5.0, metrics_endpoints=[f"{base}/metrics"],
        # the mock exposes both families on one port; the kind filter is what
        # keeps engine aliases from being fed by the GPU endpoint and vice versa
        gpu_metrics_endpoints=[f"{base}/metrics"],
        scrape_interval=0.3, out_dir="/tmp/loadgen-tests",
        tag=f"pytest-{time.time_ns()}", seed=7)
    defaults.update(over)          # caller wins, no kwarg collisions
    cfg = loadgen.Config(**defaults)
    rc = asyncio.run(loadgen.amain(cfg))
    assert rc == 0, f"loadgen exited {rc}"
    d = Path(cfg.out_dir) / cfg.tag
    # results dir is timestamped; find the newest
    runs = sorted(Path(cfg.out_dir).glob(f"*{cfg.tag}*"))
    d = runs[-1]
    return {"dir": d, "summary": json.loads((d / "summary.json").read_text()),
            "meta": json.loads((d / "meta.json").read_text()),
            "cfg": cfg}


# --------------------------------------------------------------------------- #
# 1. metric semantics
# --------------------------------------------------------------------------- #
def test_output_length_is_actually_fixed(server):
    """ignore_eos + max_tokens must pin the completion length, otherwise every
    comparison is confounded by output-length variance."""
    r = run_loadgen(server, num_requests=8, output_len=32)
    bench = [int(x["completion_tokens"]) for x in _csv(r["dir"])
             if x["mode"] == "bench"]
    assert bench, "no bench rows"
    assert set(bench) == {32}, f"output length not fixed: {set(bench)}"
    assert r["summary"]["tokens"]["completion_tokens_mean"] == 32


def test_token_source_is_usage_not_chunk_count(server):
    """Counting SSE chunks under-counts tokens whenever the engine batches
    deltas. The run must report it used the real usage object."""
    r = run_loadgen(server, num_requests=6)
    assert r["summary"]["tokens"]["token_source"] == ["usage"]


def test_ttft_and_tpot_are_positive_and_ordered(server):
    r = run_loadgen(server, num_requests=10, output_len=32)
    lat = r["summary"]["latency_ms"]
    assert lat["ttft"]["mean"] > 0
    assert lat["tpot"]["mean"] > 0
    assert lat["ttft"]["p99"] >= lat["ttft"]["p50"] >= lat["ttft"]["min"]
    # E2E must be at least TTFT + (n_out-1)*TPOT for the median request
    e2e_p50 = lat["e2e"]["p50"]
    assert e2e_p50 > lat["ttft"]["p50"]


def test_ttft_never_exceeds_e2e(server):
    rows = [x for x in _csv(run_loadgen(server, num_requests=8)["dir"])
            if x["mode"] == "bench" and x["status"] == "ok"]
    for x in rows:
        assert float(x["ttft_ms"]) <= float(x["e2e_ms"]) + 1e-6


def test_itl_count_matches_output_tokens(server):
    """n output tokens must produce n-1 inter-token gaps. If this breaks, the
    TPOT denominator is wrong and every decode number is off by a factor."""
    rows = [x for x in _csv(run_loadgen(server, num_requests=6,
                                        output_len=24)["dir"])
            if x["mode"] == "bench" and x["status"] == "ok"]
    for x in rows:
        itl = json.loads(x["itl_ms"])
        assert len(itl) == int(x["completion_tokens"]) - 1, \
            f"{len(itl)} gaps for {x['completion_tokens']} tokens"


# --------------------------------------------------------------------------- #
# 2. load model fidelity
# --------------------------------------------------------------------------- #
def test_closed_loop_never_exceeds_concurrency(server):
    """The whole point of a closed-loop sweep is that in-flight == N."""
    r = run_loadgen(server, mode="closed", max_concurrency=4, num_requests=20,
                    output_len=32)
    rows = [x for x in _csv(r["dir"]) if x["mode"] == "bench"]
    events = []
    for x in rows:
        events.append((float(x["start_ts"]), 1))
        events.append((float(x["end_ts"]), -1))
    events.sort()
    cur = peak = 0
    for _, d in events:
        cur += d
        peak = max(peak, cur)
    assert peak <= 4, f"concurrency violated: peak in-flight {peak}"
    assert peak >= 3, f"load generator never reached the requested concurrency ({peak})"


def test_open_loop_delivers_requested_rate_when_not_saturated(server):
    """At a rate far below the knee, achieved rps must track the requested rps.
    If it does not, the arrival process is broken and every 'X rps' claim is
    fiction. Uses the deterministic (uniform) arrival process so the assertion
    is exact rather than statistical."""
    r = run_loadgen(server, mode="open", request_rate=8.0, num_requests=24,
                    max_concurrency=32, output_len=16, arrival_process="uniform")
    ach = r["summary"]["throughput"]["requests_per_sec"]
    assert 7.0 <= ach <= 9.0, f"achieved {ach} rps vs requested 8 (uniform arrivals)"
    assert r["summary"]["throughput"]["attempted_rps"] == 8.0


def test_poisson_arrivals_are_memoryless_not_deterministic(server):
    """Poisson mode must actually be bursty: mean gap == 1/rate but variance of
    gaps == variance of an exponential. A 'Poisson' generator that emits fixed
    gaps understates tail latency and hides real queueing behaviour."""
    r = run_loadgen(server, mode="open", request_rate=10.0, num_requests=40,
                    max_concurrency=48, output_len=8, arrival_process="poisson")
    rows = [x for x in _csv(r["dir"]) if x["mode"] == "bench"]
    gaps = sorted(float(x["arrival_ts"]) for x in rows)
    diffs = [b - a for a, b in zip(gaps, gaps[1:])]
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
    assert mean == pytest.approx(0.1, rel=0.35), f"mean gap {mean} != 1/rate"
    # for an exponential, variance == mean^2; for fixed spacing variance == 0
    assert var > mean ** 2 * 0.25, f"gaps look deterministic (var={var}, mean={mean})"


def test_open_loop_queueing_is_visible_when_over_capacity(server):
    """Push past the mock's capacity (6): achieved rps must clamp and the
    server-side waiting gauge must have been non-zero at some point. This is
    the test that proves we can actually see saturation."""
    r = run_loadgen(server, mode="open", request_rate=40.0, num_requests=40,
                    max_concurrency=32, output_len=24)
    s = r["summary"]
    assert s["throughput"]["requests_per_sec"] < 40.0 * 0.8
    wait = s["server"]["series"].get("requests_waiting")
    assert wait and wait["max"] > 0, f"queue never observed: {s['server']['series'].keys()}"
    assert s["latency_ms"]["ttft"]["p99"] > s["latency_ms"]["ttft"]["min"] * 2


def test_warmup_requests_are_excluded_from_statistics(server):
    r = run_loadgen(server, num_requests=10, warmup_requests=3)
    rows = _csv(r["dir"])
    assert sum(1 for x in rows if x["mode"] == "warmup") == 3
    assert sum(1 for x in rows if x["mode"] == "bench") == 10
    assert r["summary"]["requests_total"] == 10


def test_seed_makes_prompts_reproducible(server):
    """Same seed => byte-identical prompt stream, in the same row order. Without
    this, two 'identical' runs differ by prompt content and the comparison is
    measuring prompt luck."""
    a = run_loadgen(server, num_requests=6, seed=99)
    b = run_loadgen(server, num_requests=6, seed=99)
    pa = [x["prompt_tokens"] for x in _csv(a["dir"]) if x["mode"] == "bench"]
    pb = [x["prompt_tokens"] for x in _csv(b["dir"]) if x["mode"] == "bench"]
    assert pa == pb, f"seeded prompts differ: {pa} vs {pb}"
    c = run_loadgen(server, num_requests=6, seed=123)
    pc = [x["prompt_tokens"] for x in _csv(c["dir"]) if x["mode"] == "bench"]
    assert pc != pa, "different seeds produced identical prompts"


def test_rows_are_in_deterministic_order(server):
    """Closed-loop workers finish out of order; requests.csv must still be
    sorted by arrival index or run-to-run diffs are meaningless."""
    r = run_loadgen(server, mode="closed", max_concurrency=6, num_requests=18)
    idx = [int(x["idx"]) for x in _csv(r["dir"]) if x["mode"] == "bench"]
    assert idx == sorted(idx) == list(range(18))


# --------------------------------------------------------------------------- #
# 3. server-side scraping
# --------------------------------------------------------------------------- #
def test_engine_metrics_are_scraped_and_summarised(server):
    r = run_loadgen(server, num_requests=12, output_len=24, scrape_interval=0.25)
    ser = r["summary"]["server"]["series"]
    assert r["summary"]["server"]["samples"] > 0
    assert r["summary"]["server"]["scrape_errors"] == 0
    for key in ("requests_running", "kv_cache_util", "gpu_util",
                "gpu_mem_used_mib", "sm_active", "prompt_tokens_total",
                "ttft_seconds_count", "gen_tokens_total_rate"):
        assert key in ser, f"missing {key}; got {sorted(ser)}"
    # kind filter must have kept the families separate: the engine endpoint
    # contributes no DCGM alias, the gpu endpoint contributes no vllm alias
    assert ser["gpu_util"]["mean"] >= 0
    ts = r["dir"] / "timeseries.csv"
    assert ts.exists() and len(ts.read_text().splitlines()) > 2


def test_dead_metrics_endpoint_is_disabled_not_fatal(server, tmp_path):
    """An exporter that is simply not deployed must not fail the run."""
    r = run_loadgen(server, num_requests=12, output_len=32, out_dir=str(tmp_path),
                    metrics_endpoints=[f"{server}/metrics", "http://127.0.0.1:1/metrics"],
                    scrape_interval=0.1, scrape_disable_after=2)
    dis = r["summary"]["server"]["disabled_endpoints"]
    assert dis and "127.0.0.1:1" in dis[0]["endpoint"]
    assert r["summary"]["server"]["samples"] > 0


def test_warm_prefix_cache_from_a_previous_run_is_flagged(server, tmp_path):
    """Regression test for a real measurement bug.

    The engine's prefix cache is process-global and survives across runs.
    loadgen uses a fixed seed, so run N+1 replays run N's prompt sequence and
    starts with an almost fully warm cache -- reporting ~100% hit ratio even
    though shared_prefix_tokens is 0. Without a warning, the control arm of a
    prefix-cache A/B silently measures nothing.
    """
    shared = dict(num_requests=8, output_len=16, shared_prefix_tokens=0,
                  seed=4242, out_dir=str(tmp_path), scrape_interval=0.2)
    first = run_loadgen(server, tag="cold", **shared)
    assert first["summary"]["server"]["window"].get(
        "prefix_cache_hit_ratio_window", 0) < 0.3, \
        "first run against a fresh server should be mostly cache misses"

    second = run_loadgen(server, tag="warm", **shared)
    win = second["summary"]["server"]["window"]
    warns = second["summary"].get("warnings", [])
    if win.get("prefix_cache_hit_ratio_window", 0) > 0.3:
        assert any("prefix-cache hit ratio" in w for w in warns), \
            f"contaminated run was not flagged; warnings={warns}"


def test_validity_warnings_are_recorded_in_summary(server, tmp_path):
    """summary.json must carry the harness's own objections, so a report
    cannot quote a number without the caveat attached to it."""
    r = run_loadgen(server, num_requests=6, out_dir=str(tmp_path),
                    input_len=8192, tokenizer="")
    warns = r["summary"]["warnings"]
    assert any("--tokenizer" in w or "tokenizer" in w for w in warns), warns


def test_prom_parser_handles_labels_counters_and_sglang_names():
    txt = '''# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="m",pod="p-1"} 7
vllm:num_requests_waiting{model_name="m",pod="p-2"} 3
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name="m"} 11
# TYPE sglang:token_usage gauge
sglang:token_usage{model_name="m"} 0.42
# TYPE DCGM_FI_DEV_GPU_UTIL gauge
DCGM_FI_DEV_GPU_UTIL{gpu="0",pod="p-1"} 88
DCGM_FI_DEV_GPU_UTIL{gpu="1",pod="p-1"} 76
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="m"} NaN
# TYPE some_unrelated_metric gauge
some_unrelated_metric 1
'''
    out = loadgen.parse_prom_text(txt)
    assert out["requests_waiting"] == 5.0          # mean across pods
    assert out["requests_running"] == 11.0         # sglang name mapped
    assert out["kv_cache_util"] == 0.42            # sglang token_usage mapped
    assert out["gpu_util"] == 82.0                 # mean across GPUs
    # the NaN-valued vllm:gpu_cache_usage_perc must be dropped, leaving the
    # sglang value as the only sample for that alias
    assert out["kv_cache_util"] == 0.42
    assert "some_unrelated_metric" not in out


def test_kind_filter_prevents_endpoint_clobbering():
    """Two exporters scraped into one row must not overwrite each other."""
    engine_txt = 'vllm:num_requests_running{m="a"} 9\nDCGM_FI_DEV_GPU_UTIL{gpu="0"} 1\n'
    gpu_txt = 'vllm:num_requests_running{m="a"} 0\nDCGM_FI_DEV_GPU_UTIL{gpu="0"} 77\n'
    e = loadgen.parse_prom_text(engine_txt, loadgen.KIND_PREFIX["engine"])
    g = loadgen.parse_prom_text(gpu_txt, loadgen.KIND_PREFIX["gpu"])
    assert e["requests_running"] == 9.0 and "gpu_util" not in e
    assert g["gpu_util"] == 77.0 and "requests_running" not in g


# --------------------------------------------------------------------------- #
# 4. statistics helpers
# --------------------------------------------------------------------------- #
def test_percentiles_match_reference():
    vals = list(range(1, 101))          # 1..100
    assert loadgen.pct(vals, 50) == pytest.approx(50.5, abs=0.5)
    assert loadgen.pct(vals, 99) == pytest.approx(99.0, abs=1.0)
    assert loadgen.pct([], 99) != loadgen.pct([], 99)   # NaN for empty


def test_percentiles_ignore_nan():
    vals = [1.0, float("nan"), 2.0, 3.0]
    s = loadgen.summarize(vals)
    assert s["count"] == 3
    assert s["mean"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# 5. analysis tools
# --------------------------------------------------------------------------- #
def test_compare_flags_non_comparable_runs(tmp_path):
    """The guard that stops someone comparing a 128-token run with a 2048-token
    run and calling it a quantization win."""
    mk = lambda name, out_len, rps, ttft: _fake_run(tmp_path / name, out_len, rps, ttft)
    a = mk("a", 256, 10.0, 100.0)
    b = mk("b", 2048, 12.0, 90.0)      # different output length!
    runs = [compare.load_run(a), compare.load_run(b)]
    warns = compare.env_guard(runs)
    assert any("output_len" in w for w in warns), warns
    report = compare.build_report(runs, 0, "t")
    assert "Comparability warnings" in report


def test_compare_delta_directions():
    assert "✅" in compare.delta(80, 100, "lower")     # latency improved
    assert "❌" in compare.delta(120, 100, "lower")    # latency regressed
    assert "✅" in compare.delta(1200, 1000, "higher")  # throughput improved
    assert compare.delta(None, 1, "lower") == ""


def test_capacity_finds_the_knee(tmp_path):
    """Synthetic sweep: SLO-compliant up to concurrency 16, then TTFT explodes.
    capacity.py must name 16 as the knee and size replicas from it."""
    for c, rps, ttft, tpot, kv, q in (
            (4, 2.0, 90, 20, 0.3, 0), (8, 3.8, 110, 24, 0.5, 0),
            (16, 6.0, 400, 40, 0.8, 0), (32, 6.1, 9000, 70, 0.99, 12),
            (64, 6.0, 25000, 95, 0.99, 40)):
        _fake_run(tmp_path / f"c{c}", 256, rps, ttft, tpot=tpot, kv=kv,
                  queue=q, concurrency=c, mode="closed")
    rc = subprocess.run([sys.executable, str(ROOT / "bench" / "capacity.py"),
                         str(tmp_path), "--target-rps", "20",
                         "--slo-ttft-p99-ms", "1500", "--slo-tpot-p99-ms", "50",
                         "--headroom", "0.3"],
                        capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    out = rc.stdout
    assert "concurrency=16" in out, out
    # ceil(20 rps * 1.3 headroom / 6.0 rps per replica) == 5
    assert "| **replicas required** | **5** |" in out, out
    assert "| **GPUs required** | **5** |" in out, out
    assert ("Approaching KV cache bound" in out or "KV cache bound" in out
            or "Latency knee" in out), out


def _csv(d: Path) -> list[dict]:
    import csv
    with open(d / "requests.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _fake_run(d: Path, out_len, rps, ttft, tpot=30.0, kv=0.5, queue=0.0,
              concurrency=8, mode="closed", in_len=1024):
    d.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": d.name, "wall_clock_sec": 100.0,
        "requests_total": 100, "requests_ok": 100, "requests_error": 0,
        "error_rate": 0.0, "errors_by_type": {},
        "throughput": {"requests_per_sec": rps, "attempted_rps": None,
                       "output_tokens_per_sec": rps * out_len,
                       "input_tokens_per_sec": rps * in_len,
                       "total_tokens_per_sec": rps * (in_len + out_len)},
        "latency_ms": {
            "ttft": {"count": 100, "mean": ttft * 0.8, "p50": ttft * 0.9,
                     "p90": ttft * 0.95, "p95": ttft * 0.97, "p99": ttft,
                     "min": ttft * 0.5, "max": ttft * 1.1, "stdev": 1.0},
            "tpot": {"count": 100, "mean": tpot * 0.9, "p50": tpot * 0.95,
                     "p90": tpot, "p95": tpot, "p99": tpot, "min": tpot * 0.8,
                     "max": tpot * 1.2, "stdev": 1.0},
            "e2e": {"count": 100, "mean": 1000, "p50": 1000, "p90": 1200,
                    "p95": 1300, "p99": 1500, "min": 500, "max": 2000,
                    "stdev": 100},
            "itl": {"count": 100, "mean": tpot, "p50": tpot, "p90": tpot,
                    "p95": tpot, "p99": tpot, "min": tpot, "max": tpot,
                    "stdev": 0},
        },
        "tokens": {"prompt_tokens_mean": in_len, "completion_tokens_mean": out_len,
                   "completion_tokens_total": 100 * out_len,
                   "prompt_tokens_total": 100 * in_len, "token_source": ["usage"]},
        "slo": {"ttft_p99_ms": 2000, "tpot_p99_ms": 100,
                "goodput_rps": rps, "slo_attainment": 1.0},
        "server": {"samples": 10, "scrape_errors": 0, "series": {
            "requests_waiting": {"mean": queue, "p50": queue, "p99": queue,
                                 "max": queue, "min": queue, "count": 10},
            "kv_cache_util": {"mean": kv, "p50": kv, "p99": kv, "max": kv,
                              "min": kv, "count": 10},
            "gpu_util": {"mean": 70, "p50": 70, "p99": 80, "max": 80, "min": 60,
                         "count": 10}}},
    }
    meta = {"run_id": d.name, "prompt_builder_method": "synthetic",
            "config": {"input_len": in_len, "output_len": out_len, "mode": mode,
                       "request_rate": rps, "max_concurrency": concurrency,
                       "num_requests": 100, "endpoint": "chat",
                       "force_output_len": True, "tokenizer": "", "notes": ""},
            "host": {"platform": "test", "cpu_count": 8},
            "git_commit": "deadbee", "started_at": "2026-01-01T00:00:00+0000"}
    (d / "summary.json").write_text(json.dumps(summary))
    (d / "meta.json").write_text(json.dumps(meta))
    return d
