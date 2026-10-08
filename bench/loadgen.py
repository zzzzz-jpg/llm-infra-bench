#!/usr/bin/env python3
"""
loadgen.py — Fixed-load generator + client-side metrics collector for
OpenAI-compatible LLM inference endpoints (vLLM / SGLang / TGI / ...).

Design goals
------------
1. *Reproducible* load: input length, output length, concurrency and arrival
   rate are all pinned and recorded in the run manifest.
2. *Client-side truth*: TTFT / ITL / TPOT are measured from SSE chunk
   arrival timestamps on the load generator, never trusted from the server.
3. *Server-side context*: Prometheus /metrics of the engine (queue depth, KV
   cache, prefix-cache hit rate, preemptions) and of DCGM (GPU util, VRAM) are
   scraped on a timer during the run and stored as a time series.
4. *No heavy deps*: aiohttp + numpy + pyyaml. Percentile math is local.

Load models
-----------
* open-loop  (`mode: open`)   : Poisson arrivals at `request_rate` rps.
                                `max_concurrency` acts only as a safety valve.
                                This is the correct model for "can the service
                                keep up with X rps" questions.
* closed-loop(`mode: closed`) : N workers, each issuing the next request the
                                instant the previous one finishes. This is the
                                correct model for "what is peak throughput /
                                what happens at saturation" questions.

Usage
-----
    python3 loadgen.py --profile profiles/baseline.yaml --tag vllm-fp16
    python3 loadgen.py --base-url http://127.0.0.1:8000 --model Qwen2.5-7B-Instruct \
        --mode open --request-rate 4 --num-requests 200 \
        --input-len 1024 --output-len 256 --max-concurrency 64
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import platform
import random
import re
import socket
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # --- target ----------------------------------------------------------- #
    base_url: str = "http://127.0.0.1:8000"
    api_key: str = ""
    model: str = ""                       # empty -> ask /v1/models
    endpoint: str = "chat"                # chat | completions
    extra_body: Dict[str, Any] = field(default_factory=dict)

    # --- fixed workload --------------------------------------------------- #
    input_len: int = 1024                 # target prompt tokens
    output_len: int = 256                 # target completion tokens
    force_output_len: bool = True         # -> ignore_eos=true
    min_input_len: int = 0                # 0 -> uniform == input_len
    min_output_len: int = 0
    num_requests: int = 200               # 0 -> use duration_sec
    duration_sec: float = 0.0
    mode: str = "open"                    # open | closed
    request_rate: float = 2.0             # rps, open-loop only
    # poisson = memoryless arrivals, the right model for real user traffic.
    # uniform = exactly 1/rate spacing: lower variance, so it is the right
    #           model for reproducible regression runs and CI assertions.
    arrival_process: str = "poisson"      # poisson | uniform
    max_concurrency: int = 64             # in-flight cap / closed-loop workers
    warmup_requests: int = 8
    rampup_sec: float = 0.0
    seed: int = 1234

    # --- prompt construction --------------------------------------------- #
    prompt_style: str = "lorem"           # lorem | shared_prefix | file
    prompt_file: str = ""
    shared_prefix_tokens: int = 0         # >0 -> that many identical leading tokens
    tokenizer: str = ""                   # HF path/name; enables exact token counts
    chars_per_token: float = 3.6          # fallback heuristic (CJK ~1.5, EN ~4)

    # --- collection ------------------------------------------------------- #
    scrape_interval: float = 1.0
    scrape_disable_after: int = 5         # consecutive failures before disabling
    strict_kind_filter: bool = True       # engine endpoints only feed engine aliases
    metrics_endpoints: List[str] = field(default_factory=list)   # e.g. vllm /metrics
    gpu_metrics_endpoints: List[str] = field(default_factory=list)  # dcgm-exporter
    record_itl: bool = True
    timeout_sec: float = 600.0
    connect_timeout_sec: float = 30.0

    # --- output ----------------------------------------------------------- #
    out_dir: str = "results"
    tag: str = ""
    notes: str = ""

    @classmethod
    def load(cls, path: Optional[str], overrides: Dict[str, Any]) -> "Config":
        data: Dict[str, Any] = {}
        if path:
            if yaml is None:
                sys.exit("PyYAML is required to read profiles: pip install pyyaml")
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        flat = dict(data.get("loadgen", data))
        for k, v in overrides.items():
            if v is not None:
                flat[k] = v
        known = {f: flat[f] for f in cls.__dataclass_fields__ if f in flat}
        unknown = set(flat) - set(cls.__dataclass_fields__)
        if unknown:
            print(f"[loadgen] WARN unknown config keys ignored: {sorted(unknown)}",
                  file=sys.stderr)
        return cls(**known)


# --------------------------------------------------------------------------- #
# Prompt construction — deterministic, length-controlled
# --------------------------------------------------------------------------- #
LOREM = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua enim ad minim veniam quis nostrud "
    "exercitation ullamco laboris nisi aliquip ex ea commodo consequat duis aute "
    "irure in reprehenderit voluptate velit esse cillum fugiat nulla pariatur "
).split()

FILLER = (
    "system throughput latency memory bandwidth scheduling batching prefill decode "
    "attention kv cache quantization tensor parallel pipeline kernel occupancy "
).split()


class PromptBuilder:
    """Builds prompts with a controlled token length.

    Exactness strategy, best first:
      1. HF tokenizer available  -> encode/decode round-trip until len == target.
      2. token-id mode           -> raw random ids in a safe vocab range (exact).
      3. heuristic               -> chars_per_token estimate (documented as such).
    """

    def __init__(self, cfg: Config, rng: random.Random):
        self.cfg = cfg
        self.rng = rng
        self.tok = None
        self.vocab_size = 32000
        self.method = "heuristic"
        if cfg.tokenizer:
            try:
                from transformers import AutoTokenizer  # type: ignore
                self.tok = AutoTokenizer.from_pretrained(cfg.tokenizer,
                                                         trust_remote_code=True)
                self.vocab_size = int(getattr(self.tok, "vocab_size", 32000) or 32000)
                self.method = "tokenizer"
                print(f"[loadgen] tokenizer loaded: {cfg.tokenizer} "
                      f"(vocab={self.vocab_size})")
            except Exception as exc:  # pragma: no cover
                print(f"[loadgen] WARN tokenizer unavailable ({exc}); "
                      f"falling back to chars_per_token={cfg.chars_per_token}",
                      file=sys.stderr)
        self._shared_prefix = self._make_prefix(cfg.shared_prefix_tokens)
        self._validate(cfg)

    @staticmethod
    def _validate(cfg: Config) -> None:
        """Catch configuration that silently measures something else."""
        if cfg.prompt_style == "shared_prefix" and cfg.shared_prefix_tokens <= 0:
            print("[loadgen] WARN prompt_style=shared_prefix but "
                  "shared_prefix_tokens=0 — the prefix-cache experiment will "
                  "have no independent variable.", file=sys.stderr)
        if cfg.shared_prefix_tokens > 0 and cfg.shared_prefix_tokens >= cfg.input_len:
            print(f"[loadgen] WARN shared_prefix_tokens "
                  f"({cfg.shared_prefix_tokens}) >= input_len ({cfg.input_len}): "
                  f"every prompt is entirely prefix, so there is no per-request "
                  f"unique part and the KV/prefill measurement is degenerate.",
                  file=sys.stderr)
        if cfg.mode == "closed" and cfg.duration_sec > 0 and cfg.num_requests > 0:
            print("[loadgen] NOTE both duration_sec and num_requests set; the run "
                  "ends at whichever comes first.", file=sys.stderr)
        if cfg.mode == "open" and cfg.request_rate <= 0:
            print("[loadgen] WARN mode=open with request_rate<=0 sends nothing.",
                  file=sys.stderr)
        if not cfg.force_output_len and cfg.min_output_len == 0:
            print("[loadgen] NOTE force_output_len is off: output lengths will "
                  "vary, so TPOT/throughput comparisons across variants need "
                  "more samples. Correct for capacity runs, wrong for A/B.",
                  file=sys.stderr)
        if cfg.tokenizer == "" and cfg.input_len >= 4096:
            print(f"[loadgen] WARN input_len={cfg.input_len} with no tokenizer: "
                  f"the chars_per_token={cfg.chars_per_token} heuristic can be "
                  f"off by +-20%, which at this length is thousands of tokens. "
                  f"Pass --tokenizer for long-context runs.", file=sys.stderr)

    # -- helpers ---------------------------------------------------------- #
    def _words(self, n: int) -> str:
        return " ".join(self.rng.choice(LOREM) for _ in range(max(1, n)))

    def _make_prefix(self, n_tokens: int) -> str:
        if n_tokens <= 0:
            return ""
        if self.tok:
            ids = [self.rng.randrange(1000, min(self.vocab_size, 30000))
                   for _ in range(n_tokens)]
            return self.tok.decode(ids)
        return self._words(int(n_tokens * self.cfg.chars_per_token / 5))

    def token_ids(self, n: int) -> List[int]:
        """Exact-length prompt as token ids (OpenAI /v1/completions accepts this)."""
        return [self.rng.randrange(1000, min(self.vocab_size, 30000))
                for _ in range(max(1, n))]

    def _text_of_len(self, target: int) -> str:
        if self.tok:
            # binary-search-ish: overshoot then trim on token ids
            est = int(target * 1.15) + 8
            txt = self._words(est)
            ids = self.tok.encode(txt, add_special_tokens=False)
            if len(ids) < target:
                while len(ids) < target:
                    ids += self.tok.encode(self._words(target),
                                           add_special_tokens=False)
            ids = ids[:target]
            return self.tok.decode(ids)
        words = max(1, int(target * self.cfg.chars_per_token / 5.0))
        return self._words(words)

    # -- public ----------------------------------------------------------- #
    def build(self) -> Dict[str, Any]:
        cfg = self.cfg
        lo = cfg.min_input_len or cfg.input_len
        hi = max(lo, cfg.input_len)
        target_in = self.rng.randint(lo, hi)
        lo_o = cfg.min_output_len or cfg.output_len
        target_out = self.rng.randint(lo_o, max(lo_o, cfg.output_len))

        body: Dict[str, Any] = {"max_tokens": target_out}
        if cfg.force_output_len:
            body["ignore_eos"] = True

        if cfg.prompt_style == "file" and cfg.prompt_file:
            text = Path(cfg.prompt_file).read_text(encoding="utf-8")
            text = (text * (target_in // max(1, len(text) // 4) + 1))[: target_in * 4]
        elif self.tok is None and cfg.endpoint == "completions" and target_in > 0:
            body["prompt"] = self.token_ids(target_in)
            body["_prompt_tokens"] = target_in
            body["_target_output_tokens"] = target_out
            return body
        else:
            text = self._shared_prefix + " " + self._text_of_len(
                max(1, target_in - len(self._shared_prefix.split())))

        body["_target_input_tokens"] = target_in
        body["_target_output_tokens"] = target_out
        if cfg.endpoint == "chat":
            body["messages"] = [
                {"role": "system",
                 "content": "You are a helpful assistant. Answer concisely."},
                {"role": "user", "content": text.strip() +
                 "\n\nSummarize the above passage in detail."},
            ]
        else:
            body["prompt"] = text
        return body


# --------------------------------------------------------------------------- #
# Per-request result
# --------------------------------------------------------------------------- #
@dataclass
class RequestResult:
    idx: int
    request_id: str
    mode: str                    # warmup | bench
    arrival_ts: float            # epoch seconds, when the request was scheduled
    start_ts: float = 0.0        # epoch seconds, when the HTTP call began
    first_token_ts: float = 0.0
    end_ts: float = 0.0
    status: str = "ok"           # ok | error
    http_status: int = 0
    error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    token_source: str = ""       # usage | chunks | target
    sse_chunks: int = 0
    ttft_ms: float = float("nan")
    e2e_ms: float = float("nan")
    tpot_ms: float = float("nan")
    itl_ms: List[float] = field(default_factory=list)
    queue_wait_ms: float = float("nan")   # arrival -> socket send (client-side queueing)

    def finish(self) -> None:
        if self.first_token_ts and self.start_ts:
            self.ttft_ms = (self.first_token_ts - self.start_ts) * 1000.0
        if self.end_ts and self.start_ts:
            self.e2e_ms = (self.end_ts - self.start_ts) * 1000.0
        self.queue_wait_ms = ((self.start_ts - self.arrival_ts) * 1000.0
                              if self.start_ts else float("nan"))
        n = self.completion_tokens or len(self.itl_ms) + 1
        if n and n > 1 and self.first_token_ts and self.end_ts:
            self.tpot_ms = (self.end_ts - self.first_token_ts) * 1000.0 / (n - 1)


# --------------------------------------------------------------------------- #
# Prometheus text-format scraping (dependency-free)
# --------------------------------------------------------------------------- #
# Prometheus text format mandates a space before the value; we accept `\s*`
# anyway so hand-rolled exporters (and some proxies) still parse.
_METRIC_RE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
                        r"(?:\{(?P<labels>[^}]*)\})?\s*(?P<value>[-+0-9.eE]+|NaN|Inf)"
                        r"(?:\s+(?P<ts>[-+0-9.eE]+))?\s*$")

# alias -> regex over the raw metric name.
# Prefix-agnostic on purpose: vLLM exports `vllm:*`, SGLang exports `sglang:*`
# and spells several of them differently (num_running_reqs vs
# num_requests_running, token_usage vs gpu_cache_usage_perc, num_queue_reqs vs
# num_requests_waiting). Normalising here is what lets ONE harness compare the
# two engines without per-engine post-processing.
INTERESTING: Dict[str, str] = {
    "requests_running":      r"(vllm:num_requests_running|sglang:num_running_reqs)$",
    "requests_waiting":      r"(vllm:num_requests_waiting|sglang:(num_queue_reqs|num_waiting_reqs))$",
    "requests_swapped":      r"vllm:num_requests_swapped$",
    "kv_cache_util":         r"(vllm:gpu_cache_usage_perc|sglang:token_usage|sglang:kv_cache_usage)$",
    "prefix_cache_hit_rate": r"(vllm:(gpu_)?prefix_cache_hit(_rate)?|sglang:cache_hit_rate|sglang:prefix_cache_hit_rate)$",
    "preemptions_total":     r"(vllm:num_preemptions_total|sglang:num_preemptions_total)$",
    "prompt_tokens_total":   r"(vllm:prompt_tokens_total|sglang:prompt_tokens_total)$",
    # vLLM exposes a cumulative counter; SGLang does not — it exposes a gauge
    # that is ALREADY a rate. Differencing a gauge produces garbage, so they
    # get separate aliases and the gauge is never window-differenced.
    "gen_tokens_total":      r"(vllm|sglang):generation_tokens_total$",
    "gen_throughput":        r"sglang:gen_throughput$",
    "num_used_tokens":       r"sglang:num_used_tokens$",
    "prefix_cache_hits_total":    r"(vllm:(gpu_)?prefix_cache_hits(_total)?|sglang:cache_hit_tokens_total)$",
    "prefix_cache_queries_total": r"(vllm:(gpu_)?prefix_cache_queries(_total)?|sglang:cache_lookup_tokens_total)$",
    "ttft_seconds_sum":      r"(vllm|sglang):time_to_first_token_seconds_sum$",
    "ttft_seconds_count":    r"(vllm|sglang):time_to_first_token_seconds_count$",
    "tpot_seconds_sum":      r"(vllm|sglang):(time_per_output_token_seconds|inter_token_latency_seconds)_sum$",
    "tpot_seconds_count":    r"(vllm|sglang):(time_per_output_token_seconds|inter_token_latency_seconds)_count$",
    "e2e_seconds_sum":       r"(vllm|sglang):e2e_request_latency_seconds_sum$",
    "e2e_seconds_count":     r"(vllm|sglang):e2e_request_latency_seconds_count$",
    "gpu_util":              r"DCGM_FI_DEV_GPU_UTIL$",
    "gpu_mem_used_mib":      r"DCGM_FI_DEV_FB_USED$",
    "gpu_mem_free_mib":      r"DCGM_FI_DEV_FB_FREE$",
    "sm_active":             r"DCGM_FI_PROF_SM_ACTIVE$",
    "tensor_active":         r"DCGM_FI_PROF_PIPE_TENSOR_ACTIVE$",
    "dram_active":           r"DCGM_FI_PROF_DRAM_ACTIVE$",
    "power_w":               r"DCGM_FI_DEV_POWER_USAGE$",
    "gpu_temp_c":            r"DCGM_FI_DEV_GPU_TEMP$",
    "mem_clock_mhz":         r"DCGM_FI_DEV_MEM_CLOCK$",
    "node_mem_used_bytes":   r"node_memory_MemTotal_bytes$",
}
_INTERESTING_RE = {k: re.compile(v) for k, v in INTERESTING.items()}


# Which metric families an endpoint is allowed to contribute. Without this,
# two exporters scraped into one row silently overwrite each other's aliases
# (e.g. a node-level DCGM pod that also happens to expose engine names).
KIND_PREFIX = {
    "engine": re.compile(r"^(vllm|sglang|tgi|lmcache|text_generation|inference)"),
    "gpu":    re.compile(r"^(DCGM|nvidia|nv_|gpu_|amd_smi)"),
    "node":   re.compile(r"^(node_|container_|kube_|process_)"),
    "any":    re.compile(r"."),
}


def parse_prom_text(text: str, allowed: Optional[re.Pattern] = None) -> Dict[str, float]:
    """Return {alias: mean-value-across-labelsets} for the metrics we care about."""
    buckets: Dict[str, List[float]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        if allowed is not None and not allowed.match(name):
            continue
        try:
            val = float(m.group("value"))
        except ValueError:
            continue
        if val != val:      # NaN: drop it here, not downstream, or one NaN
            continue        # sample poisons the mean for that alias
        for alias, rx in _INTERESTING_RE.items():
            if rx.match(name):
                buckets.setdefault(alias, []).append(val)
                break
    return {k: (sum(v) / len(v) if v else float("nan")) for k, v in buckets.items()}


class MetricsSampler:
    """Background task scraping engine + GPU Prometheus endpoints on a timer."""

    def __init__(self, cfg: Config, session: aiohttp.ClientSession):
        self.cfg = cfg
        self.session = session
        self.rows: List[Dict[str, Any]] = {}  # type: ignore[assignment]
        self.rows = []
        self.counters_prev: Dict[str, Tuple[float, float]] = {}
        self.disabled_endpoints: List[Dict[str, str]] = []
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self.errors = 0

    def start(self) -> None:
        eps = [(e, "engine") for e in self.cfg.metrics_endpoints] + \
              [(e, "gpu") for e in self.cfg.gpu_metrics_endpoints]
        if not eps:
            return
        self._task = asyncio.create_task(self._run(eps))

    async def _run(self, eps: List[Tuple[str, str]]) -> None:
        fails: Dict[str, int] = {u: 0 for u, _ in eps}
        dead: set = set()
        while not self._stop.is_set():
            t0 = time.time()
            row: Dict[str, Any] = {"ts": round(t0, 3)}
            for url, kind in eps:
                if url in dead:
                    continue
                allowed = (KIND_PREFIX.get(kind, KIND_PREFIX["any"])
                           if self.cfg.strict_kind_filter else KIND_PREFIX["any"])
                try:
                    async with self.session.get(
                            url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                        if r.status != 200:
                            raise RuntimeError(f"http {r.status}")
                        parsed = parse_prom_text(await r.text(), allowed)
                    fails[url] = 0
                except Exception as exc:
                    fails[url] += 1
                    self.errors += 1
                    # circuit breaker: an exporter that is simply not deployed
                    # must not spam the log or inflate scrape_errors forever
                    if fails[url] == max(1, self.cfg.scrape_disable_after):
                        dead.add(url)
                        self.disabled_endpoints.append(
                            {"endpoint": url, "kind": kind,
                             "reason": f"{self.cfg.scrape_disable_after} "
                                       f"consecutive failures: {exc}"})
                        print(f"[loadgen] WARN metrics endpoint disabled "
                              f"(unreachable): {url}", file=sys.stderr)
                    continue

                for alias, val in parsed.items():
                    if val != val:      # NaN
                        continue
                    if alias.endswith(("_total", "_count", "_sum")):
                        key = f"{kind}:{alias}"
                        prev = self.counters_prev.get(key)
                        if prev is not None:
                            pv, pt = prev
                            dt = t0 - pt
                            if val >= pv and dt > 1e-6:
                                row[f"{alias}_rate"] = round((val - pv) / dt, 4)
                        self.counters_prev[key] = (val, t0)
                    row[alias] = round(val, 6)
            if len(row) > 1:
                self.rows.append(row)
            await asyncio.sleep(max(0.05, self.cfg.scrape_interval - (time.time() - t0)))

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=6)
            except asyncio.TimeoutError:
                self._task.cancel()

    def window_deltas(self) -> Dict[str, float]:
        """Counter deltas over the measurement window.

        Engine gauges like `vllm:gpu_prefix_cache_hit_rate` are cumulative since
        process start, so two back-to-back runs on the same server report almost
        identical values and the A/B signal disappears. Differencing the
        underlying counters over the run window gives the per-run truth.
        """
        out: Dict[str, float] = {}
        if len(self.rows) < 2:
            return out
        first, last = self.rows[0], self.rows[-1]
        # all three suffixes are counters in Prometheus convention
        for k in sorted(set(first) & set(last)):
            if not k.endswith(("_total", "_count", "_sum")):
                continue
            try:
                d = float(last[k]) - float(first[k])
            except (TypeError, ValueError):
                continue
            if d >= 0:
                out[k] = round(d, 4)
        hits = out.get("prefix_cache_hits_total")
        queries = out.get("prefix_cache_queries_total")
        if hits is not None and queries:
            out["prefix_cache_hit_ratio_window"] = round(hits / queries, 4)
        span = float(last.get("ts", 0)) - float(first.get("ts", 0))
        if span > 0:
            out["window_seconds"] = round(span, 3)
            for k in list(out):
                if k.endswith(("_total", "_count")):
                    out[k.rsplit("_", 1)[0] + "_per_sec"] = round(out[k] / span, 4)
        # engine-side means over the window: directly comparable to the
        # client-side numbers, and the difference localises gateway/network cost
        for phase, sum_k, cnt_k in (("ttft", "ttft_seconds_sum", "ttft_seconds_count"),
                                    ("tpot", "tpot_seconds_sum", "tpot_seconds_count"),
                                    ("e2e", "e2e_seconds_sum", "e2e_seconds_count")):
            if out.get(cnt_k) and sum_k in out:
                out[f"engine_{phase}_mean_ms_window"] = round(
                    out[sum_k] / out[cnt_k] * 1000.0, 3)
        return out

    def write(self, path: Path) -> None:
        if not self.rows:
            return
        cols: List[str] = ["ts"]
        for r in self.rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)


# --------------------------------------------------------------------------- #
# The client
# --------------------------------------------------------------------------- #
def build_payload(builder: PromptBuilder, cfg: Config) -> Dict[str, Any]:
    raw = builder.build()
    target_in = raw.pop("_target_input_tokens", None)
    target_out = raw.pop("_target_output_tokens", None)
    payload: Dict[str, Any] = {
        "model": cfg.model,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        **raw,
    }
    payload.update(cfg.extra_body)
    if target_in is not None:
        payload["_target_input_tokens"] = target_in
    payload["_target_output_tokens"] = target_out
    return payload


async def send_request(session: aiohttp.ClientSession, cfg: Config,
                       builder: PromptBuilder, sem: asyncio.Semaphore,
                       idx: int, mode: str) -> RequestResult:
    res = RequestResult(idx=idx, request_id=uuid.uuid4().hex[:12], mode=mode,
                        arrival_ts=time.time())
    payload = build_payload(builder, cfg)
    target_in = payload.pop("_target_input_tokens", None)
    target_out = payload.pop("_target_output_tokens", None)
    url = f"{cfg.base_url.rstrip('/')}/v1/" + (
        "chat/completions" if cfg.endpoint == "chat" else "completions")
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    timeout = aiohttp.ClientTimeout(total=cfg.timeout_sec,
                                    connect=cfg.connect_timeout_sec)

    async with sem:
        res.start_ts = time.time()
        chunk_times: List[float] = []
        try:
            async with session.post(url, json=payload, headers=headers,
                                    timeout=timeout) as resp:
                res.http_status = resp.status
                if resp.status != 200:
                    res.status = "error"
                    res.error = (await resp.text())[:500]
                    res.end_ts = time.time()
                    res.finish()
                    return res
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    now = time.time()
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    usage = obj.get("usage")
                    if usage:
                        res.prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                        res.completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                        res.token_source = "usage"
                    text = ""
                    if cfg.endpoint == "chat":
                        for ch in obj.get("choices", []):
                            delta = ch.get("delta") or {}
                            text += delta.get("content") or ""
                    else:
                        for ch in obj.get("choices", []):
                            text += ch.get("text") or ""
                    if text:
                        res.sse_chunks += 1
                        chunk_times.append(now)
                        if res.first_token_ts == 0.0:
                            res.first_token_ts = now
                res.end_ts = time.time()
        except asyncio.TimeoutError:
            res.status, res.error = "error", "timeout"
            res.end_ts = time.time()
        except Exception as exc:
            res.status, res.error = "error", f"{type(exc).__name__}: {exc}"
            res.end_ts = time.time()

    if len(chunk_times) > 1:
        res.itl_ms = [round((chunk_times[i] - chunk_times[i - 1]) * 1000.0, 4)
                      for i in range(1, len(chunk_times))]
    if not res.completion_tokens:
        if target_out:
            res.completion_tokens = int(target_out)
            res.token_source = res.token_source or "target"
        else:
            res.completion_tokens = res.sse_chunks
            res.token_source = "chunks"
    if not res.prompt_tokens and target_in:
        res.prompt_tokens = int(target_in)
        res.token_source = res.token_source or "target"
    res.finish()
    return res


# --------------------------------------------------------------------------- #
# Load drivers
# --------------------------------------------------------------------------- #
async def run_open_loop(cfg: Config, session: aiohttp.ClientSession,
                        builder: PromptBuilder, sem: asyncio.Semaphore,
                        mode: str) -> List[RequestResult]:
    rng = random.Random(cfg.seed)
    tasks: List[asyncio.Task] = []
    t_start = time.time()
    idx = 0
    deadline = t_start + cfg.duration_sec if cfg.duration_sec > 0 else float("inf")
    limit = cfg.num_requests if cfg.num_requests > 0 else 10 ** 9
    while len(tasks) < limit and time.time() < deadline:
        # non-homogeneous Poisson: linear ramp-up of the rate
        rate = cfg.request_rate
        if cfg.rampup_sec > 0:
            frac = min(1.0, (time.time() - t_start) / cfg.rampup_sec)
            rate = cfg.request_rate * (0.1 + 0.9 * frac)
        if rate <= 0:
            break
        gap = (1.0 / rate if cfg.arrival_process == "uniform"
               else rng.expovariate(rate))
        await asyncio.sleep(gap)
        tasks.append(asyncio.create_task(
            send_request(session, cfg, builder, sem, idx, mode)))
        idx += 1
    return await asyncio.gather(*tasks, return_exceptions=False)


async def _closed_worker(cfg: Config, session: aiohttp.ClientSession,
                         builder: PromptBuilder, sem: asyncio.Semaphore,
                         out: List[RequestResult], counter: List[int],
                         lock: asyncio.Lock, mode: str, deadline: float) -> None:
    while True:
        async with lock:
            if counter[0] >= (cfg.num_requests or 10 ** 9) or time.time() >= deadline:
                return
            i = counter[0]
            counter[0] += 1
        out.append(await send_request(session, cfg, builder, sem, i, mode))


async def run_closed_loop(cfg: Config, session: aiohttp.ClientSession,
                          builder: PromptBuilder, sem: asyncio.Semaphore,
                          mode: str) -> List[RequestResult]:
    out: List[RequestResult] = []
    counter = [0]
    lock = asyncio.Lock()
    deadline = (time.time() + cfg.duration_sec) if cfg.duration_sec > 0 else float("inf")
    workers = [asyncio.create_task(_closed_worker(cfg, session, builder, sem, out,
                                                  counter, lock, mode, deadline))
               for _ in range(cfg.max_concurrency)]
    await asyncio.gather(*workers)
    return out


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def pct(vals: List[float], q: float) -> float:
    vals = [v for v in vals if v == v and v is not None]
    if not vals:
        return float("nan")
    if np is not None:
        return float(np.percentile(np.asarray(vals, dtype=float), q))
    s = sorted(vals)
    k = (len(s) - 1) * q / 100.0
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return float(s[f] + (s[c] - s[f]) * (k - f))


def summarize(vals: List[float]) -> Dict[str, float]:
    vals = [v for v in vals if v == v]
    if not vals:
        return {"count": 0}
    return {
        "count": len(vals),
        "mean": round(sum(vals) / len(vals), 3),
        "min": round(min(vals), 3),
        "p50": round(pct(vals, 50), 3),
        "p90": round(pct(vals, 90), 3),
        "p95": round(pct(vals, 95), 3),
        "p99": round(pct(vals, 99), 3),
        "max": round(max(vals), 3),
        "stdev": round(float(np.std(vals)), 3) if np is not None and len(vals) > 1 else 0.0,
    }


# Aliases that MUST appear if the engine is a version we understand.
EXPECTED_ALIASES = ("requests_running", "requests_waiting", "kv_cache_util")


def keys_of_interest(rows: List[Dict[str, Any]]) -> List[str]:
    """Which of the core engine aliases did we actually see?"""
    seen: set = set()
    for r in rows:
        seen |= {k for k in r if k != "ts"}
    return [a for a in EXPECTED_ALIASES if a in seen]


def ts_summary(rows: List[Dict[str, Any]], keys: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and r[k] is not None]
        if vals:
            out[k] = summarize(vals)
    return out


def host_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "loadgen_version": VERSION,
    }
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    info["mem_total_kib"] = int(line.split()[1])
                    break
    except OSError:
        pass
    try:
        out = os.popen("nvidia-smi --query-gpu=index,name,memory.total,driver_version "
                       "--format=csv,noheader 2>/dev/null").read().strip()
        if out:
            info["gpus"] = [dict(zip(["index", "name", "memory_total", "driver"],
                                     [c.strip() for c in l.split(",")]))
                            for l in out.splitlines()]
    except Exception:
        pass
    return info


async def resolve_model(cfg: Config, session: aiohttp.ClientSession) -> None:
    if cfg.model:
        return
    try:
        async with session.get(f"{cfg.base_url.rstrip('/')}/v1/models",
                               timeout=aiohttp.ClientTimeout(total=15)) as r:
            data = await r.json()
            ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
            if ids:
                cfg.model = ids[0]
                print(f"[loadgen] auto-detected model: {cfg.model}")
    except Exception as exc:
        print(f"[loadgen] WARN could not auto-detect model ({exc}); "
              f"pass --model", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
async def amain(cfg: Config) -> int:
    rng = random.Random(cfg.seed)
    builder = PromptBuilder(cfg, rng)
    sem = asyncio.Semaphore(cfg.max_concurrency)
    run_id = time.strftime("%Y%m%d-%H%M%S") + (f"-{cfg.tag}" if cfg.tag else "")
    out = Path(cfg.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)

    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300,
                                     force_close=False)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=cfg.connect_timeout_sec)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        await resolve_model(cfg, session)

        # connectivity pre-flight
        try:
            async with session.get(f"{cfg.base_url.rstrip('/')}/health",
                                   timeout=aiohttp.ClientTimeout(total=10)) as r:
                print(f"[loadgen] /health -> {r.status}")
        except Exception:
            print("[loadgen] WARN /health unreachable (continuing)", file=sys.stderr)

        sampler = MetricsSampler(cfg, session)
        warm_rows: List[RequestResult] = []   # kept in requests.csv as an audit
                                              # trail, excluded from statistics

        if cfg.warmup_requests > 0:
            print(f"[loadgen] warmup: {cfg.warmup_requests} requests "
                  f"@ concurrency {min(cfg.max_concurrency, cfg.warmup_requests)}")
            warm = await asyncio.gather(*[
                send_request(session, cfg, builder, sem, -1 - i, "warmup")
                for i in range(cfg.warmup_requests)])
            warm_rows = list(warm)
            n_ok = sum(1 for w in warm if w.status == "ok")
            print(f"[loadgen] warmup ok={n_ok}/{len(warm)}")
            if n_ok == 0:
                print("[loadgen] FATAL warmup produced zero successful requests; "
                      "aborting so the run is not recorded as data.", file=sys.stderr)
                for w in warm[:3]:
                    print(f"  http={w.http_status} err={w.error[:200]}", file=sys.stderr)
                return 2

        sampler.start()
        t0 = time.time()
        print(f"[loadgen] START mode={cfg.mode} rate={cfg.request_rate}rps "
              f"concurrency={cfg.max_concurrency} n={cfg.num_requests} "
              f"dur={cfg.duration_sec or 'n/a'} in={cfg.input_len} out={cfg.output_len}")
        driver = run_open_loop if cfg.mode == "open" else run_closed_loop
        bench_results = await driver(cfg, session, builder, sem, "bench")
        wall = time.time() - t0
        # deterministic row order: warmups first, then bench requests by
        # arrival index. Closed-loop workers finish out of order, so without
        # this the CSV row order (and anything diffed against it) is not
        # reproducible between identical runs.
        bench_results.sort(key=lambda r: r.idx)
        warm_rows.sort(key=lambda r: -r.idx)
        results = warm_rows + bench_results
        await sampler.stop()

    # ---------------- aggregate ---------------------------------------- #
    ok = [r for r in bench_results if r.status == "ok"]
    err = [r for r in bench_results if r.status != "ok"]
    all_itl = [v for r in ok for v in r.itl_ms] if cfg.record_itl else []
    out_toks = sum(r.completion_tokens for r in ok)
    in_toks = sum(r.prompt_tokens for r in ok)

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "wall_clock_sec": round(wall, 3),
        "requests_total": len(bench_results),
        "requests_ok": len(ok),
        "requests_error": len(err),
        "error_rate": round(len(err) / max(1, len(bench_results)), 5),
        "errors_by_type": {},
        "throughput": {
            "requests_per_sec": round(len(ok) / wall, 4) if wall else 0,
            "attempted_rps": round(cfg.request_rate, 4) if cfg.mode == "open" else None,
            "output_tokens_per_sec": round(out_toks / wall, 2) if wall else 0,
            "input_tokens_per_sec": round(in_toks / wall, 2) if wall else 0,
            "total_tokens_per_sec": round((out_toks + in_toks) / wall, 2) if wall else 0,
        },
        "latency_ms": {
            "ttft": summarize([r.ttft_ms for r in ok]),
            "tpot": summarize([r.tpot_ms for r in ok]),
            "e2e": summarize([r.e2e_ms for r in ok]),
            "itl": summarize(all_itl),
            "client_queue_wait": summarize([r.queue_wait_ms for r in ok]),
        },
        "tokens": {
            "prompt_tokens_mean": round(in_toks / max(1, len(ok)), 1),
            "completion_tokens_mean": round(out_toks / max(1, len(ok)), 1),
            "completion_tokens_total": out_toks,
            "prompt_tokens_total": in_toks,
            "token_source": sorted({r.token_source for r in ok}),
        },
    }
    for r in err:
        key = r.error.split(":")[0][:60] or f"http_{r.http_status}"
        summary["errors_by_type"][key] = summary["errors_by_type"].get(key, 0) + 1

    # SLO / goodput
    slo = {"ttft_p99_ms": 2000.0, "tpot_p99_ms": 100.0}
    if slo:
        l = summary["latency_ms"]
        good = sum(1 for r in ok
                   if (r.ttft_ms == r.ttft_ms and r.ttft_ms <= slo["ttft_p99_ms"])
                   and (r.tpot_ms == r.tpot_ms and r.tpot_ms <= slo["tpot_p99_ms"]))
        summary["slo"] = {**slo, "goodput_rps": round(good / wall, 4) if wall else 0,
                          "slo_attainment": round(good / max(1, len(bench_results)), 5)}

    # ---------------- validity checks ---------------------------------- #
    # A benchmark should argue with you. Each of these fires when the data
    # says the run measured something other than what was intended.
    warnings: List[str] = []

    cq = summary["latency_ms"]["client_queue_wait"]
    if cq.get("p99", 0) and cq["p99"] > 50:
        warnings.append(
            f"load generator may be the bottleneck: client_queue_wait p99 = "
            f"{cq['p99']:.0f} ms (time between 'should have been sent' and "
            f"'socket opened'). Shard the load generator or reduce concurrency.")

    if summary["tokens"]["token_source"] and             summary["tokens"]["token_source"] != ["usage"]:
        warnings.append(
            f"token counts came from {summary['tokens']['token_source']}, not "
            f"from the API usage object. Throughput in tok/s is an estimate; "
            f"pass stream_options.include_usage (the harness does) or check "
            f"that the engine supports it.")

    chunk_ratio = (sum(r.sse_chunks for r in ok) / max(1, out_toks)) if ok else 0
    if ok and chunk_ratio < 0.5:
        warnings.append(
            f"only {chunk_ratio:.2f} SSE chunks per token — a proxy is very "
            f"likely buffering the stream (nginx proxy_buffering on). "
            f"TTFT measured here is not what users experience.")

    if sampler.rows and not keys_of_interest(sampler.rows):
        warnings.append(
            "server-side metrics were scraped but no known engine/GPU series "
            "matched. Metric names may have changed in your engine version: "
            "extend loadgen.INTERESTING.")

    if not sampler.rows and (cfg.metrics_endpoints or cfg.gpu_metrics_endpoints):
        warnings.append(
            "no server-side samples at all: queue/KV/GPU columns are empty, so "
            "this run cannot be diagnosed if the numbers look odd.")

    hit = sampler.window_deltas().get("prefix_cache_hit_ratio_window")
    if hit is not None and hit > 0.3 and cfg.shared_prefix_tokens <= 0:
        warnings.append(
            f"prefix-cache hit ratio was {hit:.2f} although "
            f"shared_prefix_tokens=0. The cache was warm from a PREVIOUS run "
            f"(it is process-global and the prompt sequence is seeded). "
            f"Flush it (scripts/flush_cache.sh) or restart the engine; "
            f"otherwise control arms are contaminated.")

    if cfg.mode == "open" and cfg.request_rate > 0:
        achieved = summary["throughput"]["requests_per_sec"]
        if achieved < cfg.request_rate * 0.9:
            warnings.append(
                f"achieved {achieved:.2f} rps vs offered {cfg.request_rate:.2f} "
                f"rps: the service is past its knee and queueing. This is a "
                f"valid saturation measurement, but do NOT report "
                f"'{cfg.request_rate} rps' as a supported rate.")

    if cfg.tokenizer == "" and cfg.input_len >= 4096:
        warnings.append(
            f"input_len={cfg.input_len} without --tokenizer: actual prompt "
            f"length is a chars/token estimate and may be off by +-20%.")

    summary["warnings"] = warnings
    for w in warnings:
        print(f"[loadgen] WARN {w}", file=sys.stderr)

    # server-side time series summary
    keys = sorted({k for row in sampler.rows for k in row if k != "ts"})
    summary["server"] = {
        "samples": len(sampler.rows),
        "scrape_errors": sampler.errors,
        "disabled_endpoints": sampler.disabled_endpoints,
        "series": ts_summary(sampler.rows, keys),
        # per-run (window-scoped) counter deltas: use these for A/B, not the
        # engine's cumulative gauges
        "window": sampler.window_deltas(),
    }

    # ---------------- persist ------------------------------------------ #
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)

    meta = {
        "run_id": run_id,
        "config": asdict(cfg),
        "host": host_info(),
        "prompt_builder_method": builder.method,
        "chars_per_token": cfg.chars_per_token,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)),
        "git_commit": os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip(),
        "notes": cfg.notes,
        "env_file": None,
    }
    env_path = Path(cfg.out_dir) / "env.json"
    if env_path.exists():
        try:
            meta["environment"] = json.loads(env_path.read_text(encoding="utf-8"))
            meta["env_file"] = str(env_path)
        except Exception:
            pass
    with open(out / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False, default=str)

    cols = ["idx", "request_id", "mode", "status", "http_status", "error",
            "arrival_ts", "start_ts", "first_token_ts", "end_ts",
            "queue_wait_ms", "ttft_ms", "tpot_ms", "e2e_ms", "sse_chunks",
            "prompt_tokens", "completion_tokens", "token_source"]
    with open(out / "requests.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols + (["itl_ms"] if cfg.record_itl else []))
        w.writeheader()
        for r in results:
            d = {k: getattr(r, k) for k in cols}
            for k in ("arrival_ts", "start_ts", "first_token_ts", "end_ts"):
                d[k] = round(d[k], 6)
            for k in ("queue_wait_ms", "ttft_ms", "tpot_ms", "e2e_ms"):
                d[k] = round(d[k], 4) if d[k] == d[k] else ""
            if cfg.record_itl:
                d["itl_ms"] = json.dumps([round(v, 3) for v in r.itl_ms])
            w.writerow(d)

    sampler.write(out / "timeseries.csv")
    print(json.dumps(summary["throughput"], indent=2))
    print(f"[loadgen] results -> {out}")
    return 0 if summary["error_rate"] < 0.5 else 1


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", help="YAML workload profile")
    p.add_argument("--base-url")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    p.add_argument("--model")
    p.add_argument("--endpoint", choices=["chat", "completions"])
    p.add_argument("--mode", choices=["open", "closed"])
    p.add_argument("--arrival-process", choices=["poisson", "uniform"])
    p.add_argument("--request-rate", type=float)
    p.add_argument("--max-concurrency", type=int)
    p.add_argument("--num-requests", type=int)
    p.add_argument("--duration-sec", type=float)
    p.add_argument("--input-len", type=int)
    p.add_argument("--output-len", type=int)
    p.add_argument("--min-input-len", type=int)
    p.add_argument("--min-output-len", type=int)
    p.add_argument("--force-output-len", dest="force_output_len",
                   action="store_true", default=None,
                   help="send ignore_eos=true + max_tokens so every response has "
                        "exactly --output-len tokens (default: on)")
    p.add_argument("--no-force-output-len", dest="force_output_len",
                   action="store_false",
                   help="let the model stop when it wants to (realistic output "
                        "lengths, but length variance confounds A/B runs)")
    p.add_argument("--warmup-requests", type=int)
    p.add_argument("--rampup-sec", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--prompt-style", choices=["lorem", "shared_prefix", "file"])
    p.add_argument("--prompt-file")
    p.add_argument("--shared-prefix-tokens", type=int)
    p.add_argument("--tokenizer")
    p.add_argument("--chars-per-token", type=float)
    p.add_argument("--scrape-interval", type=float)
    p.add_argument("--scrape-disable-after", type=int)
    p.add_argument("--metrics-endpoint", action="append", dest="metrics_endpoints")
    p.add_argument("--gpu-metrics-endpoint", action="append",
                   dest="gpu_metrics_endpoints")
    p.add_argument("--extra-body", help="JSON merged into every request body")
    p.add_argument("--out-dir")
    p.add_argument("--tag")
    p.add_argument("--notes")
    a = p.parse_args(argv)
    if a.extra_body:
        a.extra_body = json.loads(a.extra_body)
    return a


def main() -> int:
    a = parse_args()
    overrides = {k: v for k, v in vars(a).items()
                 if k not in ("profile", "extra_body") and v is not None}
    if getattr(a, "extra_body", None):
        overrides["extra_body"] = a.extra_body
    cfg = Config.load(a.profile, overrides)
    try:
        return asyncio.run(amain(cfg))
    except KeyboardInterrupt:
        print("\n[loadgen] interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
