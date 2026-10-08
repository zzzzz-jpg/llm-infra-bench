#!/usr/bin/env python3
"""
mock_server.py — CPU-only stand-in for a vLLM/SGLang OpenAI-compatible server.

Purpose: let the whole benchmark → collect → compare pipeline be verified end
to end (CI, laptops, no GPU) and give docs reproducible example numbers.
It is NOT a performance reference: the latencies it emits are synthetic.

What it emulates faithfully
---------------------------
* POST /v1/chat/completions and /v1/completions, stream=true SSE framing,
  `data: {...}` chunks + `data: [DONE]`, final usage chunk when
  `stream_options.include_usage` is set.
* `max_tokens`, `ignore_eos`, `temperature`, prompt as string OR token-id list.
* GET /health, GET /v1/models, GET /metrics (Prometheus text format with the
  vllm:* and DCGM_FI_* series the real exporter ships).
* Load-dependent behaviour: an admission queue, so `num_requests_waiting` grows
  and TTFT degrades once concurrency exceeds `--capacity`, which is what makes
  the knee of the latency/throughput curve visible in tests.

Known limitation, stated so nobody quotes these numbers: the DCGM_FI_* series
from a `--only-dcgm` instance are a random walk that is NOT coupled to engine
load (the real dcgm-exporter reads the same physical GPU the engine runs on,
which a CPU-only mock has no way to emulate). In mock runs the engine series
(queue, KV, prefix hits, token counters) DO respond to load; the GPU series
only exercise the scraping and summarising code paths.

Run:  python3 mock_server.py --port 8000 --capacity 16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time
from collections import deque, OrderedDict
from typing import Any, Dict, List, Optional

from aiohttp import web

START = time.time()
MODEL = "mock-7b-instruct"


class Engine:
    """Toy continuous-batching simulator with an admission queue."""

    def __init__(self, capacity: int, ttft_ms: float, tok_ms: float,
                 gpu_util_idle: int = 5):
        # crude radix-cache emulation: remember the head of recent prompts
        self.seen_prefixes: "OrderedDict[str, int]" = OrderedDict()
        self.prefix_block_tokens = 64
        self.capacity = capacity
        self.ttft_ms = ttft_ms
        self.tok_ms = tok_ms
        self.gpu_util_idle = gpu_util_idle
        self.running = 0
        self.waiting: deque = deque()
        self.lock = asyncio.Lock()
        # prometheus-style counters
        self.prompt_tokens_total = 0
        self.generation_tokens_total = 0
        self.preemptions_total = 0
        self.ttft_sum = 0.0
        self.ttft_count = 0
        self.tpot_sum = 0.0
        self.tpot_count = 0
        self.e2e_sum = 0.0
        self.e2e_count = 0
        self.prefix_hits = 0
        self.prefix_lookups = 0

    async def acquire(self) -> None:
        async with self.lock:
            if self.running < self.capacity:
                self.running += 1
                return
            ev = asyncio.Event()
            self.waiting.append(ev)
        await ev.wait()

    async def release(self) -> None:
        async with self.lock:
            if self.waiting:
                self.waiting.popleft().set()
            else:
                self.running -= 1

    @property
    def kv_util(self) -> float:
        return min(0.99, self.running / max(1, self.capacity) * 0.85)

    @property
    def gpu_util(self) -> int:
        load = self.running / max(1, self.capacity)
        base = self.gpu_util_idle + load * 95
        if ONLY_DCGM:
            # standalone exporter: simulate a busy GPU with a random walk
            self._drift = getattr(self, "_drift", 70.0)
            self._drift = max(20.0, min(99.0, self._drift + random.uniform(-6, 6)))
            return int(self._drift)
        return int(min(100, base))


ENGINE: Engine


def count_prompt_tokens(prompt: Any) -> int:
    if isinstance(prompt, list):
        return len(prompt)
    if isinstance(prompt, str):
        return max(1, len(prompt) // 4)
    return 8


async def stream_completion(req: web.Request, is_chat: bool) -> web.StreamResponse:
    body: Dict[str, Any] = await req.json()
    model = body.get("model", MODEL)
    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 128)
    ignore_eos = bool(body.get("ignore_eos", False))
    stream = bool(body.get("stream", False))
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    prompt = body.get("prompt") if not is_chat else None
    if is_chat:
        msgs = body.get("messages") or []
        prompt = msgs[-1].get("content", "") if msgs else ""
    p_tokens = count_prompt_tokens(prompt)

    # prefix-cache accounting: how many leading tokens were already resident?
    head = prompt[:512] if isinstance(prompt, str) else str(prompt[:128])
    ENGINE.prefix_lookups += p_tokens
    hit_tokens = 0
    if PREFIX_CACHE and head in ENGINE.seen_prefixes:
        hit_tokens = min(ENGINE.seen_prefixes[head], p_tokens)
    if PREFIX_CACHE:
        ENGINE.seen_prefixes[head] = p_tokens
        ENGINE.seen_prefixes.move_to_end(head)
    while len(ENGINE.seen_prefixes) > 64:
        ENGINE.seen_prefixes.popitem(last=False)
    ENGINE.prefix_hits += hit_tokens
    # a prefix hit shortens prefill -> lower TTFT
    prefill_discount = 0.55 * (hit_tokens / max(1, p_tokens))

    t_arrive = time.time()
    await ENGINE.acquire()
    t_start = time.time()

    # TTFT grows with queueing + prefill size
    prefill = ENGINE.ttft_ms / 1000.0 * (1 + p_tokens / 4096.0) \
        * (1 - prefill_discount)
    jitter = random.uniform(0.85, 1.2)
    await asyncio.sleep(prefill * jitter)
    ttft = time.time() - t_arrive
    ENGINE.ttft_sum += ttft
    ENGINE.ttft_count += 1

    # per-token time degrades a little as the batch fills (decode contention)
    per_tok = ENGINE.tok_ms / 1000.0 * (1 + 0.35 * ENGINE.running / max(1, ENGINE.capacity))
    n_out = max_tokens if ignore_eos else random.randint(max(1, max_tokens // 2), max_tokens)

    rid = f"chatcmpl-mock-{random.randint(10**8, 10**9-1)}"
    created = int(time.time())

    if not stream:
        await asyncio.sleep(per_tok * n_out)
        now = time.time()
        ENGINE.e2e_sum += now - t_arrive
        ENGINE.e2e_count += 1
        ENGINE.prompt_tokens_total += p_tokens
        ENGINE.generation_tokens_total += n_out
        text = " ".join(["token"] * n_out)
        await ENGINE.release()
        payload: Dict[str, Any] = {
            "id": rid, "object": "chat.completion" if is_chat else "text_completion",
            "created": created, "model": model,
            "usage": {"prompt_tokens": p_tokens, "completion_tokens": n_out,
                      "total_tokens": p_tokens + n_out},
        }
        if is_chat:
            payload["choices"] = [{"index": 0, "finish_reason": "length",
                                   "message": {"role": "assistant", "content": text}}]
        else:
            payload["choices"] = [{"index": 0, "finish_reason": "length",
                                   "logprobs": None, "text": text}]
        return web.json_response(payload)

    resp = web.StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream", "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no"})
    await resp.prepare(req)

    def chunk(delta_text: str, finish: Optional[str] = None,
              usage: Optional[Dict[str, int]] = None) -> bytes:
        if is_chat:
            choice = {"index": 0, "delta": ({"content": delta_text} if delta_text else {}),
                      "finish_reason": finish}
            if usage is None and not delta_text and not finish:
                choice["delta"] = {"role": "assistant", "content": ""}
            obj: Dict[str, Any] = {"id": rid, "object": "chat.completion.chunk",
                                   "created": created, "model": model,
                                   "choices": [choice]}
        else:
            obj = {"id": rid, "object": "text_completion", "created": created,
                   "model": model,
                   "choices": [{"index": 0, "text": delta_text,
                                "finish_reason": finish, "logprobs": None}]}
        if usage is not None:
            obj["usage"] = usage
        return f"data: {json.dumps(obj)}\n\n".encode()

    await resp.write(chunk(""))
    first = True
    emitted = 0
    tok_times: List[float] = []
    for i in range(n_out):
        await asyncio.sleep(per_tok * random.uniform(0.8, 1.25))
        tok_times.append(time.time())
        piece = " " + random.choice(["model", "latency", "batch", "cache", "gpu",
                                     "token", "queue", "kernel", "memory"])
        await resp.write(chunk(piece))
        emitted += 1
        if first:
            first = False
    await resp.write(chunk("", finish="length" if ignore_eos else "stop"))

    usage = {"prompt_tokens": p_tokens, "completion_tokens": emitted,
             "total_tokens": p_tokens + emitted}
    if include_usage:
        await resp.write(chunk("", finish=None, usage=usage))
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()

    now = time.time()
    e2e = now - t_arrive
    ENGINE.e2e_sum += e2e
    ENGINE.e2e_count += 1
    ENGINE.prompt_tokens_total += p_tokens
    ENGINE.generation_tokens_total += emitted
    if len(tok_times) > 1:
        tpot = (tok_times[-1] - tok_times[0]) / (len(tok_times) - 1)
        ENGINE.tpot_sum += tpot
        ENGINE.tpot_count += 1
    await ENGINE.release()
    return resp



ONLY_DCGM = False
SGLANG_STYLE = False   # emit sglang:* metric names instead of vllm:*
PREFIX_CACHE = True    # --no-prefix-cache emulates vLLM's prefix caching off


async def metrics(_req: web.Request) -> web.Response:
    up = time.time() - START
    lines: List[str] = []

    def emit(name: str, help_: str, value: float, labels: str = "",
             kind: str = "gauge") -> None:
        lines.append(f"# HELP {name} {help_}")
        lines.append(f"# TYPE {name} {kind}")
        lines.append(f"{name}{{{labels}}} {value}" if labels else f"{name} {value}")

    def g(name, help_, value, labels=""):
        emit(name, help_, value, labels, "gauge")

    def c(name, help_, value, labels=""):
        emit(name, help_, value, labels, "counter")

    load = ENGINE.running / max(1, ENGINE.capacity)
    lbl = f'model_name="{MODEL}"'

    if not ONLY_DCGM:
        P = "sglang" if SGLANG_STYLE else "vllm"
        if SGLANG_STYLE:
            g("sglang:num_running_reqs", "Running reqs", ENGINE.running, lbl)
            g("sglang:num_queue_reqs", "Queued reqs", len(ENGINE.waiting), lbl)
            g("sglang:token_usage", "KV/token usage", round(ENGINE.kv_util, 4), lbl)
            g("sglang:cache_hit_rate", "Radix cache hit rate",
              round(ENGINE.prefix_hits / max(1, ENGINE.prefix_lookups), 4), lbl)
            g("sglang:gen_throughput", "Decode tok/s", round(load * 3200, 1), lbl)
        else:
            g(f"{P}:num_requests_running", "Running requests", ENGINE.running, lbl)
            g(f"{P}:num_requests_waiting", "Waiting requests", len(ENGINE.waiting), lbl)
            g(f"{P}:num_requests_swapped", "Swapped requests", 0, lbl)
            g(f"{P}:gpu_cache_usage_perc", "KV cache utilization",
              round(ENGINE.kv_util, 4), lbl)
            # cumulative-since-start gauge, exactly like the real engine
            g(f"{P}:gpu_prefix_cache_hit_rate", "Prefix cache hit rate",
              round(ENGINE.prefix_hits / max(1, ENGINE.prefix_lookups), 4), lbl)
            c(f"{P}:prefix_cache_hits_total", "Prefix cache hit tokens",
              ENGINE.prefix_hits, lbl)
            c(f"{P}:prefix_cache_queries_total", "Prefix cache query tokens",
              ENGINE.prefix_lookups, lbl)
            c(f"{P}:num_preemptions_total", "Preemptions",
              ENGINE.preemptions_total, lbl)
        c(f"{P}:prompt_tokens_total", "Prompt tokens",
          ENGINE.prompt_tokens_total, lbl)
        c(f"{P}:generation_tokens_total", "Generation tokens",
          ENGINE.generation_tokens_total, lbl)
        c(f"{P}:time_to_first_token_seconds_sum", "TTFT sum",
          round(ENGINE.ttft_sum, 4), lbl)
        c(f"{P}:time_to_first_token_seconds_count", "TTFT count",
          ENGINE.ttft_count, lbl)
        c(f"{P}:time_per_output_token_seconds_sum", "TPOT sum",
          round(ENGINE.tpot_sum, 5), lbl)
        c(f"{P}:time_per_output_token_seconds_count", "TPOT count",
          ENGINE.tpot_count, lbl)
        c(f"{P}:e2e_request_latency_seconds_sum", "E2E sum",
          round(ENGINE.e2e_sum, 4), lbl)
        c(f"{P}:e2e_request_latency_seconds_count", "E2E count",
          ENGINE.e2e_count, lbl)
        g(f"{P}:cache_config_info", "Cache config", 1,
          f'{lbl},enable_prefix_caching="{PREFIX_CACHE}",block_size="16"')

    if not ONLY_DCGM:
        pass  # engine block above

    # ---- DCGM-exporter style GPU series (skipped in engine-only mode) ----
    for i in range(2):
        l2 = f'gpu="{i}",uuid="GPU-mock-{i}",device="nvidia{i}",modelName="Mock A100-SXM4-80GB"'
        g("DCGM_FI_DEV_GPU_UTIL", "GPU utilization",
          max(0, min(100, ENGINE.gpu_util - i * 3)), l2)
        g("DCGM_FI_DEV_FB_USED", "Framebuffer used MiB",
          30000 + ENGINE.running * 120, l2)
        g("DCGM_FI_DEV_FB_FREE", "Framebuffer free MiB",
          50000 - ENGINE.running * 120, l2)
        g("DCGM_FI_PROF_SM_ACTIVE", "SM active ratio", round(0.05 + load * 0.9, 4), l2)
        g("DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", "Tensor pipe active",
          round(0.02 + load * 0.55, 4), l2)
        g("DCGM_FI_PROF_DRAM_ACTIVE", "DRAM active", round(0.1 + load * 0.6, 4), l2)
        g("DCGM_FI_DEV_POWER_USAGE", "Power W", round(120 + ENGINE.gpu_util * 2.4, 1), l2)
        g("DCGM_FI_DEV_GPU_TEMP", "Temp C", 45 + int(ENGINE.gpu_util / 6), l2)

    if not ONLY_DCGM:
        g("vllm:uptime_seconds", "uptime", round(up, 1))
    return web.Response(text="\n".join(lines) + "\n",
                        headers={"Content-Type": "text/plain; version=0.0.4"})


async def models(_req: web.Request) -> web.Response:
    return web.json_response({"object": "list",
                              "data": [{"id": MODEL, "object": "model",
                                        "owned_by": "mock"}]})


async def health(_req: web.Request) -> web.Response:
    return web.Response(text="")


async def reset_prefix_cache(_req: web.Request) -> web.Response:
    """vLLM: POST /reset_prefix_cache — drop cached KV blocks.

    Counters are NOT reset (same as the real engine), which is why the harness
    differences them over the measurement window instead of reading the gauge.
    """
    ENGINE.seen_prefixes.clear()
    return web.Response(text="true")


async def flush_cache(_req: web.Request) -> web.Response:
    """SGLang: POST /flush_cache — flush the radix cache and KV pool."""
    ENGINE.seen_prefixes.clear()
    ENGINE.running = 0
    return web.json_response({"status": "ok"})


def build_app(capacity: int, ttft_ms: float, tok_ms: float,
              only_dcgm: bool = False, sglang_style: bool = False,
              hide_engine_routes: bool = False,
              prefix_cache: bool = True) -> web.Application:
    global ENGINE, MODEL, ONLY_DCGM, SGLANG_STYLE, PREFIX_CACHE
    ONLY_DCGM = only_dcgm
    SGLANG_STYLE = sglang_style
    PREFIX_CACHE = prefix_cache
    ENGINE = Engine(capacity, ttft_ms, tok_ms)
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/metrics", metrics)
    app.router.add_post("/reset_prefix_cache", reset_prefix_cache)
    app.router.add_post("/flush_cache", flush_cache)
    if not hide_engine_routes:
        app.router.add_post("/v1/chat/completions",
                            lambda r: stream_completion(r, True))
        app.router.add_post("/v1/completions",
                            lambda r: stream_completion(r, False))
    return app


def main() -> None:
    global MODEL, ONLY_DCGM, SGLANG_STYLE, PREFIX_CACHE
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--capacity", type=int, default=16,
                   help="max concurrent in-flight requests before queueing")
    p.add_argument("--ttft-ms", type=float, default=120.0)
    p.add_argument("--tok-ms", type=float, default=12.0)
    p.add_argument("--model", default="mock-7b-instruct")
    p.add_argument("--only-dcgm", action="store_true",
                   help="expose only DCGM_FI_* series, like a real dcgm-exporter")
    p.add_argument("--no-prefix-cache", action="store_true",
                   help="emulate --no-enable-prefix-caching / --disable-radix-cache")
    p.add_argument("--sglang-style", action="store_true",
                   help="emit sglang:* metric names instead of vllm:*")
    a = p.parse_args()
    MODEL = a.model
    app = build_app(a.capacity, a.ttft_ms, a.tok_ms, only_dcgm=a.only_dcgm,
                    sglang_style=a.sglang_style, hide_engine_routes=a.only_dcgm,
                    prefix_cache=not a.no_prefix_cache)
    print(f"[mock] listening on {a.host}:{a.port} capacity={a.capacity} "
          f"ttft={a.ttft_ms}ms tok={a.tok_ms}ms model={MODEL}")
    web.run_app(app, host=a.host, port=a.port, print=None)


if __name__ == "__main__":
    main()
