#!/usr/bin/env python3
"""
record_env.py — Snapshot the hardware + software environment into results/env.json.

Requirement: "input length, output length, concurrency, request rate and
*hardware information* are all recorded." loadgen.py picks this file up
automatically and embeds it in every run's meta.json, so no result can exist
without the machine it was produced on.

Collected: GPU (name, VRAM, clocks, power cap, driver, CUDA, ECC, MIG,
persistence mode, NUMA/topology), CPU, RAM, kernel, container limits, network,
disk, and the versions of vllm / sglang / torch / transformers / flash-attn.

    python3 scripts/record_env.py --out bench/results/env.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def sh(cmd: str, timeout: int = 25) -> str:
    try:
        return subprocess.run(shlex.split(cmd), capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except Exception:
        return ""


def csv_dicts(out: str) -> List[Dict[str, str]]:
    lines = [l for l in out.splitlines() if l.strip()]
    if len(lines) < 2:
        return []
    hdr = [h.strip() for h in lines[0].split(",")]
    rows = []
    for line in lines[1:]:
        vals = [v.strip() for v in line.split(",")]
        if len(vals) == len(hdr):
            rows.append(dict(zip(hdr, vals)))
    return rows


def gpus() -> List[Dict[str, Any]]:
    q = ("index,name,pci.bus_id,uuid,driver_version,memory.total,memory.used,"
         "memory.free,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,"
         "utilization.gpu,utilization.memory,persistence_mode,ecc.mode.current,"
         "compute_mode,mig.mode.current")
    rows = csv_dicts(sh(f"nvidia-smi --query-gpu={q} --format=csv"))
    if not rows:
        return []
    topo = sh("nvidia-smi topo -m")
    for r in rows:
        r["_nvlink"] = "see topology_matrix"
    if topo:
        rows[0]["topology_matrix"] = topo
    return rows


def cpu() -> Dict[str, Any]:
    info: Dict[str, Any] = {"logical_cpus": os.cpu_count(),
                            "platform": platform.platform(),
                            "arch": platform.machine()}
    try:
        txt = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"model name\s*:\s*(.+)", txt)
        if m:
            info["model"] = m.group(1).strip()
        info["physical_cores"] = len({l.split(":")[1].strip()
                                      for l in txt.splitlines()
                                      if l.startswith("physical id")}) or None
        flags = re.search(r"flags\s*:\s*(.+)", txt)
        if flags:
            f = flags.group(1).split()
            info["isa_highlights"] = [x for x in ("avx2", "avx512f", "amx_bf16",
                                                  "amx_int8", "bf16") if x in f]
    except OSError:
        pass
    loadavg = sh("cat /proc/loadavg")
    if loadavg:
        info["loadavg"] = loadavg.split()[:3]
    return info


def memory() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            if k in ("MemTotal", "MemAvailable", "SwapTotal", "Hugepagesize"):
                out[k] = v.strip()
    except OSError:
        pass
    thp = sh("cat /sys/kernel/mm/transparent_hugepage/enabled")
    if thp:
        out["transparent_hugepage"] = thp
    return out


def container_limits() -> Dict[str, Any]:
    """K8s/docker CPU+memory limits — the numbers that silently cap throughput."""
    lim: Dict[str, Any] = {}
    reads = {
        "cgroup": "/proc/self/cgroup",
        "cpu_max": "/sys/fs/cgroup/cpu.max",
        "cpu_quota": "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
        "cpu_period": "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
        "mem_max": "/sys/fs/cgroup/memory.max",
        "mem_limit": "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    }
    for k, p in reads.items():
        try:
            lim[k] = Path(p).read_text().strip()
        except OSError:
            pass
    env_keys = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
                "HF_HOME", "TRANSFORMERS_CACHE", "VLLM_CACHE_ROOT",
                "http_proxy", "https_proxy", "no_proxy")
    lim["env"] = {k: os.environ.get(k) for k in env_keys if os.environ.get(k)}
    return lim


def packages() -> Dict[str, Any]:
    pkgs: Dict[str, Any] = {}
    for name, mod in (("vllm", "vllm"), ("sglang", "sglang"), ("torch", "torch"),
                      ("transformers", "transformers"),
                      ("flash_attn", "flash_attn"), ("triton", "triton"),
                      ("xformers", "xformers"), ("cuda_python", "cuda"),
                      ("aiohttp", "aiohttp"), ("numpy", "numpy")):
        try:
            m = __import__(mod)
            pkgs[name] = getattr(m, "__version__", "installed")
        except Exception:
            pkgs[name] = None
    try:
        import torch  # type: ignore
        pkgs["torch_cuda_version"] = torch.version.cuda
        pkgs["torch_cudnn"] = torch.backends.cudnn.version()
        pkgs["torch_gpu_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            pkgs["torch_gpu_count"] = torch.cuda.device_count()
            pkgs["torch_gpu_capability"] = [
                list(torch.cuda.get_device_capability(i))
                for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return pkgs


def serving_probe(base_url: str) -> Dict[str, Any]:
    """Ask the live server what it thinks it is (model id, engine, config)."""
    import urllib.request
    out: Dict[str, Any] = {"base_url": base_url}
    for path in ("/health", "/version", "/v1/models", "/get_server_info",
                 "/get_model_info"):
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=8) as r:
                body = r.read(20000).decode("utf-8", "replace")
                out[path] = {"status": r.status, "body": body[:4000]}
        except Exception as exc:
            out[path] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="bench/results/env.json")
    p.add_argument("--probe-url", default="", help="live server base url to probe")
    a = p.parse_args()

    env: Dict[str, Any] = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": socket.gethostname(),
        "kernel": sh("uname -a"),
        "nvidia_smi_version": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
        "cuda_version": (sh("nvcc --version").splitlines() or [""])[-1],
        "gpus": gpus(),
        "cpu": cpu(),
        "memory": memory(),
        "container_limits": container_limits(),
        "packages": packages(),
        "network": {"mtu": sh("cat /sys/class/net/eth0/mtu"),
                    "ip": sh("hostname -i")},
        "disk": sh("df -h . | tail -1"),
        "git_commit": sh("git rev-parse HEAD"),
        "git_dirty": sh("git status --porcelain") != "",
    }
    if a.probe_url:
        env["server"] = serving_probe(a.probe_url)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(env, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    n_gpu = len(env["gpus"])
    print(f"[record_env] wrote {out} — {n_gpu} GPU(s), "
          f"cpu={env['cpu'].get('logical_cpus')}, "
          f"vllm={env['packages'].get('vllm')}, sglang={env['packages'].get('sglang')}")
    if n_gpu == 0:
        print("[record_env] WARN no GPUs detected. Results from this host must not "
              "be mixed with GPU results in a comparison.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
