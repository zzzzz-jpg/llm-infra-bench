# 02 · 固定负载方法论

> "固定负载"不是"随便压一压"。它的意思是：**除了被研究的那一个变量，
> 其它一切都被钉死并被记录下来**，包括那些看起来无关的东西（seed、warmup 数、
> 压测机 CPU、甚至一天中的时间）。

---

## 1. 被固定的五个维度

| 维度 | 参数 | 默认值 | 为什么必须固定 |
|---|---|---|---|
| **输入长度** | `input_len` / `min_input_len` | 1024 tok | prefill 的计算量 ≈ 2·P·L，L 翻倍 TTFT 就翻倍。输入长度不固定 = TTFT 不可比 |
| **输出长度** | `output_len` + `force_output_len` | 256 tok | decode 步数 = 输出 token 数。不固定则吞吐与 TPOT 都被长度方差污染 |
| **并发** | `max_concurrency` | 64 | 决定 batch 宽度，进而决定 KV 占用与 TPOT |
| **请求速率** | `mode` + `request_rate` | open / 4 rps | 决定系统是否在拐点前。开环与闭环回答的是**不同的问题** |
| **硬件与软件** | `record_env.py` → `env.json` | 自动 | GPU 型号、驱动、CUDA、vLLM 版本、cgroup 限额，任一不同都不可比 |

再加上三个容易被忽略但会改变结论的：

| 维度 | 参数 | 说明 |
|---|---|---|
| **prompt 内容** | `seed` | 同 seed ⇒ 逐字节相同的 prompt 流。不同 prompt 会命中不同的 prefix cache、不同的 token 分布 |
| **预热** | `warmup_requests` | 首批请求包含 CUDA graph 捕获、KV 池首次分配、tokenizer 预热。这些**必须**被排除，且**单独存进 requests.csv** 以便审计 |
| **间隔冷却** | `COOLDOWN` (run_matrix.sh) | 连续两个 run 之间等 KV 池排空，否则上一个 run 的残留占用会算到下一个头上 |

---

## 2. 开环 vs 闭环：这是两个不同的问题

```
闭环 (closed-loop, mode=closed)
  N 个 worker，每个 worker 收到响应后立刻发下一个请求。
  到达速率 = 服务速率（自我调节）。零 think time。

     worker ──req──▶ [service] ──resp──▶ worker ──req──▶ ...

开环 (open-loop, mode=open)
  按 Poisson(λ) 独立到达，不管前一个请求有没有回来。
  到达速率与服务速率解耦。

     λ=4rps ──▶ ──▶ ──▶ ──▶ [service]  如果服务跟不上，队列无限增长
```

| | 闭环 | 开环 |
|---|---|---|
| 回答的问题 | **峰值吞吐是多少？拐点在哪？** | **能不能承接 X rps 且满足 SLO？** |
| 自变量 | 并发 N | 到达率 λ |
| 过饱和时的表现 | 延迟上升，吞吐持平（优雅） | 队列发散，延迟无界（真实崩溃） |
| 用于 | 容量规划、找 knee | SLO 验证、HPA 目标值 |
| 本项目里 | `profiles/saturation.yaml`、`SUITE=conc` | `profiles/baseline.yaml`、`SUITE=rate` |

### 2.1 闭环的经典陷阱：coordinated omission

闭环压测里，一个慢了 10 秒的请求会**阻止**这个 worker 发出后面的请求。
于是"系统变慢"的同时"施压也变慢"，尾部延迟被系统性低估。
Gil Tene 把这个叫 coordinated omission（协调遗漏），它是绝大多数
"我们 P99 只有 200ms" 结论的来源。

**本项目的对策：**
1. SLO 与容量结论一律用**开环**数据（`mode=open`，Poisson 到达）。
2. 闭环数据只用来找峰值吞吐和拐点，且在报告里标注 `mode=closed`。
3. `loadgen.py` 记录 `client_queue_wait`（从"应该发出的时刻"到"真正建连的时刻"）。
   开环模式下，如果这个值显著大于 0，说明压测器自己成了瓶颈 —— 测的是客户端，不是服务。
4. `alert: LoadGeneratorStarved` 用 node-exporter 的数据在压测机饱和时告警。

### 2.2 到达过程：Poisson 还是等间隔？

```yaml
arrival_process: poisson   # 默认：无记忆，有突发性，贴近真实用户流量
arrival_process: uniform   # 严格 1/λ 等间隔：方差为零，可复现性最强
```

- **Poisson** 用于容量与 SLO 结论。真实流量是突发的，等间隔到达会让你
  高估系统能力（同样的均值速率，Poisson 的队列更长）。
  `tests/test_poisson_arrivals_are_memoryless_not_deterministic` 会检查
  实际间隔的方差确实接近指数分布（var ≈ mean²），防止有人偷偷把它写成等间隔。
- **uniform** 用于回归测试与 CI 断言：`test_open_loop_delivers_requested_rate`
  用它把"达成 rps"的断言范围收窄到 ±12%，否则指数分布的方差会让测试偶发失败。

---

## 3. 长度控制的三种精度等级

`PromptBuilder` 按可用条件自动选择，并把用了哪种写进 `meta.json`
（字段 `prompt_builder_method`）：

| 方法 | 触发条件 | 精度 | 代价 |
|---|---|---|---|
| `tokenizer` | 设了 `--tokenizer` | **精确到 token** | 需要 transformers + 下载 tokenizer |
| token-id 模式 | `endpoint=completions` 且无 tokenizer | **精确到 token** | 只能测 `/v1/completions`；prompt 是随机 token id，语义无意义 |
| `heuristic` | 默认 | ±10–20% | 零依赖；`chars_per_token` 需要按语言校准（英文 ≈4，中文 ≈1.5） |

```python
# token-id 模式为什么精确：OpenAI 的 /v1/completions 允许 prompt 直接是 token id 数组，
# vLLM 与 SGLang 都实现了这个扩展。于是输入长度不再是"估计"，而是字面值。
body["prompt"] = [rng.randrange(1000, 30000) for _ in range(target_in)]
```

**建议**：真实实验一律带 `--tokenizer <模型路径>`。
`verify_run.py` 会在 `token_source` 不是 `usage` 时报错
（意味着 token 数是数 SSE chunk 数出来的，而引擎可能一个 chunk 里塞多个 token）。

### 输出长度：`ignore_eos` 的取舍

```yaml
force_output_len: true    # 发送 max_tokens=N + ignore_eos=true
```

- **开（默认，用于 A/B）**：每个响应恰好 N 个 token。
  decode 步数完全一致，TPOT 与吞吐的方差只来自系统，不来自模型想说什么。
  `tests/test_output_length_is_actually_fixed` 断言实测 `completion_tokens` 集合 == {N}。
- **关（用于贴近生产的容量估算）**：模型自己决定何时停，
  输出长度服从真实分布（通常长尾）。这时"吞吐 tok/s"仍然可比，
  但"每请求延迟"的方差会大得多，需要更多样本。

**正确做法是两个都跑**：A/B 用固定长度（隔离变量），
容量规划用真实长度（反映生产）。`meta.json` 记录了是哪种，
`compare.py` 会在两者混用时报警。

---

## 4. 样本量、重复次数与噪声底

### 4.1 一次 run 要多长？

目标：**尾部百分位有足够样本**。P99 需要至少 ~1000 个请求才有意义
（1000 个样本的 P99 只由最慢的 10 个决定，噪声极大）。

| 用途 | 最少请求数 | 说明 |
|---|---|---|
| 冒烟 / CI | 24–50 | 只验证流水线，不看数字 |
| A/B 单点 | ≥ 300 | P99 勉强可用，P95 可靠 |
| 容量结论 | ≥ 1000 | 或 `duration_sec ≥ 300` 取先到者 |
| 发布前 SLO 验证 | ≥ 3000，≥ 10 min | 覆盖 GC、日志轮转、KV 碎片等慢周期 |

### 4.2 先测噪声底，再声称有提升

**这是整个方法论里最重要的一条。** 同一个配置连跑 3 次，
得到的吞吐/延迟散布就是你的测量噪声底。任何小于噪声底的"提升"都是幻觉。

```bash
for i in 1 2 3; do
  make bench TAG=noise-floor SUITE=baseline N_REQ=500
  sleep 30
done
python3 bench/compare.py bench/results --baseline-tag noise-floor
# 看 Out tok/s 一列的 delta：如果三次之间就有 ±4% 的摆动，
# 那么任何 <5% 的"量化收益"都不能下结论，必须增加重复次数。
```

经验值（单卡、独占、无邻居干扰）：吞吐噪声底通常 1–3%，TTFT P99 可达 5–15%。
**共享集群或云上有其它租户时，噪声底可能到 20%**，这时唯一办法是增加重复次数
并报告置信区间，而不是报告单次数字。

### 4.3 报告什么

不要报告单次的 P99。报告：

```
指标: 中位数 [IQR]，n=重复次数
例：Out tok/s = 4820 [4790–4860]，n=3
    TTFT p99  = 412 ms [380–495]，n=3   ← 注意 IQR 相对中位数的宽度，它就是不确定度
```

`compare.py` 的结论模板要求填 "with the CI/σ if repeats exist"，
就是这个原因。

---

## 5. 有效性检查清单（跑之前过一遍）

```
□ 独占 GPU：没有别的进程在用这张卡（nvidia-smi 里只有你的 vLLM）
□ 时钟已同步：压测机与服务机 chrony/ntp 正常（客户端-服务端时间戳差值可解释）
□ 压测机不饱和：压测机 CPU < 70%，否则测的是客户端
     └ 检查 summary.latency_ms.client_queue_wait 的 p99 是否 ≈ 0
□ 模型已预热：warmup_requests > 0，且 warmup 被排除在统计之外
□ 输出长度已固定（A/B 场景）：force_output_len=true
□ tokenizer 已指定（真实实验）：meta.prompt_builder_method == "tokenizer"
□ token 计数来自 usage：summary.tokens.token_source == ["usage"]
□ 服务端指标抓到了：summary.server.samples > 0 且 scrape_errors == 0
□ 引擎配置一致：两次 run 的 runs/*-serve-cmd.txt diff 只差被研究的那一行
□ 跑过噪声底：知道多少百分比以内的差异不算差异
□ 冷却时间足够：run 之间 KV 池已排空（COOLDOWN ≥ 10s）
□ 结果已通过 verify_run.py
```

`tests/verify_run.py` 自动化了其中 6 条；剩下的需要人来做，
因为它们需要访问被测机器本身。

---

## 6. 一个完整的可复现记录长什么样

```jsonc
// bench/results/demo/<ts>-fp16-conc12/meta.json（节选）
{
  "run_id": "20261008-042210-fp16-conc12",
  "config": {
    "base_url": "http://127.0.0.1:8000",
    "model": "mock-qwen2.5-7b-instruct",
    "endpoint": "chat",
    "input_len": 1024, "output_len": 256, "force_output_len": true,
    "num_requests": 32, "mode": "closed", "max_concurrency": 12,
    "arrival_process": "poisson", "seed": 1234, "warmup_requests": 4,
    "chars_per_token": 3.6, "scrape_interval": 1.0,
    "notes": "closed-loop sweep, concurrency=12"
  },
  "prompt_builder_method": "heuristic",   // ← 告诉你输入长度是估计值
  "host": {"hostname": "...", "cpu_count": 2, "gpus": []},
  "environment": { /* record_env.py 的完整快照：驱动、CUDA、cgroup 限额、包版本 */ },
  "git_commit": "a1b2c3d",
  "started_at": "2026-10-08T04:22:10+0000"
}
```

拿着这个 JSON，任何人都能重跑得到统计上相同的结论 ——
这就是"固定负载"的全部含义。
