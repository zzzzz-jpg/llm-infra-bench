# 03 · 指标字典：定义、来源、公式、失效模式

需求里点名的八类指标，逐个给出**精确定义**、**从哪里采**、**怎么算**、
以及**什么情况下这个数字会骗你**。

---

## 1. 客户端延迟指标（由 `loadgen.py` 从 SSE 时间戳推导）

对第 *i* 个请求，记录四个时间戳（epoch 秒，压测机单调时钟同源）：

```
arrival_ts       应该发出的时刻（开环：由到达过程决定；闭环：worker 空闲的时刻）
start_ts         实际发起 HTTP POST 的时刻
first_token_ts   收到第一个非空 content chunk 的时刻
end_ts           流结束（[DONE] 或连接关闭）的时刻
token_times[1..n] 每个非空 content chunk 的到达时刻
```

### TTFT — Time To First Token

```
TTFT_i = first_token_ts - start_ts
```

**含义**：用户从按下回车到看见第一个字的时间。包含 tokenization、
排队、prefill 全量计算、第一次采样、序列化、网络。

**分解**（诊断时按这个顺序查）：
```
TTFT = 客户端排队(client_queue_wait)
     + 网络往返/2
     + 网关缓冲(≈0，若 >0 说明 proxy_buffering 没关)
     + 引擎排队(num_requests_waiting 期间)
     + prefill 计算 ≈ 2·P·L_prompt / (FLOPs·MFU)
     + 首 token 采样与 detokenize
```
`capacity_model.py` 会算出 prefill 的理论下限，实测 TTFT 减去它
就是"排队 + 开销"，这才是可优化的部分。

**会骗你的情况**：
- 网关开了缓冲 → 所有 token 一起到，TTFT ≈ E2E。
  检查方法：`sse_chunks` 应该等于 `completion_tokens`；若 `sse_chunks == 1` 就是被缓冲了。
- 用了 `stream=false` → 根本没有 TTFT，只有 E2E。
- 压测机饱和 → `client_queue_wait` 被算进 TTFT。
  `loadgen.py` 单独报告 `client_queue_wait`，正常应 < 1ms。

### TPOT — Time Per Output Token

```
TPOT_i = (end_ts - first_token_ts) / (n_completion_tokens - 1)
```

**含义**：稳态出字速度，决定"打字机效果"是否流畅。分母是 **n−1**，
因为第一个 token 的时间已经被 TTFT 吃掉了。
`tests/test_itl_count_matches_output_tokens` 钉住了
"n 个输出 token 必须产生 n−1 个间隔"，分母写错会让所有 decode 数字偏移一个系数。

**会骗你的情况**：
- token 数用 SSE chunk 数近似 → 引擎可能一个 chunk 多个 token。
  `summary.tokens.token_source` 必须是 `["usage"]`。
- 请求被抢占后重算 → TPOT 里混进了一整次 prefill。
  对照 `vllm:num_preemptions_total` 是否非零。

### ITL — Inter-Token Latency

```
ITL_i = [token_times[k] - token_times[k-1] for k in 2..n]
```

TPOT 是**每请求的平均**，ITL 是**每个间隔的原始分布**。
两者的 P99 不一样，而且都重要：

- `TPOT p99` = 最慢的那 1% 请求的平均出字速度。
- `ITL p99` = 所有间隔里最慢的 1%。**这个才对应"卡顿"** ——
  一个请求整体很快，但中间卡了 800ms，用户会看到明显的停顿。

continuous batching 下 ITL 会周期性尖峰：每当一个长 prompt 进入 batch，
那一步的 decode 就要等 prefill 做完。这正是 chunked prefill 要解决的问题，
所以 **ITL p99 是评价 chunked prefill 效果的正确指标，不是 TTFT**。

### E2E 延迟

```
E2E_i = end_ts - start_ts ≈ TTFT_i + (n_i - 1) · TPOT_i
```

**会骗你的情况**：固定输出长度时 E2E 几乎完全由 TPOT 决定，
所以"E2E 变好了"通常只是"TPOT 变好了"的重复表述，不是独立证据。

---

## 2. 百分位：怎么算才对

```python
P99 = np.percentile(values, 99)     # 线性插值，与 numpy 默认一致
```

### 三条必须遵守的规则

1. **百分位不能平均。** `mean(P99_of_run1, P99_of_run2) ≠ P99_of_all`。
   要合并多次重复，就把 `requests.csv` 拼起来重新算，
   或者报告"三次 P99 的中位数 + IQR"（本项目采用后者，因为它同时给出噪声）。

2. **客户端百分位 ≠ Prometheus 百分位。**
   Prometheus 的 `histogram_quantile()` 是在**预设 bucket 边界**上做线性插值：
   ```
   vllm TTFT bucket 边界通常是 0.001,0.005,0.01,...,10 秒
   如果真实 P99 = 1.83s 而 bucket 是 [1.0, 2.5]，
   histogram_quantile 会返回 ~1.79s（插值），误差可达数十个百分点。
   ```
   客户端百分位是对**原始样本**排序，没有 bucket 误差。
   **结论**：告警与仪表盘用 Prometheus 的（因为要实时、要跨副本聚合），
   实验结论用客户端的。两者差得离谱时，说明 bucket 边界需要重新划分。

3. **样本量决定尾部的可信度。** 100 个样本的 P99 就是最慢的那一个，
   它是单点观测，不是统计量。`summary.json` 里每个百分位都带 `count`，
   看到 `count < 300` 就别引用 P99。

---

## 3. 吞吐指标

| 指标 | 公式 | 用途 | 陷阱 |
|---|---|---|---|
| **RPS（达成）** | `requests_ok / wall_clock` | 服务能力 | 开环下可能 < 请求速率（说明已过拐点）；`attempted_rps` 一并记录 |
| **输出吞吐** | `Σ completion_tokens / wall_clock` | **decode 容量，最主要的容量指标** | 必须用 usage 里的 token 数 |
| **输入吞吐** | `Σ prompt_tokens / wall_clock` | prefill 容量 | 长上下文场景这才是瓶颈 |
| **总吞吐** | 上两者之和 | 综合 | 混合了两种性质不同的工作，慎用 |
| **Goodput** | `满足 SLO 的请求数 / wall_clock` | **真正能承诺给业务的量** | 依赖 SLO 定义；本项目默认 TTFT p99 ≤ 2s 且 TPOT p99 ≤ 100ms，可在 `loadgen.py` 的 `slo` 块改 |
| **SLO attainment** | 满足 SLO 的请求占比 | 质量视角 | 与 goodput 一起看 |
| **错误率** | `requests_error / requests_total` | 可用性 | 见 §6 |

**为什么 goodput 比 throughput 重要**：
在拐点之后，吞吐还能再涨 10%，但 TTFT P99 可能涨 20 倍。
只看吞吐的团队会把 `max-num-seqs` 一路调高，直到用户投诉。
`capacity.py` 找的是**满足 SLO 的最大负载**（knee），不是峰值吞吐，
两者在 demo 数据里分别是 conc=12（889 tok/s）和 conc=16（899 tok/s）——
多 1% 的吞吐换来 TTFT P99 从 81ms 涨到 3117ms。

---

## 4. 服务端指标（引擎 `/metrics`，1s 粒度抓取）

`loadgen.py` 的 `INTERESTING` 表把 vLLM 与 SGLang 的命名归一：

| 归一别名 | vLLM | SGLang | 怎么读 |
|---|---|---|---|
| `requests_running` | `vllm:num_requests_running` | `sglang:num_running_reqs` | 当前 batch 宽度。稳定等于 `max-num-seqs` ⇒ 已撞admission 上限 |
| `requests_waiting` | `vllm:num_requests_waiting` | `sglang:num_queue_reqs` | **TTFT 的先导指标**。>0 持续 ⇒ 已过拐点 |
| `requests_swapped` | `vllm:num_requests_swapped` | — | 被换出到 CPU 的序列；非零说明 KV 不够 |
| `kv_cache_util` | `vllm:gpu_cache_usage_perc` | `sglang:token_usage` | KV 池占用率。>0.9 ⇒ 即将开始抢占 |
| `prefix_cache_hit_rate` | `vllm:gpu_prefix_cache_hit_rate` | `sglang:cache_hit_rate` | ⚠️ **进程启动以来累计值**，见下 |
| `prefix_cache_{hits,queries}_total` | 同名 counter | `cache_hit_tokens_total` / `cache_lookup_tokens_total` | 用它做**窗口差分**才是本次 run 的命中率 |
| `preemptions_total` | `vllm:num_preemptions_total` | — | 抢占次数。非零 = 纯浪费的 GPU 工作 |
| `prompt_tokens_total` | 同名 | 同名 | 累计 prefill token，差分成 prefill tok/s |
| `gen_tokens_total` | `vllm:generation_tokens_total` | `sglang:generation_tokens_total` | 累计 decode token，差分成引擎侧 decode tok/s（可与客户端值交叉验证） |
| `ttft_seconds_{sum,count,bucket}` | 同名 | 同名 | 引擎内部 TTFT，与客户端 TTFT 的差 = 网络+网关+序列化 |
| `tpot_seconds_*` / `inter_token_latency_seconds_*` | 前者 | 后者 | 同上 |

### 4.1 累计 gauge 的陷阱（真实踩到的坑）

vLLM 的 `gpu_prefix_cache_hit_rate` 是 **hits/queries since process start**。
在同一个进程上连跑两次实验：

```
run1 (prefix cache ON, 共享前缀 512)  → gauge = 0.42
run2 (prefix cache ON, 无共享前缀)     → gauge = 0.39   ← 差异被历史稀释，几乎看不出来
```

`loadgen.py` 的对策：对 `*_total` counter 做**窗口内差分**，
结果写在 `summary.server.window`：

```jsonc
"window": {
  "prefix_cache_hits_total": 402_000,
  "prefix_cache_queries_total": 786_000,
  "prefix_cache_hit_ratio_window": 0.5116,   // ← 用这个做 A/B
  "prefix_cache_hits_per_sec": 15_400,
  "window_seconds": 26.1
}
```

**规则**：跨变体对比时，要么重启引擎，要么只看窗口差分。
`docs/04-experiments.md` 的 Prefix Cache 实验用的是后者。

### 4.2 交叉验证

客户端算出的 `output_tokens_per_sec` 与引擎侧
`rate(vllm:generation_tokens_total[1m])` 应该一致（差 <5%）。
不一致意味着：连接被中途切断（客户端少算）、
有多个客户端在打同一个服务（引擎多算）、或者抓取时机与压测窗口错位。
`summary.json` 两个值都有，对比它们是免费的正确性检查。

---

## 5. GPU 指标（dcgm-exporter）

| 指标 | 含义 | 为什么单看它会误判 |
|---|---|---|
| `DCGM_FI_DEV_GPU_UTIL` | **过去采样窗口内有 kernel 在跑的时间占比** | ⚠️ 最常见误读：一个只用了 1% SM 的 kernel 一直占着 GPU，这个值也是 100%。**"GPU util 100% 但很慢"完全可能** |
| `DCGM_FI_PROF_SM_ACTIVE` | 至少一个 warp 活跃的 SM 比例的时间加权 | 真正的"算力占用" |
| `DCGM_FI_PROF_SM_OCCUPANCY` | 活跃 warp 数 / 最大可驻留 warp 数 | 低 ⇒ kernel 并行度不足 |
| `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | Tensor Core 流水线活跃比例 | **prefill 高、decode 低**是正常形态 |
| `DCGM_FI_PROF_DRAM_ACTIVE` | HBM 带宽利用比例 | **decode 阶段应该高**（decode 是带宽受限）；如果 decode 时它低而 TPOT 高，说明 batch 太窄 |
| `DCGM_FI_DEV_FB_USED` / `FB_FREE` | 显存已用/剩余 (MiB) | vLLM 会预分配 KV 池，所以这个值**加载后就该是平的**。若随流量增长 ⇒ `gpu-memory-utilization` 配置有问题或有别的进程 |
| `DCGM_FI_DEV_POWER_USAGE` | 功耗 (W) | 逼近 `POWER_MGMT_LIMIT` ⇒ 会降频 |
| `DCGM_FI_DEV_GPU_TEMP` | 温度 | >85 °C 触发降频，表现为"负载没变但 TPOT 变差" |
| `DCGM_FI_DEV_SM_CLOCK` / `MEM_CLOCK` | 实际频率 | 判断是否降频的唯一直接证据 |
| `DCGM_FI_DEV_XID_ERRORS` | XID 错误码 | 硬件/驱动故障。XID 79（掉卡）、74（NVLink）需要下线节点 |
| `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL` | 不可纠正 ECC | 非零即换卡 |
| `DCGM_FI_PROF_NVLINK_TX_BYTES` | NVLink 流量 | TP > 1 时应该很高；很低说明 all-reduce 没走上 NVLink（PCIe fallback，性能崩塌） |

### 5.1 用 GPU 指标判断瓶颈形态

```
                     SM_ACTIVE  TENSOR_ACTIVE  DRAM_ACTIVE  含义
prefill 主导            高          高            中        计算受限，量化/TP 有效
decode 主导，batch 窄    低          低            中        带宽受限但没吃满 → 加大 batch
decode 主导，batch 宽    中          中            高        健康的 decode 饱和
都不高但延迟高           低          低            低        ⚠️ 不是 GPU 问题：
                                                            查 CPU（tokenizer）、
                                                            Python 开销、网关、
                                                            或压测机自身
```

第四行是最有价值的诊断：很多人看到"GPU 利用率低"就去换更大的卡，
而真实原因是 detokenizer 占了 CPU、或 Ingress 在缓冲。

---

## 6. 队列与错误率

### 队列

三层队列，别混为一谈：

| 层 | 指标 | 满了会怎样 |
|---|---|---|
| 网关 | `nginx_ingress_controller_nginx_process_connections` / `limit-connections` | 502 / 429，**不进入引擎** |
| 引擎调度器 | `vllm:num_requests_waiting` | TTFT 线性恶化（每多一个排队请求 ≈ 多一次 prefill 时间） |
| 客户端 | `client_queue_wait`（loadgen 自测） | >0 说明压测器自身饱和，**数据作废** |

### 错误率

`loadgen.py` 把错误分类记录在 `summary.errors_by_type`：

| 类型 | 典型原因 | 第一反应 |
|---|---|---|
| `http_429` | 网关限流 | 对比限流阈值与实测 knee，哪个设错了 |
| `http_500` | 引擎内部错误 | 看引擎日志；常见 CUDA OOM 或非法 token id |
| `http_503` | 无健康后端 | 副本全挂或 readiness 全红 |
| `timeout` | 客户端超时 | 是网关 60s 切断，还是真的慢？对比 E2E 分布 |
| `ClientConnectorError` | 连接池/端口耗尽 | 压测机 `somaxconn`、`tcp_tw_reuse` |
| `ServerDisconnectedError` | 引擎崩溃或 preStop 期间 | 对照 Pod restart 计数 |

**注意**：vLLM 的 `request_success_total{finished_reason="abort"}` 上升
意味着**客户端主动断开**。这在流式场景里通常是延迟问题的伪装 ——
用户等不及关掉了页面，表现为"错误率上升"，根因是 TTFT。
告警 `InferenceErrorRateHigh` 的 description 里专门写了这一条。

---

## 7. 一次 run 的完整指标清单

```
summary.json
├── wall_clock_sec, requests_total/ok/error, error_rate, errors_by_type{}
├── throughput{ requests_per_sec, attempted_rps, output/input/total_tokens_per_sec }
├── latency_ms{
│     ttft{count,mean,min,p50,p90,p95,p99,max,stdev},
│     tpot{...}, e2e{...}, itl{...}, client_queue_wait{...}
│   }
├── tokens{ prompt/completion_tokens_mean+total, token_source[] }
├── slo{ ttft_p99_ms, tpot_p99_ms, goodput_rps, slo_attainment }
└── server{
      samples, scrape_errors, disabled_endpoints[],
      series{ <别名>{count,mean,min,p50,p90,p95,p99,max,stdev} },   ← 逐秒采样的统计
      window{ <counter>_total 差分, *_per_sec, prefix_cache_hit_ratio_window }
    }

requests.csv   逐请求：idx, status, http_status, error, 四个时间戳,
               queue_wait_ms, ttft_ms, tpot_ms, e2e_ms, sse_chunks,
               prompt/completion_tokens, token_source, itl_ms(JSON 数组)

timeseries.csv 逐秒：ts + 上述所有服务端别名的瞬时值 + counter 的差分速率
```

`requests.csv` 保留逐请求原始数据，意味着**任何统计口径的改动都可以离线重算**，
不需要重新压测。这是把 GPU 时间花在刀刃上的关键设计。
