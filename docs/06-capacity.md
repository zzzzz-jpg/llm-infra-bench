# 06 · 容量模型与扩容建议

两条腿走路：**实测**（`capacity.py`，回答"这套系统现在能承接多少"）
和**第一性原理**（`capacity_model.py`，回答"这个实测值物理上说得通吗"）。
只有实测会被配置错误骗到；只有理论会忽略真实系统的开销。两者对上，结论才可信。

---

## 1. 从实测到集群的推导链

```
并发/速率扫描 ──▶ knee（满足 SLO 的最大负载点）──▶ 每副本 rps
                                                      │
                              目标峰值 rps × (1+headroom) ÷ 每副本 rps
                                                      │
                                                 副本数 N ──▶ GPU 数 = N × TP
                                                      │
                            ┌─────────────────────────┼──────────────────────────┐
                            ▼                         ▼                          ▼
                    HPA minReplicas           maxReplicas ≥ N            ResourceQuota ≥ N
                    = N（或 N−1）             （留突发空间）              （否则 Pod Pending）
                            │
                            ▼
                    scale target = 0.75 × knee rps
                    --max-num-seqs = knee concurrency
                    PDB minAvailable = N−1
                    网关 limit-rps = N × knee rps
```

一条链上的每个数字都来自同一个实测点。**这是"容量建议"的全部含义** ——
不是一个经验值，而是六个互相一致的配置项。

### 1.1 demo 数据的完整推导（合成数据，仅示范格式）

```bash
python3 bench/capacity.py bench/results/demo --tag-filter fp16-conc \
    --target-rps 20 --slo-ttft-p99-ms 1500 --slo-ttp-p99-ms 50 \
    --headroom 0.3 --gpu-hours-cost 2.5
```

| run | conc | RPS | out tok/s | TTFT p99 | TPOT p99 | KV util | queue p99 | SLO |
|---|---|---|---|---|---|---|---|---|
| fp16-conc4 | 4 | 1.51 | 386 | 149 ms | 10.0 | 0.27 | 0 | ✅ |
| fp16-conc8 | 8 | 2.75 | 703 | 154 ms | 10.9 | 0.52 | 0 | ✅ |
| **fp16-conc12** | **12** | **3.45** | **883** | **150 ms** | **12.1** | **0.68** | **0** | ✅ **← knee** |
| fp16-conc16 | 16 | 3.48 | 891 | **3266 ms** | 11.9 | 0.68 | 4 | ❌ |
| fp16-conc24 | 24 | 3.48 | 891 | 3288 ms | 11.9 | 0.68 | 12 | ❌ |

读这张表的三个要点：

1. **吞吐在 knee 之后基本不再增长**（883 → 891 → 891，+0.9%），
   而 TTFT p99 涨了 **22 倍**。这是饱和的标准形态：
   你多压的每一分负载都变成了排队，而不是产出。
2. **KV util 在 conc16/24 停在 0.68 不再上涨**。不是 KV 有余量，
   而是多出来的请求全在准入队列里等（`requests_waiting` 0 → 4 → 12），
   还没进到占 KV 的阶段。**不要只看 KV util 判断饱和**，
   要和 `requests_waiting` 一起看。
3. `capacity.py` 的诊断是 **Admission bound**（撞 `max-num-seqs` 上限），
   不是 KV bound。对策是**扩副本**，不是提高 `--max-num-seqs` ——
   提高它只会把 TTFT 悬崖换成 TPOT 悬崖。

推导结果：

| 项 | 值 | 来源 |
|---|---|---|
| 每副本 SLO 内 rps | 3.45 | knee 实测 |
| 目标峰值 rps | 20 | 业务输入 |
| headroom | 30% | 覆盖扩容延迟与突发 |
| **副本数** | **ceil(20 × 1.3 / 3.45) = 8** | 公式 |
| **GPU 数** | **8**（TP=1） | 副本数 × TP |
| HPA minReplicas | 7 | N−1，容忍一次 drain |
| HPA maxReplicas | 16 | 2×N，突发空间 |
| HPA scale target | 2.59 rps/pod | 0.75 × knee |
| `--max-num-seqs` | 12 | knee 并发 |
| PDB minAvailable | 7 | N−1 |
| 网关 limit-rps | 27.6 | N × knee rps |
| $/1M output tokens | $0.78 | 见 §5 |

---

## 2. KV cache 预算：并发上限的第一性原理

```
bytes_per_token = 2 (K 和 V)
                × num_hidden_layers
                × num_key_value_heads        ← GQA！不是 num_attention_heads
                × head_dim
                × dtype_bytes

每序列 KV = bytes_per_token × context_length / TP
并发上限  = 可用 KV 显存 ÷ 每序列 KV
可用 KV   = VRAM × gpu_memory_utilization − 权重字节/TP − 激活与碎片(~1–2 GiB)
```

### Qwen2.5-7B-Instruct（28 层，4 个 KV head，head_dim 128，GQA）

```bash
python3 bench/capacity_model.py --preset Qwen2.5-7B --kv-gb 40 --context 8192 --gpu A100-80G
```

```
  KV bytes per token (fp16)  57,344 B = 56.0 KiB
  context   2048 tok : 112.00 MiB/序列 ->  365 并发
  context   4096 tok : 224.00 MiB/序列 ->  182 并发
  context   8192 tok : 448.00 MiB/序列 ->   91 并发
  context  16384 tok : 896.00 MiB/序列 ->   45 并发
  context  32768 tok :   1.75 GiB/序列 ->   22 并发
```

⚠️ **最常见的算错**：用 `num_attention_heads`(28) 而不是
`num_key_value_heads`(4)。GQA 模型上这会把 KV 高估 **7 倍**，
于是你以为只能跑 13 并发，实际能跑 91，然后过度采购 7 倍的卡。

### 2.1 三个直接可执行的结论

| 观察 | 行动 |
|---|---|
| 实测 knee 并发 ≪ KV 并发上限 | **不是显存问题**。查 `--max-num-seqs`、调度开销、prefill 算力 |
| 实测 knee 并发 ≈ KV 并发上限 | 显存就是天花板。`--kv-cache-dtype fp8` 直接 ~2x 并发；或缩短 `--max-model-len`；或权重量化腾显存 |
| `--max-model-len` 远大于实际最长请求 | **纯浪费**。KV 是按 max_model_len 预留调度的，把它从 32k 降到 8k 可以让并发上限涨 4 倍 |

demo 数据里 KV 并发上限（91）远高于实测 knee（12），
所以诊断是 admission bound 而非 KV bound —— 两个数字对上，结论才成立。

---

## 3. Roofline：TPOT 与 TTFT 的物理下限

### 3.1 Decode：带宽受限

每生成一个 token，要把**全部权重**从 HBM 读一遍（batch=1 时）：

```
tpot_floor_ms ≈ (params × bytes_per_param) / (HBM带宽 × 效率) × 1000
```

| GPU | HBM GB/s | 7B FP16 TPOT 下限 | 单流上限 tok/s | 7B int4 TPOT 下限 |
|---|---|---|---|---|
| A100-40G | 1555 | 12.2 ms | 82 | 3.1 ms |
| A100-80G | 2039 | 9.3 ms | 107 | 2.3 ms |
| H100-SXM | 3350 | 5.7 ms | 176 | 1.4 ms |
| H200 | 4800 | 4.0 ms | 253 | 1.0 ms |
| L4 | 300 | 63.3 ms | 16 | 15.8 ms |
| L40S | 864 | 22.0 ms | 46 | 5.5 ms |
| RTX 4090 | 1008 | 18.9 ms | 53 | 4.7 ms |

```bash
python3 bench/capacity_model.py --gpu-table     # 完整表格
python3 bench/capacity_model.py --preset Qwen2.5-7B --gpu A100-80G \
    --measured-tpot-ms 12      # 与你的实测值对照
```

**判读规则**：

| 实测 TPOT / 下限 | 结论 |
|---|---|
| < 1.5x | 健康。剩余收益只能来自 batching，不能来自 kernel |
| 1.5–2.5x | 正常。有可见的采样/detokenize/Python 开销 |
| > 2.5x | **有问题**：PCIe 卡无 NVLink、权重没常驻 HBM、被 CPU 限流、降频、或 CUDA graph 未启用 |

demo 里 fp16 TPOT=12.0ms 对 A100-80G 的下限 9.3ms 是 **1.28x** → 判定"健康"。
这不是巧合，是我按真实形态设定 mock 参数的结果；
真机上如果算出 4x，就说明该去查上面那一行。

### 3.2 Prefill：算力受限

```
prefill_flops ≈ 2 × params × prompt_tokens
ttft_floor_ms ≈ prefill_flops / (峰值FLOPs × MFU) × 1000      # MFU 实际取 0.3–0.6
```

Qwen2.5-7B，1024-token prompt，A100-80G（312 TFLOPs FP16，MFU 0.5）：
```
  2 × 7.62B × 1024 = 15.6 TFLOP  →  TTFT 下限 ≈ 100 ms
  8192-token prompt              →  ≈ 800 ms（线性）
```

**关键推论：prefill 随 prompt 长度线性增长，且不可通过 batching 摊薄**
（batching 摊薄的是权重重读，prefill 的计算量本身就是 O(L)）。
所以长上下文流量的 TTFT 问题，只有三个解法：
chunked prefill（改善 ITL，不改善 TTFT 总量）、prefix cache（跳过重复部分）、加算力（TP）。

### 3.3 Batching 的理论收益

```
compute_bound_decode_ceiling ≈ 峰值FLOPs × MFU / (2 × params)
```
A100-80G + 7B FP16：≈ 10,200 tok/s，对比单流 107 tok/s
⇒ **batching 理论上能带来 ~96x 吞吐提升**。

实测峰值若远低于这个数（比如只有 2000 tok/s），说明 batch 没有填满，
按这个顺序查：offered rate 不够 → `--max-num-seqs` 太小 →
KV 上限（§2）→ `--max-num-batched-tokens` 太小 → 调度器开销。

---

## 4. 冷启动与 HPA 反应时间

这是**最常被忽略、也最容易让自动扩容彻底失效**的一项。

### 4.1 扩容全链路耗时

```
T_detect     指标采集间隔 + 告警/HPA 同步周期      ≈ 15s + 15s
T_stabilize  HPA scaleUp.stabilizationWindowSeconds ≈ 60s（本项目设置）
T_schedule   找到有空闲 GPU 的节点                  ≈ 0s（有空位）/ ∞（没有！）
T_pull       镜像拉取                              ≈ 0s（已缓存）– 10min
T_load       权重读取 + CUDA 上下文 + KV 池 + graph 捕获 ≈ 1–10 min  ← 主导项
T_ready      readiness 首次通过                     ≈ 5s
─────────────────────────────────────────────────────────
T_total                                            ≈ 2–12 min
```

### 4.2 判断 HPA 是否**有可能**应对你的尖峰

```
尖峰上升时间 T_spike  vs  T_total

T_spike > T_total          → HPA 有效，按 §1 配置即可
T_spike ≈ T_total          → 边缘，需要预热池 + 更激进的 scaleUp policy
T_spike < T_total          → HPA 在数学上无效。必须：
                             (a) 常驻副本覆盖尖峰（花钱买确定性）
                             (b) 定时预扩容（cron HPA / KEDA scaledTrigger，
                                 如果尖峰可预测，例如每天 9:00）
                             (c) 网关排队 + 限流削峰（牺牲延迟）
                             (d) 缩短 T_load：节点本地 NVMe 缓存权重、
                                 预热镜像、safetensors + mmap、
                                 关 CUDA graph（用 decode 性能换启动速度）
```

**实测 T_load**：
```bash
kubectl -n llm-inference get pod <pod> -o json | jq -r '
  .status.conditions[] | select(.type=="Initialized" or .type=="Ready") | "\(.type) \(.lastTransitionTime)"'
# 或从引擎日志：grep "init engine" 到 "Application startup complete"
```
把这个数填进 `startupProbe.failureThreshold × periodSeconds ≥ 2 × T_load`，
并写进容量报告。告警 `InferenceStartupTooSlow` 负责监控它的退化。

### 4.3 缩容的隐藏成本

缩掉一个副本丢掉的不只是算力：
- 已加载的权重（重建要 T_load）
- **已预热的 prefix cache**（重建要一整个流量周期）
- 该副本上正在进行的流式响应（靠 preStop + grace period 保护）

所以 `scaleDown.stabilizationWindowSeconds: 600`（10 分钟）不是保守，是必要。

---

## 5. 成本模型

```
$ / 1M output tokens = (GPU_hourly_cost × GPUs_per_replica)
                       ÷ (peak_output_tokens_per_sec × 3600)
                       × 1e6
```

demo 数据：`2.5 × 1 / (891 × 3600) × 1e6 = $0.78 / 1M output tokens`。

三个必须注意的点：

1. **用 knee 吞吐还是 peak 吞吐？**
   peak（891 tok/s）是 SLO 违约状态下的数字。
   诚实的成本应该用 **knee 吞吐**（883 tok/s），
   或者更严格：用 **goodput**（满足 SLO 的 token 速率）。
   三者差异越大，说明系统越接近悬崖运行。
2. **要算上 headroom 的成本。** 你为了应对尖峰常驻 30% 空闲副本，
   这 30% 是真实成本。`实际单位成本 = 上式 × (1 + headroom)`。
3. **量化的收益要按成本算，不按速度算。**
   AWQ 让吞吐 +51%（demo 数据）⇒ 单位成本 −34%。
   但如果质量下降导致需要人工复核 5% 的输出，人工成本可能远超省下的 GPU 钱。

---

## 6. 什么时候需要更复杂的架构

按顺序判断，不要跳步：

| 信号 | 判断 | 行动 |
|---|---|---|
| 单副本 knee 并发 ≪ KV 上限，GPU util 低 | batch 没填满 | 调 `max-num-seqs` / `max-num-batched-tokens`，**不要加卡** |
| 单副本 knee 并发 ≈ KV 上限 | 显存受限 | fp8 KV cache → 权重量化 → 加显存 |
| TPOT 远高于 roofline 下限（>2.5x） | 引擎/硬件问题 | 查 CUDA graph、PCIe、降频、CPU 限流 |
| TTFT 差、TPOT 好，且 prompt ≫ completion | **prefill 主导** | chunked prefill、prefix cache、TP↑ |
| TTFT 好、TPOT 差，且 completion ≫ prompt | **decode 主导** | 加 batch、量化、投机解码 |
| 上面都试过，流量仍大且 prompt/completion 比例悬殊 | prefill 与 decode 互相干扰 | **考虑 prefill/decode 分离部署**（不同 GPU 池、不同并行策略）；代价是 KV 需要跨节点传输，运维复杂度显著上升 |
| 单卡放不下模型 | 必须并行 | TP（同节点，要 NVLink）优先于 PP（跨节点，气泡开销） |
| 单请求延迟是硬约束且已到 roofline | 物理极限 | 换更快的卡（HBM 带宽决定 decode），没有软件解 |

---

## 7. 容量建议报告模板（交付物）

```markdown
# 容量建议 — <模型> on <GPU> @ <日期>

## 1. 基准
- 负载契约：in=<L> tok, out=<M> tok, <open/closed>-loop, seed=<S>
- SLO：TTFT p99 ≤ <X> ms, TPOT p99 ≤ <Y> ms, 错误率 ≤ <Z>%
- 硬件：<GPU 型号> × <n>, 驱动 <v>, CUDA <v>, vLLM <v>
- 噪声底：±<p>%（<k> 次同配置重复）
- 数据来源：bench/results/<run_ids>

## 2. 单副本能力（实测）
| 指标 | 值 |
|---|---|
| knee 并发 | |
| knee rps | |
| knee output tok/s | |
| knee TTFT p99 / TPOT p99 | |
| peak output tok/s（SLO 违约，仅参考） | |
| 瓶颈诊断 | <capacity.py 的输出> |

## 3. 理论交叉校验
| 量 | 理论 | 实测 | 比值 | 判读 |
|---|---|---|---|---|
| TPOT @conc=1 | <roofline 下限> | | | |
| KV 并发上限 @<ctx> | <公式> | <knee 并发> | | |
| TTFT @<L> tok | <prefill 下限> | | | |

## 4. 集群规模
| 项 | 值 | 依据 |
|---|---|---|
| 目标峰值 rps | | 业务 |
| headroom | | |
| **副本数 / GPU 数** | | §1 公式 |
| HPA min / max / target | | |
| PDB minAvailable | | |
| `--max-num-seqs` | | = knee 并发 |
| 网关 limit-rps | | = N × knee rps |
| ResourceQuota GPUs | | ≥ HPA maxReplicas |

## 5. 冷启动与扩容可行性
| 项 | 值 |
|---|---|
| T_load 实测 | |
| startupProbe 预算 | ≥ 2 × T_load |
| T_total（检测→就绪） | |
| 业务尖峰上升时间 | |
| **HPA 是否可行** | 可行 / 边缘 / 不可行（需预热池或定时扩容） |

## 6. 成本
| 项 | 值 |
|---|---|
| $/1M output tokens（knee 吞吐） | |
| 含 headroom 的实际单位成本 | |
| 月度 GPU 成本（N 副本 × 单价 × 730h） | |

## 7. 未测量 / 风险
- 上下文长度分布变化（本结论仅在 in=<L> 下成立）
- 多副本 prefix cache 稀释（未测路由策略）
- 邻居干扰（本次为独占节点 / 共享节点）
- 未覆盖：<TP>1、fp8 KV、投机解码、长上下文、多轮>
```

---

## 8. 快速命令参考

```bash
# 理论：KV 预算 + roofline
python3 bench/capacity_model.py --preset Qwen2.5-7B --kv-gb 40 --context 8192 --gpu A100-80G
python3 bench/capacity_model.py --gpu-table
python3 bench/capacity_model.py --model /models/Qwen2.5-7B-Instruct \
    --kv-gb 40 --measured-tpot-ms 12 --measured-ttft-ms 320

# 实测：拐点 + 副本数 + HPA 片段
python3 bench/capacity.py bench/results --tag-filter <tag> \
    --target-rps 40 --slo-ttft-p99-ms 2000 --slo-tpot-p99-ms 100 \
    --headroom 0.3 --gpu-hours-cost 2.5 --gpus-per-replica 1 \
    --out capacity-plan.md

# 变量对照后的容量重估
for T in vllm-fp16 vllm-awq sglang-fp16; do
  python3 bench/capacity.py bench/results --tag-filter $T \
      --target-rps 40 --out bench/results/capacity-$T.md
done
grep -h "replicas required" bench/results/capacity-*.md
```
