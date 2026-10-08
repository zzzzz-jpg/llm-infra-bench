# 07 · 报告示例与讲评（含一次真实的翻车事故复盘）

> **数据来源声明**：本文所有数字来自 `bench/results/demo/`，由 **mock 推理服务**
> （`bench/mock_server.py`，纯 CPU、无 GPU、延迟为参数化合成值）实测产生。
> 它们的用途是**验证报告管线与方法论**，不代表任何真实硬件/引擎的性能。
> 拿到真实 GPU 后，把 `--base-url` 换成真实端点重跑，即可得到同结构的真报告。
>
> 复现：`make demo`（约 6 分钟，23 个 run + 10 份自动报告）。

本文回答三个问题：

1. 一份合格的压测报告长什么样（§1–§5，直接摘自自动生成的报告）；
2. 从报告到扩容决策怎么走（§6）；
3. 这套数据差点全废的那次事故，以及由此固化进工具链的防御（§7）。

---

## 1. 测试环境与固定负载（每个 run 都自动记录）

报告可信的前提是"被测条件完全一致，只动一个变量"。每次 run 的
`meta.json` + `env.json` 自动记录以下内容（摘自 demo 数据）：

| 项目 | 值 |
|---|---|
| 模型 | Qwen2.5-7B-Instruct（mock 别名，8 组变体见下） |
| 引擎 | vLLM / SGLang（mock），AWQ-int4 变体，prefix-cache 开/关变体 |
| 负载形态 | closed-loop（固定并发）或 open-loop（泊松到达，固定速率） |
| 输入 / 输出长度 | 1024 / 256 token（强制输出长度，`ignore_eos`） |
| 请求数 / 随机种子 | 32 / 42（可复现） |
| 预热 | 4–16 个请求，不计入统计 |
| 硬件 | 2-core CPU 沙箱（demo）；真实运行时 `env.json` 记录 GPU 型号、显存、驱动 |
| 服务端指标 | 引擎 `/metrics` + DCGM exporter，15 s 窗口差分 |

demo 矩阵的 8 个变体（5 个 mock 进程）：

```
:8000  vllm-fp16            （基线 + prefix-cache ON 组）
:8001  sglang-fp16          （引擎对比）
:8002  vllm-no-prefix-cache （prefix-cache OFF 组）
:8003  vllm-awq-int4        （量化对比）
:9400  dcgm-exporter        （GPU 指标）
```

> mock 的已知局限（写进报告以免误引）：DCGM 序列是随机游走，**与负载不耦合**；
> 引擎侧序列（队列、KV、prefix 命中、token 计数器）**随负载真实变化**。

## 2. 实验 A · 量化：AWQ int4 vs FP16（@ concurrency=12）

自动报告：`report-A-quantization.md`。两组唯一差异是量化方式，负载完全一致。

**客户端指标（测量窗口内）：**

| 指标 | fp16-conc12 | awq-conc12 | Δ |
|---|---|---|---|
| RPS（实际达成） | 3.45 | 5.22 | **+51.2%** ✅ |
| 输出吞吐 tok/s | 883 | 1,335 | **+51.2%** ✅ |
| TTFT mean / p99 (ms) | 131.4 / 150.0 | 93.8 / 105.1 | **−28.6% / −29.9%** ✅ |
| TPOT mean / p99 (ms) | 11.62 / 12.14 | 7.64 / 7.83 | **−34.3% / −35.5%** ✅ |
| E2E p99 (ms) | 3,230 | 2,100 | −35.0% ✅ |
| 错误率 | 0% | 0% | = |

**服务端指标（同窗口）：**

| 信号 | fp16 | awq | 解读 |
|---|---|---|---|
| KV cache 利用率 | 0.68 | 0.43 | 权重占用变小 → 同显存下 KV 池更大 |
| 引擎 prefill tok/s（窗口差分） | 3,252 | 4,869 | prefill 提速 ≈ TTFT 下降的来源 |
| 引擎 decode tok/s（窗口差分） | 682 | 1,022 | decode 提速 ≈ TPOT 下降的来源 |
| 引擎侧 TTFT mean (ms) | 118.6 | 84.4 | 与客户端差 ≈10 ms = 网络+序列化开销 |
| GPU util % / 功耗 W | 32 / 201 | 89 / 336 | 吞吐上去后 GPU 吃得更满 |

**讲评**：客户端结论（吞吐、延迟）与服务端证据（prefill/decode 速率、KV 占用）
互相印证——这是"可观测"的意义：不只知道快了多少，还知道**为什么**快。
真实场景下这张表还缺一个维度：**输出质量**（量化会掉点），需要另配
eval（如 lm-eval-harness），报告里必须同时给出，否则"快 51%"是半句话。

## 3. 实验 B · Prefix Cache：2×2 对照设计

自动报告：`report-B-prefixcache.md`。这是全套实验里方法论上最重要的一组。

**设计**：缓存开/关 × 共享前缀 0/512 token，四格全测，closed-loop conc=4
（故意不饱和——prefix cache 省的是几十毫秒的 prefill，饱和时排队延迟是秒级，
效应会被淹没，测出来永远是"无差异"，见 `docs/04-experiments.md`）。

| 单元格 | TTFT mean (ms) | TTFT p99 (ms) | 窗口内命中率 | 输出 tok/s |
|---|---|---|---|---|
| cache ON + prefix 0 | 130.2 | 148.8 | **0.000** | 382.7 |
| cache ON + prefix 512 | **67.0** | **77.0** | **0.995** | 394.2 |
| cache OFF + prefix 0 | 132.4 | 150.8 | 0.000 | 385.0 |
| cache OFF + prefix 512 | 135.8 | 156.2 | 0.000 | 384.1 |

**读法**：

* 只有 (ON, 512) 一格 TTFT 减半（−48.5% vs ON/0 格）——这正是缓存应有的行为；
* **负对照成立**：(OFF, 512) vs (OFF, 0) 差 +2.6%，在噪声范围内。
  如果这一格也"变快"，说明共享前缀本身（比如 prompt 更短了）在起作用，
  实验就测错了东西；
* 窗口命中率 0.995 是服务端证据：512 token 前缀在 32 个请求间几乎全命中；
* 吞吐基本不动（394 vs 384 tok/s）：conc=4 时系统是延迟受限而非算力受限，
  **prefix cache 的收益体现在 TTFT/成本，不是吞吐**——这是最常见的误期待。

> 命中率必须用**窗口差分**（`prefix_cache_hits_total` 计数器在测量窗口内的增量），
> 不能用引擎的 `prefix_cache_hit_rate` gauge——那是进程启动以来的累计值，
> 两个 run 打在同一个 server 进程上时几乎相同，A/B 会得出"无差异"的假结论。
> 这一坑与 §7 的事故同源，工具链里已用 `_window` 后缀字段固化。

## 4. 实验 C · 引擎：vLLM vs SGLang（@ concurrency=12）

自动报告：`report-C-engine.md`。同模型同精度，只换引擎。

| 指标 | vllm-fp16 | sglang-fp16 | Δ |
|---|---|---|---|
| RPS | 3.45 | 3.88 | +12.5% ✅ |
| 输出 tok/s | 883 | 993 | +12.5% ✅ |
| TTFT mean / p99 (ms) | 131.4 / 150.0 | 118.9 / 133.3 | −9.5% / −11.2% ✅ |
| TPOT mean / p99 (ms) | 11.62 / 12.14 | 10.37 / 10.72 | −10.8% / −11.7% ✅ |
| KV cache 利用率 | 0.68 | 0.57 | SGLang 内存管理略省（mock 设定） |
| preemptions（窗口） | 0 | **—** | SGLang 不导出该计数器 |
| prefix 命中计数器（窗口） | 0.00 | **—** | 同上（只有累计 gauge） |

**讲评**：

* 表里的 "—" 是**指标缺口**，不是零。跨引擎对比前必须先对表指标名
  （loadgen 的 `EXPECTED_ALIASES` 就是干这个的），缺什么就明说什么，
  不能把 "—" 当 0 读；
* mock 里 12.5% 的差距是脚本设定值；真实结论受模型、长度分布、版本影响很大，
  换任何一个变量都必须重测——这正是把整套负载参数写死进 `meta.json` 的原因。

## 5. 饱和曲线与 knee（拐点）

fp16 基线的 closed-loop 并发扫描（`capacity-plan.md` 自动判定）：

| concurrency | RPS | 输出 tok/s | TTFT p99 (ms) | TPOT p99 (ms) | KV util | queue p99 | SLO |
|---|---|---|---|---|---|---|---|
| 4 | 1.51 | 386 | 149 | 10.0 | 0.27 | 0 | ✅ |
| 8 | 2.75 | 703 | 154 | 10.9 | 0.52 | 0 | ✅ |
| **12** | **3.45** | **883** | **150** | **12.1** | **0.68** | **0** | ✅ **← knee** |
| 16 | 3.48 | 891 | **3,266** | 11.9 | 0.68 | 4 | ❌ |
| 24 | 3.48 | 891 | **3,288** | 11.9 | 0.68 | 12 | ❌ |

* knee 之后**吞吐只多 0.9%，TTFT p99 恶化 22 倍**——过拐点加并发是纯亏；
* 自动诊断：**Admission bound**（`num_requests_waiting` 从 0 跳到 4→12，
  请求在排队等准入槽位），对策是横向扩容而不是调大 `--max-num-seqs`；
* open-loop 速率扫描给出另一个视角：offered 3/4/5 rps 时实际只达成
  2.48/2.68/2.81 rps，loadgen 自动打出 advisory warning：
  **"不要把 offered rate 报告成可支撑速率"**——这三条 warning 是特性，不是缺陷。

AWQ 变体同法测得 knee 在 conc16（7.22 rps，TTFT p99 107 ms），
即量化把单卡拐点容量抬高了 ≈2.1 倍——容量规划应该按**你实际要上的变体**重测。

## 6. 从报告到扩容决策（capacity-plan.md 摘录）

输入：目标峰值 20 rps，SLO = TTFT p99 ≤ 1500 ms 且 TPOT p99 ≤ 50 ms，
余量 30%，$2.5/GPU·h，1 GPU/副本：

| 决策项 | 值 | 推导 |
|---|---|---|
| 单副本 SLO 内 RPS | **3.45** | knee 实测（不是拍脑袋） |
| 需要副本 / GPU 数 | **8 / 8** | 20 × 1.3 ÷ 3.45 = 7.5 → 8 |
| PDB minAvailable | 7 | 滚动更新时不低于 N−1 |
| 单副本在途上限 | `--max-num-seqs 12` | = knee 并发，防止单 pod 过载 |
| HPA | min 7 / max 16，目标 **2.59 rps/pod** | knee 的 75%；另可挂 `vllm:num_requests_waiting > 1` 持续 60 s |
| 成本 | **≈ $0.78 / 1M 输出 token** | $2.5 ÷ (891 tok/s × 3.6) 按峰值吞吐计 |

两个必写的 caveat（模板里已内置）：

* **冷启动**：7B 副本从拉起到可服务要经过权重加载 + CUDA graph 捕获，
  HPA 的 `scaleUp.stabilizationWindowSeconds` 和 readiness probe 的
  `initialDelaySeconds` 必须按实测 `pod_ready_seconds` 设，否则扩容永远迟到；
* **knee 是负载形态的函数**：换输入/输出长度分布，3.45 这个数就作废，
  所以 `meta.json` 把负载形态和结果绑死在一起存档。

## 7. 方法论事故复盘：prefix cache 跨 run 污染

这一节是本项目最有价值的部分——**一次真实的、被工具链抓住的测量事故**。

### 事故经过

第二轮 demo 数据里出现了不可能正确的结果：**所有** run（包括根本没开
prefix cache、prompt 也毫无共享前缀的对照组）窗口命中率都是 **1.0**。
对照组命中率为 1 意味着"缓存了从未出现过的东西"，物理上不成立 → 数据必错。

### 排查

1. 单发 3 个全新 prompt 给 mock：计数器 `queries +3, hits +0` → 引擎统计本身没错；
2. 检查 PromptBuilder：同一 run 内 prompt 互不相同 → 不是 run 内自缓存；
3. 锁定根因：**引擎的 prefix cache 是进程级、跨 run 存活的**，而 loadgen
   用固定 seed（=42）生成 prompt，于是第 N+1 个 run 重放的 prompt 序列与
   第 1 个 run 完全相同——cache 在第一次 run 时就全部焐热了。
   第二轮所有对照组的 hit=1.0 正是第一轮 run 留下的缓存。

### 修复（全部进了工具链，不只是"下次小心点"）

| 防御层 | 实现 |
|---|---|
| 消除根因 | `scripts/flush_cache.sh`：每个 run 前 POST `/reset_prefix_cache`；`run_matrix.sh` 默认 `FLUSH_CACHE=1` 自动调用；真实 vLLM/SGLang 没有该端点时，文档要求**重启引擎进程**再测下一组 |
| mock 支持 | `mock_server.py` 实现 `/reset_prefix_cache`、`/flush_cache` 端点 |
| 事中告警 | loadgen 运行结束自动做 8 项效度检查写入 `summary["warnings"]`，其中包括本事故的探测器：**"窗口命中率 > 0.3 但 shared_prefix_tokens = 0" → 立即标记疑似污染**；另有"达成 rps 远低于 offered rps 时禁止把 offered 报成容量"等 |
| 事后校验 | `tests/verify_run.py` 汇总展示所有 warning（本次 demo：23/23 valid + 3 条 advisory，全部正确） |
| 回归测试 | `tests/test_harness.py` 新增 2 个用例锁死该行为（pytest 23/23） |

### 教训（写报告时值得原样引用）

1. **对照组出现"物理上不可能的值"是免费的报警器**——hit=1.0 出现在
   no-cache 组，比任何显著性检验都直接。看到好得离谱的数，先怀疑测量；
2. **一切进程级状态（cache、内存池、JIT、页缓存）都是跨 run 变量**。
   A/B 之间要么 flush，要么重启，要么随机化顺序并取多轮——"跑完 A 直接跑 B"
   是隐性顺序效应温床；
3. **固定 seed 是可复现性和状态污染的双刃剑**：它保证了负载可复现，
   也保证了污染可复现。两者必须配套处理；
4. 把检查**固化进代码**（warnings + 回归测试），不要固化进记忆。

## 8. 复现与自查清单

```bash
make dev-up        # 起 5 个 mock（真实环境换成 vLLM/SGLang 部署）
make demo          # 全矩阵 23 run + 报告 + 校验（内部每 run 前自动 flush cache）
python3 tests/verify_run.py bench/results/demo   # 期望: 23/23 valid
```

发布任何一份报告前的自查（工具已覆盖前 5 项）：

- [ ] 每个 run 的 `warnings` 为空，或每条 warning 都在正文中被解释；
- [ ] 对照组没有出现"不可能的值"（如 no-cache 组命中 > 0）；
- [ ] A/B 两组 `meta.json` 中除被测变量外全部字段一致（compare.py 会校验并拒绝）；
- [ ] 引擎侧计数器用窗口差分，不用累计 gauge；
- [ ] 达成 rps < offered rps 时，容量结论以达成值为准；
- [ ] p99 的样本量足够（32 请求的 p99 ≈ 最大值，正式报告应 ≥ 500 请求）；
- [ ] 量化对比附带质量评估，引擎对比注明指标缺口；
- [ ] 硬件、驱动、引擎版本、负载形态、seed 全部可追溯（env.json/meta.json）。

---

*上一篇：[06 · 容量规划](06-capacity.md) ｜ 返回：[项目 README](../README.md)*
