# 04 · 对照实验设计

需求 4 要求"比较至少两个变量"。本项目实现了**三组**，并且把方法论写死在工具里：
`compare.py` 会在两次运行不可比时主动报警，而不是照样出一张好看的表。

---

## 0. 所有实验共用的规则

### 0.1 一次只动一个变量

"其它一切不变"的具体清单（`compare.py` 的 `env_guard` 会检查前 8 项）：

```
input_len, output_len, mode, request_rate, max_concurrency, num_requests,
endpoint, force_output_len, prompt_builder_method(seed/tokenizer), GPU 清单
```

任何一项不同，报告顶部就会出现 ⚠️ 警告。这不是形式主义：
把 `input_len=1024` 的 FP16 和 `input_len=2048` 的 AWQ 放在一起，
你会"发现"量化让 TTFT 变差了 2 倍。

### 0.2 先测噪声底

```bash
for i in 1 2 3; do make bench TAG=noise SUITE=baseline N_REQ=500; sleep 30; done
python3 bench/compare.py bench/results --baseline-tag noise
```

三次同配置之间的摆动就是噪声底。**小于它的差异不能写进结论。**

### 0.3 每个变体必须重启引擎，且必须过 `wait_ready.sh`

不重启的后果见 §2.3（累计 gauge）。不过 `wait_ready.sh` 的后果是
第一个变体的 TTFT 里混进模型加载时间。

### 0.4 记录被研究变量本身

`serve_*.sh` 会把解析后的完整命令行写进 `runs/<ts>-serve-cmd.txt`。
两个变体的这两个文件 `diff` 出来，**应该只差你研究的那一行**。
把这个 diff 贴进报告，比任何文字描述都有说服力。

---

## 1. 实验 A：量化前后（FP16 vs AWQ int4）

### 1.1 假设与预期方向

| 维度 | 预期 | 机理 |
|---|---|---|
| 权重显存 | ↓ ~4x（FP16→int4） | 14.2 GiB → ~4.3 GiB |
| 可用 KV | ↑ | 省下的显存全部变成 KV 池 ⇒ **并发上限提高** |
| decode TPOT | ↓ | decode 是带宽受限，搬运字节数减少 4x ⇒ 单流 tok/s 上限提高 |
| prefill TTFT | ↓ 或 ≈ | prefill 是计算受限；int4 需要反量化，收益取决于 kernel 质量 |
| 峰值吞吐 | ↑ | 主要来自更大 batch，而不是单流更快 |
| **质量** | **↓（幅度未知）** | ⚠️ 必须独立评测，见 §1.5 |

### 1.2 执行

```bash
# 变体 1：FP16
MODEL=/models/Qwen2.5-7B-Instruct MAX_NUM_SEQS=256 GPU_MEM_UTIL=0.90 \
  ./scripts/serve_vllm.sh
./scripts/wait_ready.sh http://127.0.0.1:8000 900
make bench TAG=vllm-fp16 SUITE=all URL=http://127.0.0.1:8000
pkill -f vllm.entrypoints

# 变体 2：AWQ int4 —— 只改 QUANT 和 MODEL
MODEL=/models/Qwen2.5-7B-Instruct-AWQ QUANT=awq MAX_NUM_SEQS=256 GPU_MEM_UTIL=0.90 \
  ./scripts/serve_vllm.sh
./scripts/wait_ready.sh http://127.0.0.1:8000 900
make bench TAG=vllm-awq SUITE=all URL=http://127.0.0.1:8000

# 对比（同一并发点 + 同一速率点）
python3 bench/compare.py bench/results/*vllm-fp16-conc12 bench/results/*vllm-awq-conc12 \
    --baseline-tag fp16 --title "量化：FP16 vs AWQ int4 @ conc 12" \
    --out bench/results/report-A-quantization.md
python3 bench/capacity.py bench/results --tag-filter vllm-awq-conc
```

### 1.3 必须看的三个衍生量

单看"吞吐 +35%"会得出错误结论。要一起看：

1. **拐点是否右移。** 量化的主要收益是 KV 变多 ⇒ 能承接更高并发。
   比较两组 `capacity.py` 的 knee concurrency 和 knee rps，而不只是同一点的吞吐。
   用 `capacity_model.py` 交叉验证：
   ```bash
   python3 bench/capacity_model.py --preset Qwen2.5-7B --kv-gb 40 --context 8192
   #   → 91 条并发序列（FP16 权重）
   python3 bench/capacity_model.py --preset Qwen2.5-7B --weight-dtype int4 --kv-gb 50
   #   → 同样 91 条（KV dtype 没变，所以并发上限不变！）
   ```
   ⚠️ 注意：`--kv-cache-dtype fp8` 才直接把并发翻倍；权重量化是通过
   "腾出显存给 KV" 间接起作用，而 vLLM 的 `gpu-memory-utilization` 是
   一个总比例，所以省下的权重显存**确实**会进 KV 池。
   要验证这一点，看 `vllm:cache_config_info` 或启动日志里的
   `# GPU blocks: N`，N 应该显著变大。

2. **TPOT 是否接近新的 roofline。**
   ```bash
   python3 bench/capacity_model.py --preset Qwen2.5-7B --weight-dtype int4 \
       --gpu A100-80G --measured-tpot-ms <实测 conc=1 的 TPOT>
   # int4 权重 3.8GiB / 1631GB/s ≈ 2.3ms 理论下限
   ```
   如果实测远高于下限，说明反量化 kernel 是瓶颈（不同量化格式差异很大：
   AWQ/GPTQ 的 marlin kernel 好，bitsandbytes 的差）。

3. **成本，不是速度。** 真正的决策量是 `$/1M output tokens`：
   ```bash
   python3 bench/capacity.py bench/results --tag-filter vllm-awq-conc \
       --gpu-hours-cost 2.5 --target-rps 40
   ```
   如果量化让单卡吞吐 +35%，那么每 1M token 的成本 −26%。
   但如果质量下降需要"用更大 prompt 补偿"（few-shot 变多），收益会被吃掉。

### 1.4 demo（mock）数据里的形态

`bench/results/demo/report-A-quantization.md` 展示的是合成数据，
但形态是对的：AWQ 变体 knee 从 conc12 移到 conc16–24，
同并发下 TPOT −34%、吞吐 +51%。**真机上不会有这么大**，
因为 mock 的 `tok-ms` 是我手写的。真实 A100 上 7B AWQ 相对 FP16
的 decode 收益通常在 1.2–1.6x，且高度依赖 kernel。

### 1.5 质量评测（不能省）

性能结论必须与质量声明同时出现，否则"量化更快"是一个不完整的命题。
最低成本的三件套：

```bash
pip install lm-eval
# 1. 通用能力
lm_eval --model hf --model_args pretrained=/models/Qwen2.5-7B-Instruct-AWQ,quantize=awq \
        --tasks gsm8k,mmlu,humaneval --num_fewshot 5 --batch_size 8
# 2. 你自己的业务集（最重要）：200–500 条真实请求，
#    用 FP16 的输出作为参考答案，量化版跑一遍，算 exact match / 编辑距离 / LLM-as-judge
# 3. 长文本与多轮：量化对长上下文的退化通常比对单轮 QA 更严重
```

报告里 `Quality caveat` 字段必须填，模板见 §4。

---

## 2. 实验 B：Prefix Cache 开关

### 2.1 假设

Prefix caching（vLLM）/ RadixAttention（SGLang）复用**公共前缀**的 KV。
收益只在一个条件下存在：**请求之间真的共享前缀**。

| 场景 | 共享前缀 | 预期收益 |
|---|---|---|
| 统一 system prompt（客服、Agent） | 200–2000 tok | TTFT 显著下降，命中率 60–95% |
| Few-shot 模板固定 | 500–4000 tok | 同上，收益更大 |
| RAG，检索结果在前 | 0（检索内容在 prompt 头部） | **几乎为零** ⇒ 把 system prompt 放到最前面 |
| 长多轮对话 | 整个历史 | 巨大，这是 RadixAttention 的主场 |
| 完全随机的独立请求 | 0 | 零收益，且有少量哈希/查找开销 |

### 2.2 执行：负载设计决定实验成败

```bash
# 变体 1：prefix caching ON
PREFIX_CACHING=1 MODEL=... ./scripts/serve_vllm.sh
./scripts/wait_ready.sh http://127.0.0.1:8000

# 变体 2：OFF
PREFIX_CACHING=0 MODEL=... ./scripts/serve_vllm.sh     # → --no-enable-prefix-caching
./scripts/wait_ready.sh http://127.0.0.1:8000

# 每个变体跑两个负载点：无共享前缀 / 512 token 共享前缀
make bench TAG=pc-on  SUITE=prefix URL=http://127.0.0.1:8000 PREFIX_TOKENS=512
make bench TAG=pc-off SUITE=prefix URL=http://127.0.0.1:8002 PREFIX_TOKENS=512
```

这是一个 **2×2 设计**（cache on/off × prefix 0/512），
比单纯 on/off 强得多，因为它能区分"cache 有效"和"负载本来就快"：

```
                     prefix=0        prefix=512      差值
cache ON             A               B               B−A  ← cache 的真实收益
cache OFF            C               D               D−C  ← 应该 ≈ 0（对照组）
```

`D−C ≈ 0` 是**阴性对照**：它证明"共享前缀本身"不会让请求变快，
变快只能来自 cache。少了这一格，你无法排除"prefix512 的 prompt 恰好更短"这类解释。

### 2.3 两个会让实验失败的坑

**坑 1：在饱和状态下测。**
Prefix cache 省的是 prefill 时间（几十毫秒）。
如果 TTFT 是 3000ms 而其中 2900ms 是排队，55% 的 prefill 折扣完全被淹没。
→ `run_matrix.sh` 的 `prefix` suite 强制用 **closed-loop + 低并发**
（`PREFIX_CONC=4`），让 TTFT ≈ prefill。
这一条在 demo 数据里可以直接看到：
饱和态测出 on/off 差异 <2%，非饱和态测出 TTFT p99 **−47%**。

**坑 2：用累计 gauge 判断命中率。** 见 `docs/03-metrics.md` §4.1。
用 `summary.server.window.prefix_cache_hit_ratio_window`。

**坑 3（生产环境特有）：多副本 + 轮询路由。**
N 个副本各自维护自己的 prefix cache，轮询会把同前缀请求打散到 N 份，
命中率掉到 1/N。要么上 prefix-aware 路由（如 llm-d / AIBrix 的
KV-aware routing，或 SGLang router 的 cache-aware 模式），
要么接受这个损失并在容量模型里体现。
告警 `PrefixCacheHitRateCollapsed` 就是为了抓这个。

### 2.4 该看的指标

| 指标 | 期望 |
|---|---|
| `TTFT mean / p99` | ↓（主要收益） |
| `window.prefix_cache_hit_ratio_window` | 与共享前缀占比一致（512/1024 ⇒ ~0.5） |
| `prompt_tokens_total_rate`（引擎侧） | ON 时**不应**下降（vLLM 仍统计全部 prompt token），但 prefill 实际计算量下降 |
| `output_tokens_per_sec` | 轻微 ↑（省下的算力给了 decode） |
| `kv_cache_util` | ON 时**可能更高**（缓存的前缀占着块）—— 这是成本，要如实报告 |
| `preemptions_total` | ON 时**可能上升**（KV 被缓存占用）⇒ 这是 prefix cache 的真实代价 |

最后两行很重要：prefix cache 不是纯收益，它用 KV 空间换 prefill 时间。
在 KV 已经很紧张的场景（长上下文、高并发）它可能是净负收益。

---

## 3. 实验 C：vLLM vs SGLang

### 3.1 flag 映射（`serve_sglang.sh` 里有完整表）

| 语义 | vLLM | SGLang |
|---|---|---|
| 显存比例 | `--gpu-memory-utilization 0.90` | `--mem-fraction-static 0.90` |
| 上下文 | `--max-model-len` | `--context-length` |
| batch 上限 | `--max-num-seqs` | `--max-running-requests` |
| prefix cache | `--enable-prefix-caching` / `--no-...` | 默认开 / `--disable-radix-cache` |
| chunked prefill | `--enable-chunked-prefill` | `--chunked-prefill-size N` |
| 关 CUDA graph | `--enforce-eager` | `--disable-cuda-graph` |
| KV dtype | `--kv-cache-dtype fp8` | `--kv-cache-dtype fp8_e5m2` |
| 指标 | 默认开 | **需要 `--enable-metrics`** |
| 就绪探测 | `/health` | `/health` + `/health_generate`（真推理） |
| 默认端口 | 8000 | 30000 |

`loadgen.py` 的别名表把 `sglang:num_running_reqs` / `token_usage` /
`num_queue_reqs` / `cache_hit_rate` / `inter_token_latency_seconds`
映射到与 vLLM 相同的别名，**所以同一份对比报告里两边的服务端指标都非空**。
这是能对比的前提；否则 SGLang 那一列会全是 `—`，
看起来像"SGLang 没有队列"，实际是指标名不同。

### 3.2 架构差异（决定了你应该期待什么）

| | vLLM | SGLang |
|---|---|---|
| KV 管理 | PagedAttention，固定 block | RadixAttention，基数树自动前缀共享 |
| 前缀复用 | 需显式开启，块级哈希 | 默认开启，树结构，跨请求自动 |
| 调度 | FCFS + 抢占重算 | 类似，另有 cache-aware 调度 |
| 结构化输出 | guided decoding（outlines/xgrammar） | 自研压缩有限状态机，通常更快 |
| 多轮/Agent | 好 | **通常显著更好**（RadixAttention 命中整个历史） |
| 指标成熟度 | 高，histogram 齐全 | 在快速追赶，部分 histogram 缺失 |

**因此实验设计必须匹配工作负载**：
- 独立单轮、无共享前缀 → 两者应该接近，差异来自 kernel 与调度开销。
- 长 system prompt / few-shot → SGLang 默认占优（除非 vLLM 也开 prefix caching）。
- 多轮 Agent → SGLang 的 RadixAttention 命中率会显著更高。

如果你的业务是第三种，却用第一种负载做对比，结论对生产无效。

### 3.3 执行

```bash
MODEL=... ./scripts/serve_vllm.sh                                   # :8000
MODEL=... PORT=30000 ./scripts/serve_sglang.sh                      # :30000
./scripts/wait_ready.sh http://127.0.0.1:30000 900

make bench TAG=vllm    SUITE=all URL=http://127.0.0.1:8000
make bench TAG=sglang  SUITE=all URL=http://127.0.0.1:30000

python3 bench/compare.py bench/results/*vllm-conc12 bench/results/*sglang-conc12 \
    --baseline-tag vllm --title "引擎：vLLM vs SGLang @ conc 12" \
    --out bench/results/report-C-engine.md
```

⚠️ **两个引擎不能同时占用同一张卡。** 要么两张卡（并确认两张卡同型号同频率，
`record_env.py` 会记录，`env_guard` 会检查），要么串行跑。

### 3.4 除性能之外必须记录的差异

- **API 兼容性缺口**：某些 OpenAI 参数一方支持另一方不支持
  （`logprobs`、`response_format`、`n>1`、tool calling 的细节）。
  跑一遍你的真实客户端代码，把报错记下来。
- **运维成熟度**：滚动升级行为、OOM 恢复、日志可读性、社区响应速度。
- **版本锁定**：两者的 API 与指标名都还在快速变动。
  `meta.json` 里记录了 `packages.vllm` / `packages.sglang` 的版本，
  报告里必须写清楚，否则半年后无法复现。

---

## 4. 结论模板（`compare.py` 会把它附在每份报告末尾）

```text
Variable under test : AWQ int4 vs FP16, Qwen2.5-7B-Instruct, A100-80G x1
Workload            : in=1024 out=256, closed-loop, conc=12, 500 req, 3 repeats
Noise floor         : ±1.8% on out tok/s (from 3x identical FP16 runs)

Throughput effect   : out tok/s 3,180 → 4,410  (+38.7%, σ=1.4%)   [>> noise floor]
                      knee concurrency 12 → 20 (KV blocks 4,100 → 9,800)
Latency effect      : TPOT p99 21.4 → 14.1 ms (−34%)
                      TTFT p99 180 → 165 ms (−8%, prefill 收益小，符合预期)
Cost effect         : $/1M out tokens 2.51 → 1.81 (−28%) @ $2.5/GPU-hour

Quality caveat      : MMLU 5-shot 0.731 → 0.718 (−1.3pt), GSM8K 0.812 → 0.781 (−3.1pt)
                      业务集 500 条：exact-match 0.86 → 0.81
                      ⚠️ 数学推理退化明显，若业务含计算类任务需人工复核
Not measured        : 长上下文(>16k)行为、多轮对话、并发>64、TP=2、
                      FP8 KV cache 叠加量化的效果

Decision            : adopt-with-condition
                      条件：仅用于摘要/改写类流量；计算类流量保留 FP16 副本，
                      按 route 分流。上线前需完成业务集 5000 条评测。
```

四个必填字段的用意：

- **Noise floor**：没有它，+3% 到底是收益还是噪声无从判断。
- **Quality caveat**：强制性能结论与质量结论同框出现。
- **Not measured**：明确结论的边界，防止被过度引用。
  这是工程报告里最常被省略、也最有价值的一栏。
- **Decision**：必须落到 adopt / reject / adopt-with-condition 之一。
  "有待观察"不是决策。

---

## 5. 还可以做（同一套 harness 直接支持）

| 变量 | 怎么跑 | 关注指标 |
|---|---|---|
| chunked prefill 开关 | `CHUNKED_PREFILL=0/1` | **ITL p99**（不是 TTFT）：长 prefill 不再阻塞 decode |
| `max-num-seqs` 扫描 | `MAX_NUM_SEQS=32/64/128/256` | knee 位置、preemptions、TPOT 尾部 |
| KV cache dtype | `KV_CACHE_DTYPE=fp8` | 并发上限、质量、`kv_cache_util` |
| TP=1 vs 2 | `TP=2` | TTFT ↓（算力翻倍）vs TPOT ↑（all-reduce 开销）；看 `NVLINK_TX_BYTES` |
| CUDA graph 开关 | `CUDA_GRAPHS=0` | 小 batch 下 TPOT 差异最大 |
| 输入长度扫描 | `--input-len 256/1k/4k/16k/32k` | TTFT 是否线性；KV 何时成为上限 |
| 输出长度扫描 | `--output-len 64/256/1024/4096` | 吞吐 vs 延迟的权衡曲线 |
| chat vs completions | `--endpoint` | chat 模板的额外 token 开销（常达 5–15%） |
| 投机解码 | `--speculative-model ...` | TPOT ↓ 但 GPU 占用 ↑；接受率是关键中间指标 |
