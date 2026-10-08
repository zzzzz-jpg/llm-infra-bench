# 01 · 架构与设计决策

## 1. 系统全景

```
                        ┌─────────────────────────────────────────────────────┐
                        │                   Kubernetes 集群                    │
                        │                                                     │
   客户端 / 业务          │   ┌───────────┐      ┌──────────────────────────┐   │
  ───────────────────────┼──▶│  Ingress  │─────▶│  Service vllm (ClusterIP) │  │
   OpenAI 兼容 SDK        │   │  (nginx)  │      └───────────┬──────────────┘   │
   /v1/chat/completions   │   │ SSE 免缓冲 │                  │                  │
   stream=true            │   │ 超时 600s │      ┌───────────▼──────────────┐   │
                        │   │ 限流 40rps│      │ Pod: vllm  (GPU x1)      │   │
                        │   └───────────┘      │  startupProbe  → /health │   │
                        │                      │  readinessProbe→ /health │   │
                        │                      │  livenessProbe → /health │   │
                        │                      │  preStop: sleep 20       │   │
                        │                      │  /dev/shm 16Gi (Memory)  │   │
                        │                      │  PVC /models (RWX)       │   │
                        │                      └───┬──────────────┬───────┘   │
                        │                          │ :8000/metrics│ GPU        │
                        │   ┌──────────────────────▼──┐   ┌───────▼─────────┐ │
                        │   │ Prometheus              │◀──│ dcgm-exporter   │ │
                        │   │  + recording rules      │   │ (DaemonSet)     │ │
                        │   │  + alerts.yml (20 条)    │   └─────────────────┘ │
                        │   └──┬───────────┬──────────┘                       │
                        │      │           │ custom.metrics.k8s.io            │
                        │   ┌──▼──────┐  ┌─▼──────────────┐                   │
                        │   │Grafana  │  │prometheus-     │                   │
                        │   │23 面板   │  │adapter → HPA   │                   │
                        │   └─────────┘  └─┬──────────────┘                   │
                        │                  │ 按队列深度 / 每副本 rps 扩缩        │
                        │              ┌───▼────────────┐                     │
                        │              │ HPA vllm 2→8   │                     │
                        │              │ PDB minAvail 1 │                     │
                        │              └────────────────┘                     │
                        └─────────────────────────────────────────────────────┘

  ┌──────────────── 测量平面（与被测系统解耦）──────────────────────────────┐
  │                                                                       │
  │  bench/loadgen.py ──SSE──▶ Service / Pod                              │
  │        │                    │                                         │
  │        │  客户端时间戳        └──HTTP──▶ /metrics (engine)              │
  │        ▼                                /metrics (dcgm)               │
  │  results/<ts>-<tag>/                                                  │
  │    summary.json   ← 聚合：TTFT/TPOT/ITL/E2E/P99、吞吐、goodput、错误率  │
  │    requests.csv   ← 逐请求原始记录（可重算任何统计量）                    │
  │    timeseries.csv ← 1s 粒度：队列、running、KV、prefix、GPU、功耗        │
  │    meta.json      ← 完整 config + env.json 硬件快照 + git commit        │
  │        │                                                              │
  │        ├──▶ bench/compare.py   → A/B 报告（含可比性守卫）                │
  │        └──▶ bench/capacity.py  → 拐点 → 副本数 → HPA 参数 → K8s 清单     │
  │              bench/capacity_model.py → KV 预算 / roofline 交叉校验      │
  └───────────────────────────────────────────────────────────────────────┘
```

**关键分离：测量平面独立于服务平面。** 压测器不依赖集群里的任何东西，
它只要能 HTTP 访问到 endpoint 和 `/metrics`。这让同一套工具可以测
裸机、docker-compose、K8s Service、K8s 单 Pod（用 headless service 直连）四种目标，
而对比结论仍然成立。

## 2. 端口与进程

| 端口 | 进程 | 暴露 | 备注 |
|---|---|---|---|
| 8000 | vLLM OpenAI API | `/v1/*`, `/health`, `/metrics` | 指标与 API 同端口，**不要**把 `/metrics` 挂到公网 Ingress |
| 30000 | SGLang OpenAI API | `/v1/*`, `/health`, `/health_generate`, `/metrics` | 需 `--enable-metrics` |
| 9400 | dcgm-exporter | `/metrics` | DaemonSet，每个 GPU 节点一份 |
| 9090 | Prometheus | — | 集群内 |
| 9100 | node-exporter | `/metrics` | 判断压测机本身是否成为瓶颈 |
| 8000–8003, 9400 | `bench/mock_server.py` | 同上 | CPU-only 模拟，仅用于流水线验证 |

## 3. 五个设计决策（以及为什么）

### 3.1 延迟在客户端测，不在服务端测

`TTFT`、`ITL`、`TPOT` 全部由 `loadgen.py` 从 SSE chunk 的**到达时间戳**推导。
引擎自己上报的 `vllm:time_to_first_token_seconds` 是**引擎内部**的时间，
不包含：网络、代理缓冲、序列化、事件循环排队。用户感受的是前者。

两者都采集，差值本身就是信息：如果客户端 TTFT ≫ 引擎 TTFT，
问题在网关（99% 是 `proxy_buffering on`）而不是模型。

### 3.2 每个 run 都是自描述的

`meta.json` 里存了完整的 `Config`（包括 seed、chars_per_token、warmup 数量）、
`env.json` 硬件快照、git commit、启动时间。
后果是：**没有"我记得那次是怎么跑的"这种事**。
`compare.py` 会读两份 meta 做可比性检查，任何影响结论的维度不一致都会
在报告顶部生成 ⚠️ 警告（比如一次 `input_len=1024`、另一次 `2048`，
或者两次 GPU 型号不同）。

### 3.3 抓取按"族"隔离（strict_kind_filter）

引擎端点只允许贡献 `vllm:*` / `sglang:*` 别名，GPU 端点只允许贡献 `DCGM_*`。
这是被真实 bug 逼出来的：如果两个 exporter 都暴露同名指标，
后抓的会**静默覆盖**先抓的值，你会得到一整列 0 而毫不知情。
`tests/test_kind_filter_prevents_endpoint_clobbering` 钉住了这个行为。

同一个 `INTERESTING` 映射表还把 SGLang 的命名
（`sglang:num_running_reqs` / `sglang:token_usage` / `sglang:num_queue_reqs`）
归一到 vLLM 的别名上 —— 这是"一套 harness 对比两个引擎"的前提，
否则对比报告里一半的格子是空的。

### 3.4 指标名会变，所以用正则而不是硬编码

vLLM 的 flag 拼写在版本间迁移过（`--enable-prefix-caching` ↔ `--no-enable-prefix-caching`），
指标名也变过（`vllm:gpu_cache_usage_perc`、`vllm:prefix_cache_hit_rate` vs
`vllm:gpu_prefix_cache_hit_rate`）。`serve_vllm.sh` 会先 `--help` 探测再决定用哪个拼写，
`loadgen.py` 的别名表用正则同时兼容多种写法。
**结果：升级引擎版本不会让 harness 静默失效。**

### 3.5 失败的 run 不能被当成数据

- warmup 全部失败 → `loadgen.py` 直接退出码 2，不写任何结果文件。
- 断路器：一个不存在的 exporter 连续失败 5 次后被禁用并记录到
  `summary.server.disabled_endpoints`，而不是刷 1000 条错误。
- `tests/verify_run.py` 对每个 run 做退化检查：输出长度是否真的被固定、
  token 数是否来自 `usage` 而不是 chunk 计数、百分位是否单调、
  闭环并发是否被突破、服务端样本是否为 0。
  `make smoke` 会跑它，CI 里它是门禁。

## 4. 一次完整实验的数据流

```
serve_vllm.sh  ──▶ runs/<ts>-serve-cmd.txt      （解析后的完整命令行，可复制粘贴重跑）
                        │
wait_ready.sh  ──▶ 真推理探测通过
                        │
record_env.py  ──▶ results/env.json             （GPU/CPU/驱动/版本/cgroup 限额）
                        │
run_matrix.sh  ──▶ loadgen.py × N 个参数点
                        │            ├─▶ requests.csv    逐请求
                        │            ├─▶ timeseries.csv  逐秒服务端
                        │            └─▶ summary.json    聚合
                        │
compare.py     ──▶ report-<tag>.md              （可比性守卫 + delta 表 + 结论模板）
capacity.py    ──▶ capacity-<tag>.md            （拐点 + 副本数 + HPA yaml 片段）
capacity_model ──▶ KV 预算 / roofline 交叉校验   （实测值是否物理上说得通）
                        │
k8s/base/40-hpa.yaml  ◀─┘                       （把测出来的数字填进集群）
```

## 5. 什么**不在**这个仓库里（以及为什么）

- **质量评测**（perplexity / 任务准确率）。量化实验只测性能是不够的，
  但质量评测是另一个项目（lm-evaluation-harness / opencompass）。
  本仓库的做法是在报告模板里**强制留一个 "Quality caveat" 字段**，
  让性能结论无法脱离质量声明单独存在。见 `docs/04-experiments.md`。
- **Prefill/Decode 分离部署**（如 vLLM 的 disaggregated prefill、
  Dynamo、Mooncake）。架构上很有价值，但它需要多组件编排，
  与本项目"先把单副本的量测做对"的目标不同层次。
  `docs/06-capacity.md` 给了判断你是否需要它的量化标准。
- **多租户 / 计费 / 配额**。Ingress 里有按连接数和 rps 的限流作为起点。
