# 可压测、可观测、可扩容的开源大模型推理服务

一套**可复现**的 7B–8B 级指令模型推理服务工程：vLLM/SGLang 部署 → 固定负载压测 →
全链路指标采集 → 变量对照实验 → Kubernetes 生产化（探针 / 限额 / 告警 / 容量建议）。

这个仓库的重点不是"跑一次压测"，而是**让每一个数字都可追溯、可复现、可被质疑**：
每次运行都自带完整配置快照与硬件清单，对比工具会在两次运行不可比时主动报错，
容量建议直接由实测拐点推导，而不是拍脑袋。

---

## 需求 → 交付物对照

| # | 需求 | 交付物 | 状态 |
|---|---|---|---|
| 1 | vLLM 部署 7B–8B 指令模型，暴露 OpenAI 兼容接口 | [`scripts/serve_vllm.sh`](scripts/serve_vllm.sh) · [`scripts/serve_sglang.sh`](scripts/serve_sglang.sh) · [`scripts/wait_ready.sh`](scripts/wait_ready.sh) · [`k8s/base/20-deployment.yaml`](k8s/base/20-deployment.yaml) | ✅ |
| 2 | 固定负载：输入长度 / 输出长度 / 并发 / 请求速率 / 硬件信息全部记录 | [`bench/profiles/*.yaml`](bench/profiles) · [`scripts/record_env.py`](scripts/record_env.py) → 每次 run 的 `meta.json` | ✅ |
| 3 | 采集 TTFT / TPOT / P99 / 吞吐 / 显存 / GPU 利用率 / 队列 / 错误率 | [`bench/loadgen.py`](bench/loadgen.py)（客户端计时 + 服务端抓取）· [`observability/`](observability)（Prometheus / DCGM / Grafana） | ✅ |
| 4 | 至少两个变量的对照实验 | [`docs/04-experiments.md`](docs/04-experiments.md)：**量化 (FP16 vs AWQ)**、**Prefix Cache 开关**、**vLLM vs SGLang** 三组，含 [`bench/compare.py`](bench/compare.py) | ✅ |
| 5 | 部署到 K8s + 探针 / 资源限额 / 告警 / 容量建议 | [`k8s/`](k8s)（23 个资源，kubeconform 校验通过）· [`observability/prometheus/alerts.yml`](observability/prometheus/alerts.yml)（20 条告警）· [`bench/capacity.py`](bench/capacity.py) + [`bench/capacity_model.py`](bench/capacity_model.py) | ✅ |

---

## 五分钟上手

### A. 没有 GPU？先验证整条流水线（约 5 分钟，纯 CPU）

```bash
make install
make dev-up          # 起 5 个 mock：vLLM / SGLang / 无PrefixCache / AWQ / DCGM
make test            # 23 个正确性测试：指标语义、负载模型、抓取、污染防御
make lint            # kustomize build + kubeconform 校验全部 K8s 清单
./scripts/demo_mock_matrix.sh    # 跑完整对照矩阵，产出全部报告
make dev-down
```

`demo_mock_matrix.sh` 会用**合成延迟**跑完 A/B/C 三组实验并生成
`bench/results/demo/report-{A,B,C}-*.md` + `capacity-plan.md`。
数字是假的（mock 服务器），流水线是真的 —— 用它先验证你对 harness 的改动，
再去花 GPU 时间。

### B. 有 GPU：跑真实基准

```bash
# 1. 装依赖（在你的推理机上）
pip install vllm          # 或 pip install "sglang[all]"
pip install -r requirements.txt

# 2. 起服务（脚本会把解析后的完整命令行写进 runs/*-serve-cmd.txt）
MODEL=Qwen/Qwen2.5-7B-Instruct \
MAX_NUM_SEQS=256 GPU_MEM_UTIL=0.90 \
  ./scripts/serve_vllm.sh

# 3. 等它真的能推理（/health 200 ≠ 能服务，见脚本注释）
./scripts/wait_ready.sh http://127.0.0.1:8000 900

# 4. 记录硬件与软件环境（会被自动嵌进每个 run 的 meta.json）
python3 scripts/record_env.py --out bench/results/env.json \
    --probe-url http://127.0.0.1:8000

# 5. 跑固定负载 + 两个扫描 + prefix 对照
make bench TAG=vllm-fp16 SUITE=all URL=http://127.0.0.1:8000 \
     TOKENIZER=Qwen/Qwen2.5-7B-Instruct

# 6. 出报告与容量建议
make report   TAG=vllm-fp16
make capacity TAG=vllm-fp16 TARGET_RPS=40 GPU_COST=2.5

# 7. 换变量，重复 2–6，然后对比
QUANT=awq MODEL=/models/Qwen2.5-7B-Instruct-AWQ ./scripts/serve_vllm.sh
make bench TAG=vllm-awq SUITE=all URL=http://127.0.0.1:8000
python3 bench/compare.py bench/results --baseline-tag vllm-fp16 \
    --title "FP16 vs AWQ" --out bench/results/report-quant.md
```

### C. 上 Kubernetes

```bash
kubectl apply -f observability/dcgm/dcgm-exporter.yaml   # GPU 指标
kubectl apply -k k8s/base                                # 服务（或 overlays/prod）
kubectl -n llm-inference rollout status deploy/vllm --timeout=20m
kubectl apply -f k8s/base/60-bench-job.yaml              # 集群内压测
make deploy-prod                                         # 生产 overlay
```

---

## 目录结构

```
llm-infra-bench/
├── bench/                      # 压测与分析（本项目的心脏）
│   ├── loadgen.py              # 固定负载生成器 + 客户端指标 + 服务端抓取
│   ├── compare.py              # A/B 对比报告，含"可比性守卫"
│   ├── capacity.py             # 实测 → 拐点 → 副本数 → HPA 参数
│   ├── capacity_model.py       # 第一性原理校验：KV 预算 + decode roofline
│   ├── mock_server.py          # CPU-only 的 OpenAI 兼容模拟引擎（CI/无 GPU 用）
│   ├── profiles/               # 固定负载定义 ×5（baseline / saturation /
│   │                           #   prefix-cache / production-mix / longcontext）
│   └── results/                # 每次 run 一个自描述目录
│       └── <ts>-<tag>/
│           ├── summary.json    # 聚合指标（TTFT/TPOT/P99/吞吐/错误率/服务端）
│           ├── requests.csv    # 逐请求原始数据（含 ITL 数组）
│           ├── timeseries.csv  # 1s 粒度的队列/KV/GPU 时间序列
│           └── meta.json       # 完整配置 + 硬件清单 + git commit
├── scripts/
│   ├── serve_vllm.sh           # 所有 A/B 旋钮都是环境变量
│   ├── serve_sglang.sh         # 同一契约，另一引擎（含 flag 映射表）
│   ├── wait_ready.sh           # 真·就绪探测（发一次真推理）
│   ├── record_env.py           # GPU/CPU/驱动/版本/cgroup 限额快照
│   ├── run_matrix.sh           # 一次跑完一个 suite（每 run 前自动 flush prefix cache）
│   ├── flush_cache.sh          # 清空引擎 prefix cache，杜绝跨 run 污染
│   ├── demo_mock_matrix.sh     # 无 GPU 全流程演练
│   ├── dev_up.sh / dev_down.sh # mock 拓扑起停
│   └── lint_k8s.sh             # kustomize build + kubeconform
├── observability/
│   ├── prometheus/             # prometheus.yml / alerts.yml / recording-rules.yml
│   │                           # / prometheus-adapter-config.yml（HPA 用）
│   ├── dcgm/                   # dcgm-exporter DaemonSet（精简 counter 集）
│   └── grafana/                # 23 面板仪表盘 + provisioning（datasource/dashboard 自动装配）
├── k8s/
│   ├── base/                   # Namespace Quota / ConfigMap / Deployment /
│   │                           # Service / Ingress(SSE) / HPA / PDB / VPA /
│   │                           # ServiceMonitor / PrometheusRule / Jobs
│   └── overlays/prod/          # 副本数、镜像 digest、拓扑硬约束、限流
├── tests/
│   ├── test_harness.py         # 23 个测试：钉住指标语义、负载模型、缓存污染防御
│   └── verify_run.py           # 断言一个 run 结构完好且非退化
├── docs/                       # 方法论与运维（见下）
├── .github/workflows/ci.yml    # lint / test(3 个 Python 版本) / 无 GPU 冒烟
├── Makefile
├── LICENSE                     # MIT（demo 数据为合成数据，见文件内声明）
└── requirements.txt
```

---

## 文档

| 文档 | 内容 |
|---|---|
| [01-architecture.md](docs/01-architecture.md) | 组件、数据流、端口、五个关键设计决策 |
| [02-methodology.md](docs/02-methodology.md) | **固定负载**的定义；开环 vs 闭环；coordinated omission；长度控制；重复次数 |
| [03-metrics.md](docs/03-metrics.md) | 指标字典：TTFT/TPOT/ITL/P99/goodput 的公式、来源、失效模式 |
| [04-experiments.md](docs/04-experiments.md) | 三组对照实验的设计、执行、噪声底、结论模板 |
| [05-k8s.md](docs/05-k8s.md) | 部署、探针设计依据、资源限额、告警 runbook、验收清单、故障演练 |
| [06-capacity.md](docs/06-capacity.md) | 容量模型：KV 预算、roofline、副本数公式、冷启动与 HPA 反应时间 |
| [07-report-example.md](docs/07-report-example.md) | 用 mock 数据填好的完整报告示例（明确标注为合成数据） |

---

## 四个最容易被做错的地方

1. **用 `/health` 判断就绪。** vLLM 在权重还在往显存里搬的时候就会返回 200。
   第一个变体的 TTFT 里会混进模型加载时间，A/B 直接失效。
   → `wait_ready.sh` 发一次真推理才返回成功。

2. **闭环压测报告"X rps"。** 闭环（N 个 worker 打满）测的是峰值吞吐，
   不是"系统能承接 X rps"。后者必须用开环（Poisson 到达）测，
   否则会犯 coordinated omission：慢请求不会触发新请求，尾部延迟被系统性低估。
   → `loadgen.py` 两种模式都实现，`meta.json` 记录用的是哪种。

3. **用引擎的累计 gauge 做 A/B。** `vllm:gpu_prefix_cache_hit_rate` 是
   **进程启动以来**的累计值；在同一进程上连跑两次，两次的读数几乎一样，
   对照信号直接消失。
   → `loadgen.py` 对 counter 做**窗口内差分**，结果放在 `summary.server.window`。

4. **prefix cache 跨 run 污染。** 引擎缓存是进程级的：固定 seed 的 loadgen
   重放同样的 prompt 序列，第 2 个 run 打在焐热的缓存上，对照组命中率能
   "测出" 100%。我们在 demo 数据里真实翻过这个车（复盘见
   [docs/07 §7](docs/07-report-example.md)）。
   → `run_matrix.sh` 每个 run 前自动 `flush_cache.sh`；loadgen 结束自动做
   8 项效度检查，"没配共享前缀但命中率 >0.3" 直接写进 `summary.warnings`。

---

## 环境要求

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | ≥ 3.9 | 压测器与分析工具（仅依赖 aiohttp / numpy / pyyaml） |
| vLLM | ≥ 0.6（推荐 0.8.x） | `serve_vllm.sh` 会探测 `--help` 适配新旧 flag 拼写 |
| SGLang | ≥ 0.4 | 需要 `--enable-metrics` 才有 `/metrics` |
| GPU | 24 GB+（7B FP16）/ 12 GB+（int4） | KV 预算见 `capacity_model.py` |
| Kubernetes | ≥ 1.26 | HPA v2、PodSecurity admission |
| Prometheus Operator | 任意近期版本 | ServiceMonitor / PrometheusRule；或用 `prometheus.yml` 裸跑 |

---

## 许可与声明

代码 MIT。`bench/results/demo/` 下的数据由 `mock_server.py` 生成，
**延迟与吞吐均为合成值，不可作为任何硬件或引擎的性能参考**；
它们的用途是验证流水线和示范报告格式。
