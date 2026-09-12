# BranchServe

双 GPU 长上下文 Agent fan-out 推理调度。当父请求已产生多个共享长上下文的子请求时,路由器在三个动作间选择:**PACK**(子请求留在 parent 所在 GPU,复用本地 Prefix Cache)、**RETRIEVE**(迁往另一 GPU 并通过共享存储携带 KV)、**RECOMPUTE**(迁往另一 GPU,空手重算共享上下文)。系统基于 vLLM 与 LMCache 构建:双 connector worker 与 lmcache server 常驻一套栈,三个动作零换栈可选(可部署口径);不修改模型或 vLLM 内核。

## 主要结果

测量条件:2×RTX 4090;Qwen3.5-4B(bf16,max_model_len 16384);双 vLLM worker(max_num_batched_tokens=1024)+ lmcache server(chunk 528,L1=100GB);4 轮追加式会话,共享前缀 8221→14359 tokens,每轮 parent 增量处理后 4 个 child 并发(各 256 输出 token);压力 = parent 卡上 N 个持续解码的背景请求(N=0/2/4/6/8),屏障保证派发时背景在位;派发策略为乐观派发(parent 返回即发 child)。除注明外每格 n=1,噪声带约 ±100ms。

child 组完成时间(child_makespan,4 轮均值,ms):

| 压力 | PACK | RETRIEVE | RECOMPUTE | 最优动作 |
|---:|---:|---:|---:|---|
| 0 | 1447 | 1492 | 1822 | PACK |
| 2 | 1425 | 1559 | 1759 | PACK |
| 4 | 1570 | 1670 | 1745 | PACK |
| 6 | 1670 | 1498 | 1775 | RETRIEVE |
| 8 | 1685 | 1426 | 1829 | RETRIEVE |

要点:

1. **调度边界(crossover)在压力 4-6 之间**:PACK 随 parent 卡压力线性恶化(最小二乘约 36ms/压力单位,截距约 1415ms);RETRIEVE 的 child 不占用 parent 卡,对压力不敏感;RECOMPUTE 在全部压力点不占优(首轮首孩重算税 2150-2280ms,摊销后仍高于 RETRIEVE 摊销值约 100-150ms)。
2. **动态路由贴近 oracle**:阈值规则(观测压力 ≥5 → RETRIEVE,否则 PACK;检测到 store 失效自动降级 RECOMPUTE)在 5 个压力点全部选中该压力下的最优动作;稳态(r1-3)相对逐轮 oracle 的 regret 落在 ±100ms 噪声带内;固定策略在选错一侧的压力点亏损 130-250ms。
3. **KV 写入 CPU 的成本被流式掩盖**:store 随 parent prefill 边算边写,parent 返回前完成;8221 tokens 首轮 CPU 忙时 192ms,后续轮仅 38-53ms,不产生额外等待。
4. **真实轨迹整轨回放(Mooncake 轨迹 500 条筛出 431 条可执行,16K 窗口)**:PACK 838.16s / Recompute 901.94s / Retrieve 770.38s。整轨吞吐口径下 Retrieve 最优(双卡并行),与上表单组延迟口径(低压力 PACK 最优)是两个 regime,结论互补。

## 三个动作的实现(统一栈)

| 动作 | child 去向 | 机制 |
|---|---|---|
| PACK | parent 卡 | 本地 APC 命中,无跨卡流量 |
| RETRIEVE | 另一卡 | 从 lmcache server 检索 parent 已存 KV |
| RECOMPUTE | 另一卡(请求携带 cache_salt) | 缓存键落空,真实全量重算 |

RECOMPUTE 动作借用 vLLM 请求级 `cache_salt`(缓存盐)实现:cache_salt 参与本地 APC 与 lmcache 仓库的键计算,子请求携带与 parent 不同的 salt 即可让两级缓存查找都落空。lmcache 0.5.4 的仓库键包含 salt,已实测验证(带 salt 子请求 local_compute=8245、零 transfer;不带 salt 对照正常 transfer 7920)。固定每会话一个 salt 时,同 salt 组内及轮间共享重算成果(首轮首孩全量重算,其余子请求及后续轮命中),即"重算过的卡保持热"的可部署语义。

## 仓库结构

```text
BranchServe/
├── core/                         # 调度器核心：Router、策略、KV/pressure metrics、数据结构
│   ├── router.py
│   ├── policies.py
│   ├── cost_model.py
│   ├── metrics.py
│   └── models.py
├── experiments/                  # 实验入口与 workload runner
│   ├── multiround_strategy.py    # 多轮三策略 runner
│   ├── multiround_pressure.py    # pressure/barrier/三动作/Dynamic
│   ├── multi_workflow.py         # Parent/Child fan-out workflow
│   ├── pressure_sweep.py         # 压力扫描
│   ├── prepare_mooncake_trace.py # Mooncake 轨迹准备
│   ├── replay_mooncake_*.py      # Mooncake 回放
│   └── prepare_qmsum.py          # QMSum 数据准备
├── analysis/                     # 汇总、对比和 oracle/regret 分析
│   ├── analyze_upressure.py
│   ├── analyze_dynamic.py
│   └── summarize_*.py
├── validation/                   # Gate1/Gate2/Gate3 路径验证与测试
│   ├── gate1_single_worker.py
│   ├── gate2_*.py
│   ├── gate3_*.py
│   └── test_*.py
├── deployment/                   # vLLM、LMCache、Router 启动与服务脚本
│   ├── branchserve_service.py
│   ├── start_worker.sh
│   └── start_mooncake_connector_stack.py
├── docs/                         # 阶段报告、实验协议和项目结论
├── README.md                     # 项目说明、结果和复现入口
└── .gitignore                    # 排除日志、模型、原始数据和运行产物
```

实验运行产生的 `artifacts/`、`logs/`、模型和原始数据保留在实验服务器，不提交到公开仓库。

## 环境要求

- 2×GPU(实测 RTX 4090 24G);驱动与 CUDA 支持torch 2.11 + vllm 0.26.0
- conda 环境:python 3.12 / torch 2.11.0+cu130 / vllm 0.26.0 / lmcache 0.5.4 / transformers
- 模型:Qwen3.5-4B(混合注意力+Mamba 架构;LMCache 走 lmcache 0.5.4 的 LMCacheMPConnector 外部模块路径,vLLM 自带 connector 不支持该混合架构)
- 主机内存 ≥ 110GB(lmcache server L1=100GB 会被 pin)

## 复现

以仓库内路径为例(实验机为 2×4090 容器,模型与 conda 环境路径按实际修改):

```bash
# 1. 启动 lmcache server 与统一栈(双 connector worker)
bash start_mooncake_connector_stack.py 所在环境执行:
  lmcache server --host 127.0.0.1 --port 5555 --chunk-size 528 \
    --separate-object-groups --l1-size-gb 100 --eviction-policy LRU
  python start_mooncake_connector_stack.py   # 起 :8000/:8001 并等待健康

# 2. 验证 RECOMPUTE 的 salt 机制(可选)
python salt_verify.py

# 3. 压力矩阵(每格一个独立 label,天然隔离,无需重启 worker)
python multiround_pressure.py --strategy pack      --pressure 6 --label exp-pack-p6
python multiround_pressure.py --strategy retrieve  --pressure 6 --label exp-retr-p6 \
  --server-log logs/<server日志>
python multiround_pressure.py --strategy recompute --pressure 6 --label exp-reco-p6
python multiround_pressure.py --strategy dynamic   --pressure 6 --label exp-dyn-p6 \
  --dynamic-threshold 5 --server-log logs/<server日志>

# 4. 汇总
python analyze_upressure.py    # 交叉表
python analyze_dynamic.py      # dynamic regret
```

配置均以命令行参数提供(压力、轮数、前缀长度、fan-out、阈值、label、salt),可在不改代码的情况下扩展到其他网格。

## 测量口径与限制

- 比较单位为 child 组完成时间(4 个子请求并发,首末包络);parent 阶段在各动作间同构,不计入组时间。
- 归因依据 vLLM `prompt_tokens_by_source_total` 三源差分(local_compute / local_cache_hit / external_kv_transfer),逐 worker 逐轮记录;APC 命中按块对齐(如 8221 tokens 实际命中 7920)。
- 背景压力为纯解码负载(锚定会话首轮前缀,APC 热命中),真实 Agent 混合负载下 PACK 的恶化斜率可能更陡。
- 每格 n=1(重复实验驱动 `run_repeats.sh`);噪声带约 ±100ms,与低压力端动作间差距同量级,正式引用应以多次重复的均值±标准差为准。
- 观察到的 LMCache 工程问题(与本项目结论独立):lmcache 0.5.4 的 lmcache server 在持续"只写不读"流量下会出现传输路径 CUDA 错误并静默失效(子请求查不到即本地重算,无报错);受控实验排除了容量耗尽、驱逐失效、GPU 显存争抢三种解释,存取配套的流量(retrieve 类)在同等规模下未复现。本项目的 dynamic 路由以 transfer=0 为信号自动降级 RECOMPUTE,已实测生效。

## 已验证能力边界

- 单机双 GPU、单模型(Qwen3.5-4B)、16K 上下文内的结论;多机、更大上下文、其他模型未测不声称。
- Dynamic 为阈值规则(非学习型成本模型);成本模型重拟合使用本表数据(36ms/压力单位),未做跨负载泛化验证。

## 项目背景

BranchServe 面向双 GPU 长上下文 Agent fan-out 推理。一个 Parent 请求会派生多个共享长 prefix 的 Child 请求。系统需要在本地 Prefix Cache 复用和跨 GPU 并行之间选择执行位置。

## 核心策略

- **PACK**：Child 全部留在 Parent worker，复用本地 prefix cache。
- **Retrieve**：Child 分到另一 worker，复用 Parent 已生成的 KV。
- **Recompute**：Child 分到另一 worker，重新计算共享 prefix。
- **Dynamic**：根据 worker pressure 和 cache 状态在 PACK 与 Retrieve 等路径之间选择。

LMCache 为长上下文提供 KV 保存、复用和跨 worker Retrieve 能力。实验区分 local cache hit、external KV transfer 和 recompute，不把缓存读取视为零成本。

## 实验结果

### Mooncake 431 条正式重放

Qwen3.5-4B、双 RTX 4090、16K serving window。PACK、Retrieve、Recompute 均完成 431/431：

| 策略 | 总耗时 |
|---|---:|
| PACK | 838.16 s |
| Retrieve | 770.38 s |
| Recompute | 901.94 s |

Retrieve 相比 PACK 快约 8.8%，相比 Recompute 快约 14.6%。该口径从 Parent 已存在后开始，主要衡量 Child 调度阶段；Mooncake hash replay 用于系统性能和 KV 路径验证，不用于回答质量评估。

### LongBench QMSum

19 份会议、76 个问题、456 份回答：

| 策略 | 平均整组耗时 |
|---|---:|
| PACK | 1.710 s |
| Retrieve | 1.647 s |
| Recompute | 2.446 s |

Retrieve 比 PACK 快约 3.7%，比 Recompute 快约 32.7%。

### 多轮 pressure

4 轮长上下文协议：初始 8K、每轮增加 2K、fan-out=4，pressure=0/2/4/6/8。结果显示低压力下 Dynamic 选择 PACK，高压力下选择 Retrieve，存在明确 pressure crossover。

## 当前结论与限制

在当前双 GPU、长共享上下文和 fan-out 场景中，低压力优先 PACK，高压力优先 Retrieve，Recompute 通常最慢。Dynamic 已能根据 pressure 完成 PACK→Retrieve 切换；更长上下文下仍需继续完善跨轮 cache-ready 状态跟踪。当前结果不外推到任意模型、GPU 数量或 32K/64K 上下文。

## 复现入口

```bash
bash scripts/start_workers.sh configs/4090-qwen35-4b.env.example
bash scripts/check_workers.sh configs/4090-qwen35-4b.env.example
python multiround_pressure.py --strategy dynamic --pressure 6 --rounds 4 --initial-tokens 8192 --append-tokens 2048 --fanout 4 --child-max-tokens 256 --out artifacts/multiround_dynamic.json
```
## 项目故事：为什么需要 BranchServe

长上下文 Agent 经常先处理一个 Parent 任务，再从同一段历史上下文派生多个 Child 分支。例如，一个长会议、代码仓库或工具调用历史先被 Parent 读取，随后多个 Child 分别回答不同问题。Child 之间共享很长的 prefix，但分支内容和生成结果不同。

这带来一个直接冲突：

```text
留在 Parent 所在 GPU：可以复用本地 Prefix Cache，但多个 Child 会排队；
迁移到另一张 GPU：可以并行执行，但需要 Retrieve KV，或者重新计算长 prefix。
```

BranchServe 的问题不是“如何把请求平均分给 GPU”，而是：

> 在当前 worker pressure、prefix cache 状态和 KV 传输成本下，这一组 Child 应该 PACK、Retrieve 还是 Recompute？

## 我们的核心假设

- 共享 prefix 越长，重复 Recompute 越昂贵；
- Parent worker 越繁忙，PACK 的排队成本越高；
- 另一张 GPU 空闲且 KV 可用时，Retrieve 可以同时获得 prefix 复用和并行收益；
- 因此最优策略可能随 pressure 变化，而不是固定不变。

## 系统如何工作

1. Parent 在一个 vLLM worker 上处理长上下文；
2. LMCache 保存已经计算出的 KV；
3. Parent 派生一组 Child，BranchServe Router 将整组 Child 作为一个调度单元；
4. Router 读取两个 worker 的 running/waiting pressure 和 cache 状态；
5. Router 选择 PACK、Retrieve 或 Recompute；
6. 所有 Child 完成后记录 group makespan、KV 来源和策略 regret。

LMCache 在这里不是“把文本放到 CPU”，而是提供 KV 的保存和复用路径。KV 可能命中 GPU 本地 Prefix Cache，也可能通过 LMCache 跨 Worker Retrieve；如果 cache 不可用，则进入 Recompute。实验记录 `local_compute`、`local_cache_hit` 和 `external_kv_transfer` 三类来源，避免把缓存读取误当成零成本。

## 实验路线

项目按由底到顶的顺序推进：

1. **基础路径**：验证单 Worker、Prefix Cache 和 LMCache 存取；
2. **跨 Worker 路径**：验证 Retrieve、Recompute、KV transfer 和状态一致性；
3. **真实任务**：在 LongBench QMSum 上比较三种策略的延迟和回答指标；
4. **系统压力**：在 Mooncake Tool/Agent trace 上完成 431/431 正式重放；
5. **多轮调度**：固定 8K 初始上下文、每轮增加 2K、4 轮 fan-out，在 pressure=0/2/4/6/8 下验证 Dynamic。

## 最重要的结果

### 三策略正式基线

Mooncake 431 条可执行轨迹全部完成：

| 策略 | 总耗时 |
|---|---:|
| PACK | 838.16 s |
| Retrieve | 770.38 s |
| Recompute | 901.94 s |

Retrieve 比 PACK 快约 8.8%，比 Recompute 快约 14.6%。该结果采用 Parent 已存在后的 Child 调度口径；Parent 写入成本单独记录，不将两种口径混淆。

### 真实文本任务

QMSum 19 份会议、76 个问题、456 份回答：

| 策略 | 平均整组耗时 | ROUGE-L |
|---|---:|---:|
| PACK | 1.710 s | 21.885 |
| Retrieve | 1.647 s | 21.889 |
| Recompute | 2.446 s | 22.158 |

Retrieve 比 PACK 快约 3.7%，比 Recompute 快约 32.7%。三种策略输出质量接近；ROUGE-L 只作为词面指标，不代表事实正确率。

### 多轮 pressure crossover

在 4 轮、8K 初始上下文、每轮增加 2K 的协议下：

```text
pressure 0/2/4：Dynamic → PACK
pressure 6/8：Dynamic → Retrieve
```

这说明当前双 GPU 长上下文 fan-out 场景存在清晰的 pressure crossover：低压力时本地复用更划算，高压力时跨 Worker Retrieve 的并行收益更大。Recompute 在当前实验范围内通常最慢。

## 结论边界

我们证明的是一个限定但可复现的命题：

> 在双 GPU、Qwen3.5-4B、长共享 prefix 和固定 fan-out 的场景中，最优放置取决于 worker pressure；Dynamic 可以在低压力选择 PACK，在高压力选择 Retrieve。

我们没有声称 Retrieve 在所有模型、上下文长度、GPU 数量和负载下都最优，也没有把 Mooncake hash replay 当作语义质量评测。32K/64K 上下文、多 Parent、多级 DAG 和更大集群属于后续扩展。

## 如何阅读本仓库

- 先读本文的项目故事和结果；
- 再看 `core/` 理解 Router、策略和 telemetry；
- 看 `experiments/` 了解多轮和 pressure 协议；
- 看 `analysis/` 复现汇总和 oracle/regret；
- 看 `validation/` 了解 Gate1/Gate2/Gate3 路径验证；
- 看 `deployment/` 了解 vLLM、LMCache 和服务启动方式；
- 看 `docs/` 阅读阶段报告和实验限制。
