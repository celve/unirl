# Reward Service — 架构与数据流

本文档描述 Reward Service 的**系统拓扑**、**请求数据流转**、**关键抽象层次**、**错误/资源隔离语义**与**扩展点**。面向三类读者：

- 新加入开发者 —— 一次看懂整个系统
- 做 debug / 扩展的人 —— 知道每个职责落在哪个文件
- 未来的维护者 —— 理解设计抉择的"为什么"

配套文档：
- [`README.md`](../README.md) —— 怎么装、怎么用
- [`docs/DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md) —— 开发时间线、决策档案
- [`CHANGELOG.md`](../CHANGELOG.md) —— 用户视角变更表

---

## 1. 静态拓扑（Static Topology）

一张图看完整个进程结构：

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Host (single machine, multi-GPU)                                        │
│                                                                          │
│   ┌──────────────────────────┐                                           │
│   │ uvicorn + FastAPI        │   (1 个进程、asyncio event loop)          │
│   │   create_app(cfg)        │   源码:  reward_service/server.py         │
│   │                          │                                           │
│   │   /score   /health       │                                           │
│   │   /rewards               │                                           │
│   │                          │                                           │
│   │   app.state.pool ────────┼──► WorkerPool  (reward_service/workers/pool.py)
│   └──────────────────────────┘                │                          │
│                                               │ 1 pool : N groups        │
│                                               ▼                          │
│   ┌────────────── Ray runtime (ray.init) ────────────────┐               │
│   │                                                      │               │
│   │  ┌───────────────┐ ┌───────────────┐   ┌──────────┐  │               │
│   │  │ WorkerGroup   │ │ WorkerGroup   │…  │ Worker…  │  │               │
│   │  │  name=clip    │ │ name=geneval2 │   │          │  │               │
│   │  │  (group.py)   │ │  (group.py)   │   │          │  │               │
│   │  │               │ │               │   │          │  │               │
│   │  │  actors[0] ───┼─┼──────────┐    │   │          │  │               │
│   │  │  actors[1] ───┼─┼─────────┐│    │   │          │  │               │
│   │  └───────────────┘ └─────────┼┼────┘   └──────────┘  │               │
│   │                              ││                      │               │
│   │   @ray.remote ScorerActor    ││  (actor.py)          │               │
│   │   ┌──────────────────────┐   ││   ┌──────────────┐   │               │
│   │   │ ScorerActor          │   ││   │ ScorerActor  │   │               │
│   │   │   scorer=ClipScorer  │   ││   │  TP=2 vLLM   │   │               │
│   │   │   num_gpus=1         │   ││   │  num_gpus=2  │   │               │
│   │   └─────────┬────────────┘   ││   └──────┬───────┘   │               │
│   └─────────────┼────────────────┼┼──────────┼───────────┘               │
│                 │                ││          │                           │
│           GPU 0 │                ││          │ GPU 6 + GPU 7             │
│                 ▼                ▼▼          ▼                           │
│            [ CUDA ctx ]   [ CUDA ctx × 2 ]   [ CUDA ctx × 2 ]            │
│                                                                          │
│                   独占 GPU，不跨 reward 共享                              │
└──────────────────────────────────────────────────────────────────────────┘
```

**关键属性**：

- **1 FastAPI 进程** · **N WorkerGroup** · **Σ num_replicas 个 ScorerActor**（Ray 子进程）
- Ray 通过 `ScorerActor.options(num_gpus=N, num_cpus=C)` 把 GPU 作为"**整数资源**"分配给 actor —— 一张卡同一时刻只属于一个 actor
- vLLM 类 scorer 若用 TP=N，actor 的 `num_gpus=N` 与 vLLM 的 `tensor_parallel_size=N` 必须一致（由 YAML 同时配）
- 多机扩展点：YAML 里加 `cluster.ray_address` 即可；架构无需改，详见 §5.3

**组件对应的源码文件**：

| 组件 | 源码 | 职责 |
|---|---|---|
| HTTP 入口 | `reward_service/server.py` | `/score` `/health` `/rewards` 三个 endpoint；bucket / dispatch / gather 逻辑 |
| Schema | `reward_service/schemas.py` | Pydantic：`RewardRequest` / `ScoreResponse` |
| 配置 | `reward_service/config.py` | YAML → `ServiceCfg` + `RewardModelCfg` dataclass；校验 `num_replicas≥1`、`num_gpus≥0`、名字唯一 |
| WorkerPool | `reward_service/workers/pool.py` | Ray runtime 生命周期 + group 注册表 + 按名字 dispatch |
| WorkerGroup | `reward_service/workers/group.py` | N 个 actor + round-robin 派发 (`itertools.cycle`) |
| ScorerActor | `reward_service/workers/actor.py` | `@ray.remote` 薄壳：构造 scorer、转发 `score()` |
| BaseScorer | `reward_service/scorers/base.py` | 抽象契约：`score(items) -> list[dict]` + `sub_metric_names` |
| Registry | `reward_service/scorers/registry.py` | `register(name, cls)` + 可选 dep 容错（`_try_import`） |
| 具体 scorer | `reward_service/scorers/{clip,pickscore,imagereward,hpsv2_scorer,hpsv3_scorer,unified_reward,geneval2,geneval,ocr,videoalign}.py` (+ vendored `_videoalign/`) | 每个 reward 一个模块，模块末尾 `register("name", Cls)` |

---

## 2. 数据流转（Request → Response）

一次 `POST /score` 的完整时序。以 batch 含 2 个 request、每个要 2 个 reward 为例：

```
Client                 FastAPI              WorkerPool            WorkerGroup          ScorerActor (Ray)
  │                       │                     │                      │                      │
  │ POST /score           │                     │                      │                      │
  │  {requests:[r0,r1]}   │                     │                      │                      │
  ├──────────────────────►│                     │                      │                      │
  │                       │                                                                   │
  │                       │ (1) asyncio.to_thread(_request_to_item × 2)                       │
  │                       │     base64 → PIL.Image  (CPU-heavy, offload event loop)           │
  │                       │                                                                   │
  │                       │ (2) _bucket_by_reward                                             │
  │                       │     r0 needs [clip, hpsv2]                                        │
  │                       │     r1 needs [clip, pickscore]                                    │
  │                       │     → buckets = {                                                 │
  │                       │         "clip":      [(0, item0), (1, item1)],                    │
  │                       │         "hpsv2":     [(0, item0)],                                │
  │                       │         "pickscore": [(1, item1)],                                │
  │                       │       }                                                           │
  │                       │                                                                   │
  │                       │ (3) pool.dispatch(name, items_for_name) 并行三次                   │
  │                       ├────────────────────►│                      │                      │
  │                       │   dispatch("clip",  ├──► round-robin ─────►│                      │
  │                       │       [item0,item1])│    actors[rr_idx]    ├──► .score.remote()─►│ ObjectRef₀
  │                       │                     │                      │                      │
  │                       ├────────────────────►│                      │                      │
  │                       │   dispatch("hpsv2", ├──► …                 ├──► .score.remote()─►│ ObjectRef₁
  │                       │       [item0])      │                      │                      │
  │                       │                     │                      │                      │
  │                       ├────────────────────►│                      │                      │
  │                       │   dispatch("pick…", │                      │                      │
  │                       │       [item1])      │                      │                      │
  │                       │◄────── 3 ObjectRef ─┤                      │                      │
  │                       │                                                                   │
  │                       │                                            ScorerActor 运行:       │
  │                       │                                              scorer.score(items)   │
  │                       │                                              →  list[dict[str,float]]
  │                       │                                                                   │
  │                       │ (4) asyncio.to_thread(_gather_with_errors, names, refs)            │
  │                       │     逐 ref ray.get —— 一个挂了不影响其他                           │
  │                       │                                                                   │
  │                       │ (5) 按 (bucket 下标, reward 名) 装回 results[i][name] 与 errors[i] │
  │                       │                                                                   │
  │  200 OK               │                                                                   │
  │  {results:[…],        │                                                                   │
  │   errors:[…]}         │                                                                   │
  │◄──────────────────────┤                                                                   │
```

**时序要点**：

1. **解码与 dispatch 全程异步**：base64→PIL 和 `ray.get` 都走 `asyncio.to_thread`，不占 event loop，`/health` 和其他 `/score` 请求可并发响应。
2. **每个 reward 一次 Ray call**：同一 reward 下所有 item 打包进一个 `score([item0, item1, ...])` 调用；bucket 把 request 索引 `i` 记下，聚合阶段凭它装回 `results[i]`。
3. **Round-robin 只在 `num_replicas>1` 时有意义**；单 replica 时永远命中 `actors[0]`。
4. **图像复制代价**：同一个 PIL.Image 出现在多个 bucket（比如 `item0` 同时给 clip + hpsv2）时，Ray 会把它 pickle 到 object store **两次**。这是已知限制（见 DEVELOPMENT_LOG §7.2）；batch 很大时可上 `ray.put(item)` 去重。

---

## 3. 关键抽象层次（Four Layers）

从"最稳定、对外承诺最多"到"最实现细节":

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  L0 · BaseScorer                               scorers/base.py              │
│  契约层（最稳定）                                                             │
│                                                                             │
│   class BaseScorer(ABC):                                                    │
│       name: str                                                             │
│       sub_metric_names: tuple[str, ...]                                     │
│       def score(items: list[ScoreItem]) -> list[dict[str, float]]: ...      │
│                                                                             │
│   —— 纯 Python，不知道 Ray、FastAPI、GPU 调度                                 │
│   —— 输出约束：len(out) == len(items)，sub-metric key 统一 str→float         │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ▲
                                       │ registry.register(name, Cls)
                                       │
┌─────────────────────────────────────────────────────────────────────────────┐
│  L1 · Concrete scorers                         scorers/{clip,hpsv2,…}.py    │
│  模型实现层                                                                   │
│                                                                             │
│   class ClipScorer(BaseScorer):                                             │
│       def __init__(self, model_name, weights_path, dtype, device): ...      │
│       def score(self, items): ...  # 纯 torch / vLLM 推理                    │
│                                                                             │
│   模块末尾: register("clip", ClipScorer)                                     │
│                                                                             │
│   —— 复用 scorers/_common.py 的 resolve_dtype / resolve_model_path /         │
│      split_last_turn / image_to_data_url / build_vllm_llm_kwargs             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ▲
                                       │ wrapped by @ray.remote
                                       │
┌─────────────────────────────────────────────────────────────────────────────┐
│  L2 · ScorerActor                              workers/actor.py             │
│  Ray 薄壳层                                                                   │
│                                                                             │
│   @ray.remote                                                               │
│   class ScorerActor:                                                        │
│       def __init__(self, scorer_name, params):                              │
│           cls = get_scorer_cls(scorer_name)   # 在目标 GPU 进程上查 registry  │
│           self.scorer = cls(**params)         # 在目标 GPU 进程上构造模型     │
│       def score(items): return self.scorer.score(items)                     │
│                                                                             │
│   —— 关键设计：传 "name + params" 而非 scorer 实例                            │
│     (避免序列化 GB 级模型权重，改在目标 GPU 上 from-scratch 构造)              │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ▲
                                       │ WorkerGroup spawns N replicas
                                       │
┌─────────────────────────────────────────────────────────────────────────────┐
│  L3 · WorkerGroup / WorkerPool                 workers/{group,pool}.py      │
│  调度与生命周期层                                                             │
│                                                                             │
│   WorkerGroup(cfg):                                                         │
│     actors = [ScorerActor.options(num_gpus=n).remote(name, params)          │
│               for _ in range(cfg.num_replicas)]                             │
│     rr = itertools.cycle(range(len(actors)))    # 简单轮询                   │
│                                                                             │
│   WorkerPool(ServiceCfg):                                                   │
│     ray.init(…)                                                             │
│     {reward_name: WorkerGroup(reward_cfg) for reward_cfg in cfg.rewards}    │
│                                                                             │
│   —— 副作用集中：Ray init / actor 创建 / shutdown 都在这层                   │
│   —— 往上抛 Ray ObjectRef，不做聚合                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ▲
                                       │
                                       │  server.py 拿到 ObjectRef → gather → assemble
                                       │
                                    ( L4 HTTP gateway 已在 §1 §2 覆盖 )
```

**每层只依赖下一层**。理论上：
- 换掉 `BaseScorer`：整个体系要重写
- 换掉 `ScorerActor` 为其他 RPC（gRPC、Ray Serve 等）：L0/L1 不动，L3 的 group/pool 改实现
- 换掉 HTTP gateway：L0-L3 全部复用

---

## 4. 错误与资源隔离（Isolation Semantics）

Reward Service 的两条硬承诺：

### 4.1 资源隔离

**承诺**：一张 GPU 在同一时刻只属于一个 reward group。

**实现机制**：`WorkerGroup._spawn_actors()` 用 `ScorerActor.options(num_gpus=self.cfg.num_gpus)`，Ray 把 GPU 作为**整数资源**分配；两个 actor 永远不会被调度到同一张卡。

**YAML 上的反映**：
```yaml
- name: clip
  num_gpus: 1        # 独占 1 张卡
  num_replicas: 1

- name: geneval2
  num_gpus: 2        # 独占 2 张卡（给 vLLM TP=2 用）
  num_replicas: 1
  params:
    tensor_parallel_size: 2    # 必须与 num_gpus 一致
```

**校验在 `config.py`**：`num_gpus < 0` 和 `num_replicas < 1` 直接拒绝加载（带 reward 名的错误信息）。

**超出预算的后果**：`ray.init()` 成功但某个 actor 会在 GPU 分配阶段 hang 或报资源不足 —— Ray 层面的错误，会反映在 `/health` 上（对应 group `ping()` 不返回）。

### 4.2 错误隔离

**承诺**：单个 reward 的失败（OOM / parse 异常 / actor crash）不会让整个 batch 返回 500。失败的 reward 会把异常写入 `ScoreResponse.errors[i][reward_name]`，其他 reward 照常返回分数。

**实现机制**：`server.py:66-80` 的 `_gather_with_errors`：

```python
for name, ref in zip(reward_names, refs):
    try:
        results.append(ray.get(ref))
    except Exception as e:
        logger.exception("reward %s failed: %s", name, e)
        results.append(e)
```

逐 ref `ray.get` —— 一个 reward 的 actor 异常被 catch，**不 poison 其他 reward 的 ObjectRef**。

**响应示例**（reward `clip` 挂了，其他 OK）：
```json
{
  "results": [
    {"hpsv2": {"hpsv2": 0.27}},
    {"hpsv2": {"hpsv2": 0.31}, "pickscore": {"pickscore": 0.82}}
  ],
  "errors": [
    {"clip": "RayActorError: ..."},
    {"clip": "RayActorError: ..."}
  ]
}
```

请求下标 `0` 要了 `clip + hpsv2`，`clip` 挂了 → 只有 `hpsv2` 的分数，`errors[0]["clip"]` 记录异常；`hpsv2` 不受影响。

### 4.3 两条承诺对客户端的意义

- 客户端可以放心地把不相关的 reward 打包进同一 batch —— **不会互相拖累**
- 客户端必须**同时读 `results[i]` 和 `errors[i]`** —— 单看 `results` 会漏掉失败的 reward

---

## 5. 扩展点（Extension Points）

### 5.1 新增一个 scorer

1. 在 `reward_service/scorers/` 下新建 `my_scorer.py`：
   ```python
   from reward_service.scorers.base import BaseScorer, ScoreItem
   from reward_service.scorers.registry import register

   class MyScorer(BaseScorer):
       name = "my_reward"
       sub_metric_names = ("my_metric",)

       def __init__(self, model_name: str, weights_path: str | None = None, ...):
           # 在目标 GPU 上加载模型
           ...

       def score(self, items: list[ScoreItem]) -> list[dict[str, float]]:
           ...

   register("my_reward", MyScorer)
   ```
2. 在 `registry.py` 的 `SCORER_MODULES` dict 里加一行 `"my_reward": "reward_service.scorers.my_scorer"`（ScorerActor 在自己的 venv 内按此映射 import 模块并触发 `register()`；**主进程不 import scorer 模块**，所以模块顶层不要 import 重依赖——放进 `__init__`/方法内，见 clip.py/imagereward.py）
3. 在 `configs/service.example.yaml` 里加对应段
4. 在 `tests/scorers/` 下新增 `test_my_scorer.py`（CPU 纯函数测试 + GPU smoke test）

**不需要改的东西**：`server.py` / `workers/` / `schemas.py` —— 新 scorer 会被 registry 自动发现、被 WorkerPool 按 YAML 自动创建 group。

### 5.1.1 隔离层级的硬约束（新 scorer 的 env 能走多远）

每个 scorer 通过 `envs/<name>.txt` + Ray `runtime_env` 拿到隔离的 venv，但隔离**只到 Python 包层、不到 Python 解释器层**，原因是 Ray 2.x runtime_env 的两条硬约束：

1. **pip 后端永远 `--system-site-packages`**（`ray/_private/runtime_env/virtualenv_utils.py` 写死，无旋钮）→ venv 继承 base 的 torch/编译扩展，**无法换一个不同的 torch 构建**（换了会与 base 漏进来的 xformers/flash-attn ABI 冲突）。
2. **conda/container 后端强制 `python={集群当前版本}`**（`ray/_private/runtime_env/conda.py` 无条件 `deps.append(f"python={py_version}")`）→ worker 必须与 cluster 同一个 Python minor（cloudpickle 兼容）。

**推论**：新 scorer 的依赖必须能跑在 **base 的 (Python × torch)** 上（仅可在 venv 里叠加/覆盖纯 Python 层，如不同的 `transformers`/`peft` 版本）。需要**不同 Python 或不同 torch 地基**的 scorer（典型：`geneval` 的 mmdet 2.x / mmcv-full 1.7.2 需要 py3.10 + torch≤2.1）**无法**通过 runtime_env 在本集群（py3.13 / torch 2.x）托管——只能：(a) 整个集群降到兼容的 py/torch，或 (b) 把该 scorer 作为**集群外 sidecar 服务**、用一个瘦 actor 代理调用。`geneval` 因此在 `configs/service.example.yaml` 里默认注释掉，详见 `envs/geneval.txt`。

### 5.2 给已有 scorer 加 sub-metric

- 改 scorer 类的 `sub_metric_names` 元组 + `score()` 返回 dict 里多加一个 key
- 响应 schema `ScoreResponse.results[i][reward_name]` 是 `dict[str, float]`，天然支持多 key，**不用改 schema**
- 加回归测试确认新 key 存在、值在合理范围

### 5.3 多机扩展

接入外部 Ray cluster 无需改代码，只改 YAML：

1. 外部起 Ray cluster（本仓库提供 pdsh 脚本）：
   ```bash
   export NODE_IP_LIST="ip1:8 ip2:8"   # 第一个是 head
   bash scripts/ray_start.sh
   ```
2. 在 YAML 根部加 `cluster:` 段：
   ```yaml
   cluster:
     ray_address: <head-ip>:6379   # 或 "auto" 若 service 跑在 head 上
     # namespace: reward-service-prod   # 可选，多服务共享同一 cluster 时用
   ```
3. 对需要跨机分布的 1-GPU reward，给 `RewardModelCfg` 加：
   ```yaml
   num_replicas: 2
   scheduling: spread   # 默认 pack = Ray 内建行为；spread = 透传 scheduling_strategy=SPREAD
   ```

**源码对应**：`reward_service/workers/pool.py::_init_ray` 读 `ClusterCfg.ray_address`；
`reward_service/workers/group.py::_actor_options` 把 `scheduling` 翻译为 Ray actor 的 `scheduling_strategy` kwarg。

**注意事项**：
- vLLM TP=2 这类跨节点 NCCL 通信的 reward 必须 `scheduling: pack` 且靠总 GPU 预算自然留在单机内——跨以太网 TP 会让吞吐崩塌。参考 `configs/service.cluster.example.yaml` 的 GenEval2 配置。
- `scheduling: spread` 只是 Ray 调度器 hint、不是 placement group 的强绑定；若实测 replica 仍挤在一台机，再考虑上 placement group。

### 5.4 加速路径（如果成为瓶颈）

| 瓶颈 | 应对 | 落点 |
|---|---|---|
| Ray pickle PIL 重复 | `ray.put(item)` 去重 | `server.py` 的 `dispatch` 改传 `ObjectRef` |
| 每步 round-robin 忽略 actor 负载 | 负载感知派发 | `group.py` 替换 `itertools.cycle` 为 load tracker |
| vLLM 首次加载慢 | 动态 batching / continuous batching | `unified_reward.py` / `geneval2.py` 换用 vLLM async engine |
| `/score` 同步聚合 | 流式返回部分结果 | `server.py` 改 streaming response |

上述都是 YAGNI 延后项 —— 真正观察到瓶颈再做，参考 DEVELOPMENT_LOG §9 的触发条件。

---

## 6. 配置到运行时对象的映射

方便调试时跟踪"YAML 的某字段最终变成了什么"：

```
YAML                                   config.py            workers/pool.py         scorers/<...>.py
────────────────────────────────────── ──────────────────── ───────────────────── ────────────────────

server:                                ServerCfg
  host: 0.0.0.0                          .host
  port: 8080                             .port                                     ( uvicorn 启动时用 )

rewards:                               list[RewardModelCfg]
  - name: clip                           .name             ─► WorkerPool._groups[.name]
    scorer: clip                         .scorer           ─► get_scorer_cls(.scorer)   ClipScorer
    num_replicas: 1                      .num_replicas     ─► len(WorkerGroup.actors)
    num_gpus: 1                          .num_gpus         ─► ScorerActor.options(num_gpus=.)
    num_cpus: 2                          .num_cpus         ─► ScorerActor.options(num_cpus=.)
    params:                              .params (dict)    ─► ScorerActor.remote(.scorer, .params)
      model_name: openai/…                                    → ClipScorer(**.params)
      weights_path: /apdcephfs_nj10/…                       → 走 resolve_model_path
      dtype: float32                                         → 走 resolve_dtype
```

对 **vLLM 类 scorer**（`unified_reward` / `geneval2`），`params` 里的 `dtype / enforce_eager / swap_space / quantization / seed / max_num_seqs / limit_mm_per_prompt / extra_llm_kwargs` 会走 `scorers/_common.py::build_vllm_llm_kwargs`，最终透传给 `vllm.LLM(**kwargs)`。详见 DEVELOPMENT_LOG §11。

---

## 7. 文件总览（按职责分类）

```
reward_service/
├── server.py            # HTTP gateway — /score /health /rewards
├── __main__.py          # CLI entry — argparse + uvicorn.run
├── config.py            # YAML → ServiceCfg / RewardModelCfg
├── schemas.py           # Pydantic HTTP schemas
├── logging_utils.py     # get_logger(name)
├── client.py            # Python SDK (RewardClient)
│
├── workers/             #  Ray 层
│   ├── pool.py          #  Ray init + group 注册表
│   ├── group.py         #  N replicas + round-robin
│   └── actor.py         #  @ray.remote ScorerActor
│
└── scorers/             #  模型层
    ├── base.py          #  BaseScorer + ScoreItem
    ├── registry.py      #  register / get_scorer_cls / _try_import
    ├── _common.py       #  共享工具: dtype, path, data_url, vLLM kwargs
    ├── clip.py          # ─┐
    ├── pickscore.py     #  │ transformers 类
    ├── imagereward.py   #  │
    ├── hpsv2_scorer.py  #  │
    ├── hpsv3_scorer.py  # ─┘
    ├── unified_reward.py# ─┐
    └── geneval2.py      # ─┘ vLLM 类
```

---

*架构图与源码以源码为准。若发现本文档与实际行为不一致，请优先信任源码并更新本文档。*
