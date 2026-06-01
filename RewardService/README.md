# Reward Service

统一的 T2I reward 推理服务：FastAPI 网关 + Ray worker groups。每个 reward 模型独占自己的 GPU，彼此资源隔离。

## 支持的 reward 模型

| 名字 | 框架 | 备注 |
|---|---|---|
| `clip` | transformers | openai/clip-vit-large-patch14，cosine similarity |
| `pickscore` | transformers | yuvalkirstain/PickScore_v1 |
| `imagereward` | 官方 `image-reward` pip 包 | ImageReward-v1.0 |
| `hpsv2` | 官方 `hpsv2` pip 包 | 支持 v2.0 / v2.1 |
| `hpsv3` | 官方 `hpsv3` pip 包 | 基于 Qwen2-VL |
| `unified_reward` | vLLM | UnifiedReward-2.0-qwen3vl 2B/4B，解析文本取 Alignment/Coherence/Style |
| `geneval2` | vLLM | VQAScore（配 `dataset_path` 时走 Soft-TIFA 多问题评分；未配时退化为单问模板） |
| `qwen35_vlm` | vLLM | Qwen3.5-35B-A3B / 122B-A10B 等 MoE VLM 作 reward；默认子指标 Consistency / Realism / Aesthetic Quality（WISE-style 模板，可在 YAML 替换） |

## 架构

```
Client ──HTTP──▶ FastAPI Gateway ──Ray actor call──▶ WorkerGroup(reward=X)
                                                     ├── actor_0 (GPU k)
                                                     └── actor_1 (GPU k+1)
```

- 每个 reward 是一个 `WorkerGroup`，持有若干 Ray actor（可配副本数）
- 每个 actor 通过 Ray 的 `num_gpus=N` 独占 N 张卡，不与其他 reward 共享
- 路由按 `required_rewards` 派发到对应 group，round-robin 选 actor

完整架构文档（拓扑、时序、抽象层、扩展点）见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## 安装

### Base 环境

```bash
# 前提：已激活 Python 3.12 环境，torch + nccl 已预装
./install.sh    # 装 ray, fastapi, uvicorn, pillow 等 base 依赖
```

Per-scorer 的依赖（transformers、vllm、hpsv2 等）**不在这里装**——它们声明在 `envs/*.txt` 中，由 Ray `runtime_env` 在 actor 首次启动时自动 pip install 到隔离的 virtualenv 里。

### 调用方（只发 HTTP，不起服务）

```bash
pip install .    # 只装 requests + Pillow
```

### 依赖隔离架构

每个 scorer 有自己的 `envs/<scorer>.txt` requirements 文件，YAML 里通过 `runtime_env` 字段引用：

```yaml
rewards:
  - name: imagereward
    scorer: imagereward
    runtime_env: envs/imagereward.txt   # transformers==4.45.2
    ...
  - name: unified_reward
    scorer: unified_reward
    runtime_env: envs/unified_reward.txt  # vllm==0.11.0 (transformers>=4.55.2)
    ...
```

这样 imagereward 和 vllm 可以用不同版本的 transformers，互不冲突。

> 首次启动时 Ray 会 pip install 每个 venv（可能慢几分钟）。后续启动复用缓存，秒级。

## 启动

```bash
cp configs/service.example.yaml configs/service.yaml
# 编辑 configs/service.yaml：取消注释要启用的 reward、填 weights_path、调 num_gpus/replicas
python -m reward_service --config configs/service.yaml
```

> 首次启动时 Ray 会 pip install 每个 scorer 的 venv（可能慢几分钟）。后续启动复用缓存，秒级。

### 多机部署

```bash
export NODE_IP_LIST="ip1:8 ip2:8"      # 第一项是 head
bash scripts/ray_start.sh              # pdsh 起 Ray cluster
python -m reward_service --config configs/service.cluster.example.yaml
# 下线：
bash scripts/ray_stop.sh
```

多机配置只需在 YAML 里加 `cluster.ray_address` 字段，详见 `configs/service.cluster.example.yaml` 注释。

## 调用

### Python SDK

```python
from PIL import Image
from reward_service.client import RewardClient, RewardRequest

client = RewardClient("http://localhost:8080")  # 跨机时换成服务机 IP
scores = client.score([
    RewardRequest(history=[("a cute dog", Image.open("dog.png"))],
                  required_rewards=["hpsv2", "clip"]),
    RewardRequest(history=[("a cute cat", Image.open("cat.png"))],
                  required_rewards=["hpsv2", "pickscore"]),
])
# scores -> [{"hpsv2": {"hpsv2": 0.27}, "clip": {"clip": 0.31}}, ...]
```

### Batch 调用

`client.score([...])` 接受任意长度的 request 列表。服务端把相同 reward 的 N 条请求聚到同一个 actor 一次算完。

```python
# 一条 prompt × 多候选图（RLHF 最常见）
prompt = "a cute dog running in the park"
candidates = [Image.open(f"cand_{i}.png") for i in range(8)]
scores = client.score([
    RewardRequest(history=[(prompt, img)], required_rewards=["clip", "hpsv2", "pickscore"])
    for img in candidates
])
best = max(range(8), key=lambda i: scores[i]["hpsv2"]["hpsv2"])
```

并发由 `configs/service.example.yaml` 中每个 reward 的 `max_concurrency × num_replicas` 决定。`server.score_timeout_s`（默认 120s）对每个 reward 独立计时，超时写入 `response.errors[i][reward]`，不阻塞同批其它 reward。

### 跨机调用

URL 换成服务机 IP，代码其余不变：

```bash
python3 scripts/remote_client_example.py --url http://10.1.2.3:8080
# 或不装 SDK，纯 requests + Pillow：
python3 scripts/remote_client_zero_deps.py --url http://10.1.2.3:8080 --image cand.jpg
```

### 不装 SDK，手写 HTTP

```python
import base64, io, requests
from PIL import Image

def _b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("ascii")

payload = {"requests": [
    {"history": [{"text": "a cute dog", "image_b64": _b64(Image.open("dog.png"))}],
     "required_rewards": ["clip", "hpsv2"]},
]}
resp = requests.post("http://10.1.2.3:8080/score", json=payload, timeout=120)
resp.raise_for_status()
body = resp.json()
# body["results"][i][reward][sub_metric] -> float
# body["errors"][i][reward] -> str（只在该 reward 失败时存在）
```

### API 参考

POST `/score` body：

| 字段 | 类型 | 说明 |
|---|---|---|
| `requests[i].history[j].text` | str | prompt |
| `requests[i].history[j].image_b64` | str | 图片 base64（PIL 可读的任何格式；服务端 `convert("RGB")`） |
| `requests[i].required_rewards` | list[str] | 需全部在 `GET /rewards` 列表里 |
| `requests[i].metadata` | dict \| null | 可选，原样透传 |

响应：

```json
{
  "results": [ { "clip": {"clip": 0.31}, "hpsv2": {"hpsv2": 0.27} } ],
  "errors":  [ { } ]
}
```

`results[i]` 和 `errors[i]` 与 `requests[i]` 一一对应。某 reward 失败时分数缺席，错误字符串（含异常类名 + 消息 + cause 链）写进 `errors[i][reward]`。

其它端点：

- `GET /health` → `{"status": "ok", "rewards": {name: [replica_states...]}}`
- `GET /rewards` → `{"rewards": [...]}`

## 测试

```bash
pytest -m "not gpu and not slow and not integration"   # CPU-only 单元测试
pytest tests/integration/ -m integration -v            # venv 安装集成测试（需 Ray + 网络）
pytest                                                 # 全量（需 GPU）
```

## Venv 检查

验证每个 scorer 的 venv 安装状态和隔离情况：

```bash
python scripts/check_venvs.py --standalone
```

## 压测

服务起来后，用 `scripts/bench_concurrent.py` 施加可控并发，测单请求延迟分布和整体吞吐。

```bash
# 单点：1000 条请求、200 并发，只打 clip
python3 scripts/bench_concurrent.py \
    --url http://10.1.2.3:8080 \
    --concurrency 200 --total 1000 --rewards clip

# 扫档：同样 1000 条请求，在 100 / 500 / 1000 / 2000 并发下各跑一轮
python3 scripts/bench_concurrent.py \
    --url http://10.1.2.3:8080 \
    --sweep 100 500 1000 2000 --total 1000 --rewards clip

# Per-reward 对比：N 个 reward 各自独立跑一轮，末尾一张横向对比表
# 单请求 /score 同时要 N 个 reward 时延迟会被最慢那个拉齐（HTTP 一次返回），
# 想看每个 reward 的纯延迟就用这个模式
python3 scripts/bench_concurrent.py \
    --url http://10.1.2.3:8080 \
    --concurrency 200 --total 500 \
    --rewards clip,hpsv2,hpsv3 --per-reward-isolated
```

输出包含每个请求的 min / mean / max 延迟，以及 p50/p90/p95/p99、吞吐、传输错误和服务端 per-reward 失败数。扫档和 per-reward 模式最后会打一张横向对比表。

## 设计约定

- `history` 是 `list[(text, image)]`，scorer 只看最后一对（T2I 场景）
- reward 返回 `dict[str, float]`（子指标名 → 分数），支持多子指标
- 资源隔离：一张 GPU 同一时刻只属于一个 reward group
- 错误隔离：某 reward 失败不影响同批其它 reward
- 依赖隔离：每个 scorer 的 pip 依赖在独立 venv 中，不同 transformers 版本互不冲突

## 已知限制

- **首次启动慢**：Ray 需要 pip install 每个 venv（vllm 尤其慢，~10 分钟）。后续启动复用缓存。
- **Ray PIL 序列化**：一个 request 打 N 个 reward 时 PIL 图像被 pickle N 次。大 batch 可改 `ray.put` + ObjectRef。
- **GenEval2 Soft-TIFA**：需配 `dataset_path`；prompt 无匹配时退化为单问模板并 warning。
- **UnifiedReward 解析**：依赖正则提取分数；模型输出漂移时降级到 NaN 并 warning。
