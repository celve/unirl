# HPSv3 Reward Model Installation

HPSv3 is a Qwen2-VL-7B based human preference scorer. It outputs `[mu, sigma]` per image; we use `mu` (index 0) as the reward score.

- Paper: https://arxiv.org/abs/2411.07232
- Repo: https://github.com/MizzenAI/HPSv3

## 1. Install hpsv3 package

```bash
git clone https://github.com/MizzenAI/HPSv3.git
cd HPSv3
pip install -e .
```

> **Note**: hpsv3 pins `transformers==4.45.2` in pyproject.toml, which may
> downgrade your current transformers version. Check with
> `pip show transformers` after installation and reinstall if needed.

## 2. Install missing dependencies

hpsv3 has undeclared dependencies not listed in its pyproject.toml:

```bash
pip install matplotlib qwen-vl-utils omegaconf tensorboard
```

Other deps (`torch`, `transformers`, `safetensors`, `peft`, `accelerate`, etc.)
are already satisfied by the DiffusionRL environment.

## 3. Download base model (Qwen2-VL-7B-Instruct)

HPSv3 uses `Qwen/Qwen2-VL-7B-Instruct` as the backbone. If the machine cannot
access HuggingFace, download it locally:

```bash
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --local-dir /path/to/Qwen2-VL-7B-Instruct
```

Then update the HPSv3 config (`HPSv3/hpsv3/config/HPSv3_7B.yaml`):

```yaml
# before
model_name_or_path: "Qwen/Qwen2-VL-7B-Instruct"
# after
model_name_or_path: "/path/to/Qwen2-VL-7B-Instruct"
```

## 4. HPSv3 checkpoint

By default, `HPSv3RewardInferencer` auto-downloads `HPSv3.safetensors` from
HuggingFace (`MizzenAI/HPSv3`). For offline use, download manually:

```bash
huggingface-cli download MizzenAI/HPSv3 HPSv3.safetensors --local-dir /path/to/hpsv3_ckpt
```

Current DiffusionRL integration follows the same style as other built-in image
scorers: it initializes `HPSv3RewardInferencer` with the library defaults and
does not currently expose a dedicated DiffusionRL CLI flag for custom HPSv3
checkpoint paths.

## 5. Known issues

### transformers version

hpsv3 pins `transformers==4.45.2` in pyproject.toml. `pip install -e .` may
downgrade transformers. If other components (e.g. SD3 pipeline) need a newer
version, reinstall after hpsv3 setup.

### Missing matplotlib

`hpsv3/model/qwen2vl_trainer.py` imports `matplotlib` but it is not declared in
`pyproject.toml`. Install it manually (step 2).

## 6. GPU memory

HPSv3 is a 7B model (~14GB in bf16). When using `local_reward_device=cuda`
(shared GPU with training), reduce batch geometry to avoid OOM.
