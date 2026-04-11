# EditReward Reward Model Installation

EditReward is a VLM-based reward model for instruction-guided image editing.
Unlike image-only scorers such as PickScore or HPSv3, it scores a triplet:
`(source image, edited image, edit instruction)`.

- Paper: https://arxiv.org/abs/2509.26346
- Repo: https://github.com/TIGER-AI-Lab/EditReward
- Models: https://huggingface.co/collections/TIGER-Lab/editreward-68ddf026ef9eb1510458abc6

## 1. Install EditReward

```bash
git clone https://github.com/TIGER-AI-Lab/EditReward.git
cd EditReward

# Python 3.10 is recommended by the upstream project.
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

pip install -U datasets pillow openai megfile sentencepiece deepspeed fire omegaconf \
  matplotlib peft trl==0.8.6 tensorboard scipy transformers==4.57.0 accelerate \
  requests packaging pandas opencv-python einops timm huggingface_hub qwen-vl-utils \
  av regex safetensors tqdm

# Optional but recommended by the upstream project.
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install -e .
```

> EditReward has a relatively heavy dependency stack (`transformers`, `trl`,
> `deepspeed`, `qwen-vl-utils`, `peft`, etc.). If your DiffusionRL environment
> already pins conflicting versions, prefer installing EditReward into a
> dedicated environment first and then reproducing the final compatible set in
> the training environment.

## 2. Download the EditReward checkpoint

DiffusionRL expects `--reward.reward-model-ckpt-path` to point at a checkpoint
directory that contains either `model.pth` or `model.safetensors`.

Example:

```bash
huggingface-cli download TIGER-Lab/EditReward-Qwen2.5-7B-VL \
  --local-dir /path/to/editreward_qwen25_7b
```

Expected layout:

```text
/path/to/editreward_qwen25_7b/
├── model.pth
# or
├── model.safetensors
└── ...
```

If you use a different upstream release (for example MiMo-VL or Qwen3-VL based
variants), keep the same rule: the path passed to DiffusionRL must be the
directory that directly contains the saved model weights.

## 3. Choose the matching EditReward config

`EditRewardInferencer` accepts a YAML config that determines the backbone /
processor setup. Common upstream configs include:

- `config/EditReward-Qwen2.5-7B-VL.yaml`
- `config/EditReward-Qwen3-VL.yaml`
- `config/EditReward-MiMo-VL-7B-SFT-2508.yaml`

If you installed EditReward with `pip install -e .`, package-relative config
paths work. Otherwise, pass an absolute config path.

DiffusionRL integration currently reads the config from:

- `EDITREWARD_CONFIG_PATH` environment variable, if set
- otherwise EditReward's own default config resolution

Example:

```bash
export EDITREWARD_CONFIG_PATH=/path/to/EditReward/EditReward/config/EditReward-Qwen2.5-7B-VL.yaml
```

## 4. DiffusionRL integration

The built-in DiffusionRL scorer name is:

- `editreward`
- alias: `edit_reward`

Minimal usage:

```bash
python -m diffusionrl.train \
  --reward.reward-components editreward \
  --reward.reward-model-ckpt-path /path/to/editreward_qwen25_7b \
  --reward.local-reward-device cuda
```

Optional environment variables supported by the built-in scorer:

```bash
export EDITREWARD_PYTHON_PATH=/path/to/EditReward
export EDITREWARD_CONFIG_PATH=/path/to/EditReward/EditReward/config/EditReward-Qwen2.5-7B-VL.yaml
export EDITREWARD_REWARD_DIM=overall_detail
export EDITREWARD_RM_HEAD_TYPE=ranknet_multi_head
export EDITREWARD_SOURCE_IMAGE_KEY=source_image_path
```

Notes:

- `EDITREWARD_PYTHON_PATH` is only needed when EditReward is not importable from
  the current Python environment.
- If unset, DiffusionRL also tries a few common local checkout locations such as
  `third_party/EditReward` and `My_Code/EditReward`.
- `EDITREWARD_REWARD_DIM=overall_detail` is the current default used by the
  DiffusionRL scorer wrapper.

## 5. Dataset / metadata contract

EditReward needs the source image in addition to the edited image and prompt.
In DiffusionRL, the edited image comes from rollout decoding, and the source
image must be provided through per-sample metadata in the prompt dataset.

Recommended JSON / JSONL sample:

```json
{
  "prompt": "Add a green bowl on the branch",
  "metadata": {
    "source_image_path": "/path/to/source.png"
  }
}
```

The built-in scorer accepts the source image from common metadata keys such as:

- `source_image_path`
- `source_image`
- `image_src`
- `input_image_path`
- `input_image`
- `original_image_path`
- `condition_image_path`

The metadata value may be:

- a local filesystem path
- a URL / URI supported by EditReward's image loader
- a nested dict like `{"path": "/path/to/source.png"}`

If no source image is present in metadata, reward computation will fail fast.

## 6. Quick verification

First verify the upstream package works by itself:

```bash
python3 -c "
from EditReward import EditRewardInferencer

inferencer = EditRewardInferencer(
    config_path='config/EditReward-Qwen2.5-7B-VL.yaml',
    checkpoint_path='/path/to/editreward_qwen25_7b',
    device='cpu',
    reward_dim='overall_detail',
    rm_head_type='ranknet_multi_head',
)
print('OK: EditReward import + model init succeeded')
"
```

Then verify DiffusionRL can resolve the built-in scorer:

```bash
python3 -c "
from diffusionrl.reward.scorers.registry import resolve_builtin_reward_scorer_class
cls = resolve_builtin_reward_scorer_class('editreward')
print(cls.__name__)
"
```

## 7. Known issues

### Version conflicts

EditReward recommends a fairly new `transformers==4.57.0` and also depends on
`trl==0.8.6`, `deepspeed`, and `qwen-vl-utils`. If another reward model or
training component pins older versions, reconcile the environment before large
training runs.

### Checkpoint path format

DiffusionRL does not accept a single file path for EditReward. Pass the
directory containing `model.pth` or `model.safetensors`.

### Missing source image metadata

This is the most common integration error. If you see a runtime failure saying
that EditReward could not find a source image, inspect the dataset metadata and
make sure it is sample-aligned after prompt expansion.

## 8. GPU memory

EditReward is a large VLM reward model. Exact memory usage depends on the
backbone variant (Qwen2.5-VL / Qwen3-VL / MiMo-VL), precision, and batch size.

Practical guidance:

- prefer `--reward.local-reward-device cpu` first for smoke tests
- if using `cuda`, start with a very small reward batch size
- when sharing GPUs with rollout / training, reduce rollout geometry before
  increasing reward batch size
