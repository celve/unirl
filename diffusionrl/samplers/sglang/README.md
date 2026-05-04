# SGLang Engine Samplers

This directory is reserved for future SGLang-based samplers.

## Overview

SGLang is a fast serving framework for LLMs. The plan is to extend it for
diffusion model inference to enable:

- Distributed sampling across multiple nodes
- Efficient batched inference
- Dynamic batching for variable-length prompts

## Planned Implementation

### SGLangSampler

```python
class SGLangSampler(BaseSampler):
    """
    SGLang-based sampler for distributed diffusion inference.

    Features:
    - Distributed inference across multiple GPUs/nodes
    - Dynamic batching for efficient resource utilization
    - Integration with SGLang's scheduling system
    """
    pass
```

### SGLangHunyuanSampler

```python
class SGLangHunyuanSampler(SGLangSampler):
    """
    HunyuanVideo-specific SGLang sampler.

    Features:
    - Optimized for HunyuanVideo architecture
    - Video-specific batching strategies
    - Memory-efficient video generation
    """
    pass
```

## Timeline

- Phase 1: Research SGLang diffusion support (Q1 2025)
- Phase 2: Implement basic SGLangSampler (Q2 2025)
- Phase 3: Add video model support (Q3 2025)

## Dependencies

```
sglang[diffusion] @ git+https://github.com/celve/sglang.git@diffusionrl#subdirectory=python
```

For source-mode development against a sibling SGLang checkout:

```bash
# Dev mode: install your sibling sglang clone editable so source changes take effect without reinstall.
pip install -e ../sglang/python
```

## Usage (Future)

```python
from diffusionrl.samplers.sglang import SGLangHunyuanSampler

sampler = SGLangHunyuanSampler(
    model_path="path/to/hunyuan",
    tensor_parallel_size=4,
    pipeline_parallel_size=2,
)

output = sampler.sample(
    prompts=["a cat running"],
    num_inference_steps=50,
)
```
