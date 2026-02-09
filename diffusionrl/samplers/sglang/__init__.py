"""
SGLang Engine Samplers (Placeholder).

This module is reserved for future SGLang-based samplers.
SGLang provides efficient serving for large language models and
can be extended to support diffusion model inference.

Future implementations may include:
- SGLangSampler: SGLang-based sampler for distributed inference
- SGLangHunyuanSampler: HunyuanVideo sampler using SGLang backend

Engine:
- SGLangInferenceEngine: Placeholder engine interface for Ray actors

See README.md for implementation plans.
"""

from .client import (
    SGLangClient,
    SGLangClientError,
    SGLangProtocolError,
    SGLangTimeoutError,
)
from .engine import SGLangInferenceEngine

__all__ = [
    "SGLangClient",
    "SGLangClientError",
    "SGLangProtocolError",
    "SGLangTimeoutError",
    # Engine (placeholder)
    "SGLangInferenceEngine",
]
