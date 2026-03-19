"""Built-in algorithm registry metadata."""

from __future__ import annotations


DEFAULT_ALGORITHM_PATHS = {
    "grpo": "diffusionrl.algorithms.grpo.GRPOAlgorithm",
    "nft": "diffusionrl.algorithms.nft.NFTAlgorithm",
    "mix_grpo": "diffusionrl.algorithms.mix_grpo.MixGRPOAlgorithm",
}


__all__ = ["DEFAULT_ALGORITHM_PATHS"]
