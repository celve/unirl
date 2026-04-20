"""Shared rollout pipeline mixin for actors that generate and score samples.

Both ``TrainActor`` (direct-sampling mode) and ``RolloutActor`` use the
same generate → reward → advantage pipeline.  This mixin eliminates the
duplication so bug-fixes (e.g. PIL conversion) only need to land once.

Host class contract
-------------------
The host must provide:

- ``self.engine``   – a ``BaseRolloutEngine`` (for ``decode_latents``)
- ``self.algorithm`` – a ``BaseAlgorithm`` (owns ``compute_advantages``)
- ``self.generate(request)`` – returns ``RolloutResponse``
- ``self.get_buffer(handle)`` / ``self.put_buffer(key, value)`` – from ``Buffer``
- ``self._ensure_reward_pipeline()`` – returns ``RewardPipeline``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from diffusionrl.types.request import RolloutRequest
    from diffusionrl.types.response import RolloutResponse
    from diffusionrl.transfer.buffer import BufferHandle


class RolloutPipelineMixin:
    """Reusable generate → reward → advantage pipeline methods."""

    # -- Buffered generation -----------------------------------------------

    def generate_buffered(self, request: "RolloutRequest") -> List["BufferHandle"]:
        response = self.generate(request)
        responses = response.split()
        return [self.put_buffer(response.to_meta(), response) for response in responses]

    # -- Reward attachment -------------------------------------------------

    def attach_reward(self, handle: "BufferHandle") -> None:
        """Decode latents to PIL and attach reward scores to a buffered response."""
        from diffusionrl.types.response import RolloutResponse

        response: RolloutResponse = self.get_buffer(handle)
        if not getattr(self, "_logged_decode_diag", False):
            import logging as _logging
            import numpy as _np
            _diag_logger = _logging.getLogger("diffusionrl.ray.mixins.rollout_pipeline")
            _final_lat = response.samples.latents
            _lat_shape = tuple(_final_lat.shape) if _final_lat is not None else None
            _lat_pairs = []
            if _final_lat is not None and _final_lat.dim() > 1 and _final_lat.shape[0] >= 2:
                for _i in range(min(_final_lat.shape[0], 4)):
                    for _j in range(_i+1, min(_final_lat.shape[0], 4)):
                        _lat_pairs.append((_i, _j, float((_final_lat[_i].float()-_final_lat[_j].float()).abs().mean().item())))
            _dec = response.samples.decoded_images
            _dec_info = ("none" if _dec is None else
                         f"list_len={len(_dec)}, type0={type(_dec[0]).__name__}")
            _img_pairs = []
            if _dec is not None and len(_dec) >= 2:
                _arrs = [_np.asarray(_dec[i], dtype=_np.float32) for i in range(min(len(_dec), 4))]
                for _i in range(len(_arrs)):
                    for _j in range(_i+1, len(_arrs)):
                        _img_pairs.append((_i, _j, float(_np.abs(_arrs[_i]-_arrs[_j]).mean()), tuple(_arrs[_i].shape)))
            _diag_logger.warning(
                "attach_reward decode diag (one-shot): final_latents.shape=%s lat_pairwise_abs_diff=%s | decoded=%s img_pairwise_abs_diff=%s",
                _lat_shape, _lat_pairs, _dec_info, _img_pairs,
            )
            self._logged_decode_diag = True
        if response.samples.decoded_images is None:
            from diffusionrl.utils.media import tensor_to_pil
            decoded = self.engine.decode_latents(response.samples.latents)
            response.samples.decoded_images = tensor_to_pil(decoded)
        self._ensure_reward_pipeline().score_and_attach(response)

    # -- Advantage computation ---------------------------------------------

    def compute_advantages(self, handle: "BufferHandle") -> None:
        """Compute advantages for a buffered response by delegating to ``self.algorithm``."""
        from diffusionrl.types.response import RolloutResponse

        response: RolloutResponse = self.get_buffer(handle)
        rewards = response.samples.rewards
        if rewards is None:
            raise RuntimeError("Cannot compute advantages: rewards not attached.")
        response.samples.advantages = self.algorithm.compute_advantages(
            rewards=rewards,
            group_ids=list(response.request.prompts.group_ids),
        )

    # -- Fused pipelines ---------------------------------------------------

    def run_rollout_pipeline(self, request: "RolloutRequest") -> List["RolloutResponse"]:
        """Fused actor-side rollout pipeline: generate + reward + advantages.

        Collapses what would otherwise be 3·N Ray RPCs per actor per
        rollout step into a single round-trip.

        Advantages are computed **once across every group this actor sees**
        rather than per-handle, so ``algorithm.use_global_std=True`` actually
        sees the full cross-group reward distribution. Per-group mean is still
        honored via ``group_ids``.
        """
        import torch

        from diffusionrl.types.response import RolloutResponse

        handles = self.generate_buffered(request)
        for h in handles:
            self.attach_reward(h)
        responses: List[RolloutResponse] = [self.get_buffer(h) for h in handles]
        for r in responses:
            if r.samples.rewards is None:
                raise RuntimeError("Cannot compute advantages: rewards not attached.")
        all_rewards = torch.cat([r.samples.rewards for r in responses])
        all_group_ids: List[str] = [
            gid for r in responses for gid in r.request.prompts.group_ids
        ]
        all_advantages = self.algorithm.compute_advantages(
            rewards=all_rewards,
            group_ids=all_group_ids,
        )
        if not getattr(self, "_logged_advantage_diag", False):
            import logging as _logging
            _diag_logger = _logging.getLogger("diffusionrl.ray.mixins.rollout_pipeline")
            from collections import Counter as _Counter
            _gid_counts = _Counter(all_group_ids)
            _diag_logger.warning(
                "Rollout-pipeline advantage diag (one-shot): rewards.shape=%s rewards[abs_mean=%.4E std=%.4E unique_count=%d head=%s] "
                "group_ids[total=%d unique=%d sizes=%s head=%s] advantages[abs_mean=%.4E std=%.4E nonzero_frac=%.4f head=%s]",
                tuple(all_rewards.shape),
                float(all_rewards.abs().mean().item()),
                float(all_rewards.std().item()) if all_rewards.numel() > 1 else 0.0,
                int(torch.unique(all_rewards).numel()),
                all_rewards.flatten()[:16].tolist(),
                len(all_group_ids),
                len(_gid_counts),
                dict(list(_gid_counts.items())[:8]),
                all_group_ids[:16],
                float(all_advantages.abs().mean().item()),
                float(all_advantages.std().item()) if all_advantages.numel() > 1 else 0.0,
                float((all_advantages != 0).float().mean().item()),
                all_advantages.flatten()[:16].tolist(),
            )
            self._logged_advantage_diag = True
        offset = 0
        for r in responses:
            n = r.samples.rewards.shape[0]
            r.samples.advantages = all_advantages[offset:offset + n]
            offset += n
        return responses

    def run_eval_pipeline(self, request: "RolloutRequest") -> List["RolloutResponse"]:
        """Eval pipeline: generate + reward, no advantages."""
        handles = self.generate_buffered(request)
        for h in handles:
            self.attach_reward(h)
        return [self.get_buffer(h) for h in handles]
