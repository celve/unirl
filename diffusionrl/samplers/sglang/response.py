from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Self, Sequence, Tuple

import torch

from diffusionrl.config.require import require
from diffusionrl.samplers.utils.embeddings import fuse_text_encoder_outputs
from diffusionrl.samplers.utils.flux import build_flux_image_ids, build_flux_text_ids
from diffusionrl.samplers.utils.media import decode_sample
from diffusionrl.sde.runtime import get_sigma_schedule
from diffusionrl.types.forward_context import ForwardContext, get_forward_context_cls
from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.sample import LogProbData, RolloutSamples
from diffusionrl.types.trajectory_store import Trajectory, compute_trajectory_positions

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.utils import GenerationResult

logger = logging.getLogger(__name__)


@dataclass
class SGLangRolloutResponse:
    """Typed wrapper around SGLang's ``DiffGenerator.generate()`` output.

    Holds the raw ``List[GenerationResult]`` SGLang returned. All
    normalization (trajectory cat + alignment, log_prob alignment, embed
    fusion, decoded media extraction) is deferred to ``to_rollout_samples``
    so the intermediate state stays close to SGLang's wire contract — the
    intermediate type IS the typed SGLang response list.
    """

    results: List["GenerationResult"]

    @classmethod
    def from_sglang_results(cls, results: List["GenerationResult"]) -> Self:
        require(bool(results), "SGLang generator returned no results for prompt batch")
        return cls(results=list(results))

    def to_rollout_samples(
        self,
        request: RolloutRequest,
        *,
        model_type: str,
        num_inference_steps: int,
        shift: float,
        sde_indices: Optional[List[int]],
        guidance_scale: float,
        height: int,
        width: int,
        use_native_logprob: bool,
        return_decoded_for_reward: bool,
    ) -> RolloutSamples:
        """Normalize the stored SGLang results into a generic ``RolloutSamples``."""
        results = self.results

        # 1. Trajectory + final latents + timestep alignment
        trajectory_items: List[torch.Tensor] = []
        for result in results:
            traj = result.trajectory_latents
            require(traj is not None, "SGLang result missing trajectory_latents")
            trajectory_items.append(traj.detach().cpu())
        trajectories_tensor = torch.cat(trajectory_items, dim=0)
        timesteps, step_indices = self._derive_timestep_alignment(
            trajectories_tensor=trajectories_tensor,
            num_inference_steps=num_inference_steps,
            shift=shift,
            results=results,
        )
        final_latents = trajectories_tensor[:, -1].clone()

        # Selective trim — when only a subset of trajectory positions is
        # ever read by the SDE step set, drop unused columns to save GPU
        # memory on the training side. Column i == trajectory position i
        # (T+1 invariant enforced by _derive_timestep_alignment).
        traj_len = int(trajectories_tensor.shape[1])
        trimmed_cols: Optional[List[int]] = None
        if sde_indices is not None and len(sde_indices) < num_inference_steps:
            needed_original = set(compute_trajectory_positions(set(sde_indices), num_inference_steps))
            keep_cols = sorted(p for p in needed_original if 0 <= p < traj_len)
            if keep_cols and len(keep_cols) < traj_len:
                trimmed_cols = keep_cols

        if trimmed_cols is not None:
            trimmed = trajectories_tensor[:, trimmed_cols]
            trajectory_store: Optional[Trajectory] = Trajectory.from_selective(trimmed, trimmed_cols, traj_len)
        else:
            trajectory_store = Trajectory.from_full(trajectories_tensor)
        del trajectories_tensor

        # 2. Native log-probs (only when policy says so; replay path leaves None).
        # Caller selects native via ``use_native_logprob`` — under replay we
        # bypass extraction entirely. Under native we fail fast on any
        # missing-per-result or shape-mismatched tensor: silently falling
        # back to replay would change the log_prob source the user asked for
        # and produce wrong GRPO ratios at the next training step.
        merged_log_probs: Optional[LogProbData] = None
        if use_native_logprob:
            per_result_log_probs: List[Optional[torch.Tensor]] = [
                result.trajectory_log_probs.detach().cpu() if result.trajectory_log_probs is not None else None
                for result in results
            ]
            missing = [i for i, lp in enumerate(per_result_log_probs) if lp is None]
            if missing:
                raise RuntimeError(
                    f"logprob_source='native' but SGLang did not return usable trajectory_log_probs "
                    f"for {len(missing)}/{len(results)} result(s) (first missing index={missing[0]}). "
                    f"Either pin a SGLang build that emits trajectory_log_probs of shape [B, T] or [T] "
                    f"or switch logprob_source='replay' to recompute on the training side."
                )
            log_prob_tensor = torch.cat([lp for lp in per_result_log_probs if lp is not None], dim=0)
            expected_steps = int(step_indices.shape[0]) - 1
            if int(log_prob_tensor.shape[1]) != expected_steps:
                raise RuntimeError(
                    f"logprob_source='native' but SGLang trajectory_log_probs shape {tuple(log_prob_tensor.shape)} "
                    f"does not match expected second dim={expected_steps} "
                    f"(= step_indices length - 1, with num_inference_steps={num_inference_steps}). "
                    f"This points at a denoising-step / SGLang version mismatch — fix the source rather than "
                    f"fall back to replay silently."
                )
            step_keys = step_indices[:expected_steps].tolist()
            merged_log_probs = LogProbData.from_dict({int(k): log_prob_tensor[:, i] for i, k in enumerate(step_keys)})

        # 3. Forward context (prompt embeddings + model-specific extras)
        forward_context = self._build_forward_context_from_results(
            results=results,
            model_type=model_type,
            batch_size=int(final_latents.shape[0]),
            height=height,
            width=width,
            guidance_scale=guidance_scale,
        )

        # 4. Decoded media for reward
        decoded_images: Optional[List[torch.Tensor]] = None
        decoded_video_tensors: List[torch.Tensor] = []
        if return_decoded_for_reward:
            decoded_images = []
            for result in results:
                canonical = decode_sample(result.samples)
                if canonical is None:
                    continue
                # ``decoded_images`` always carries a 3D ``[C,H,W]`` tensor: full
                # image for 3D, middle frame for video. PIL conversion is
                # deferred to ``attach_media_preview`` (the only consumer that
                # actually needs PIL); reward handlers consume the float tensor.
                preview = canonical if canonical.dim() == 3 else canonical[:, canonical.shape[1] // 2]
                decoded_images.append(preview)
                if canonical.dim() == 4:
                    decoded_video_tensors.append(canonical)
            if not decoded_images:
                decoded_images = None
            if decoded_images is None and not decoded_video_tensors:
                logger.warning(
                    "SGLang generate(return_decoded_for_reward=True) returned no decodable media. "
                    "Reward stage will fail without decoded media."
                )

        decoded_videos = torch.stack(decoded_video_tensors, dim=0) if decoded_video_tensors else None

        return RolloutSamples(
            latents=final_latents,
            timesteps=timesteps,
            step_indices=step_indices,
            sampling_params=request.sampling_params,
            prompts=request.prompts,
            trajectories=trajectory_store,
            log_probs=merged_log_probs,
            forward_context=forward_context,
            decoded_images=decoded_images,
            decoded_videos=decoded_videos,
        )

    # ------------------------------------------------------------------
    # Static extractors
    # ------------------------------------------------------------------

    @classmethod
    def _derive_timestep_alignment(
        cls,
        *,
        trajectories_tensor: torch.Tensor,
        num_inference_steps: int,
        shift: float,
        results: Sequence[Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        traj_len = int(trajectories_tensor.shape[1])
        require(
            traj_len == int(num_inference_steps) + 1,
            f"SGLang trajectory must be T+1 (got {traj_len}, expected {num_inference_steps + 1}). "
            f"Modern SGLang prepends initial latents at "
            f"sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py — "
            f"upgrade SGLang or fix the sampler to emit a T+1 trajectory.",
        )
        timesteps = get_sigma_schedule(int(num_inference_steps), shift=float(shift)).cpu()
        step_indices = torch.arange(timesteps.shape[0], dtype=torch.long)
        require(
            int(timesteps.shape[0]) == traj_len,
            f"SGLang timestep/trajectory mismatch: {int(timesteps.shape[0])} vs {traj_len}",
        )
        cls._verify_sglang_timesteps(results, expected=timesteps)
        return timesteps, step_indices

    @staticmethod
    def _verify_sglang_timesteps(
        results: Sequence[Any],
        *,
        expected: torch.Tensor,
        atol: float = 1e-5,
        rtol: float = 1e-4,
    ) -> None:
        """Cross-check SGLang's emitted ``trajectory_timesteps`` against the
        client-recomputed sigma schedule.

        The client recomputes timesteps via ``get_sigma_schedule(num_steps, shift)``
        and uses those values for log-prob math. SGLang internally computes
        the same schedule on its side; if those two ever drift (sigma
        formula change, shift propagation bug, num_steps mismatch), all
        downstream math is silently wrong. Raises ``RuntimeError`` on any
        divergence.
        """
        expected_f32 = expected.to(torch.float32)
        for i, result in enumerate(results):
            actual = getattr(result, "trajectory_timesteps", None)
            if actual is None:
                raise RuntimeError(
                    f"Result index {i} missing trajectory_timesteps; cannot cross-check "
                    f"SGLang's sigma schedule against the client-recomputed one. "
                    f"Either upgrade SGLang to emit trajectory_timesteps or pin a build "
                    f"that does — silent agreement on timesteps is not safe."
                )
            actual_t = actual.detach().cpu() if torch.is_tensor(actual) else torch.as_tensor(actual)
            actual_t = actual_t.to(torch.float32)
            if actual_t.shape != expected_f32.shape:
                raise RuntimeError(
                    f"SGLang trajectory_timesteps shape mismatch on result {i}: "
                    f"got {tuple(actual_t.shape)}, expected {tuple(expected_f32.shape)}. "
                    f"Investigate sigma_schedule formula drift on the SGLang side."
                )
            if not torch.allclose(actual_t, expected_f32, atol=atol, rtol=rtol):
                max_diff = (actual_t - expected_f32).abs().max().item()
                raise RuntimeError(
                    f"SGLang trajectory_timesteps value mismatch on result {i}: "
                    f"max abs diff={max_diff:.3e} (atol={atol:g}, rtol={rtol:g}). "
                    f"Client schedule (head): {expected_f32.tolist()[:5]}; "
                    f"SGLang returned (head): {actual_t.tolist()[:5]}. "
                    f"Verify that get_sigma_schedule(num_steps, shift) matches the "
                    f"SGLang-side sigma formula and that ``shift`` is propagated end-to-end."
                )

    @classmethod
    def _build_forward_context_from_results(
        cls,
        *,
        results: Sequence[Any],
        model_type: str,
        batch_size: int,
        height: int,
        width: int,
        guidance_scale: float,
    ) -> Optional[ForwardContext]:
        prompt_embeds_list: List[torch.Tensor] = []
        pooled_list: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []
        negative_prompt_list: List[torch.Tensor] = []
        negative_pooled_list: List[torch.Tensor] = []

        for result in results:
            prompt_embeds = fuse_text_encoder_outputs(result.prompt_embeds)
            require(
                prompt_embeds is not None,
                "SGLang result missing prompt_embeds — request must pin return_prompt_embeds=True",
            )
            prompt_embeds_list.append(prompt_embeds.detach().cpu())

            pooled_prompt_embeds = fuse_text_encoder_outputs(result.pooled_prompt_embeds)
            if pooled_prompt_embeds is not None:
                pooled_list.append(pooled_prompt_embeds.detach().cpu())

            encoder_attention_mask = fuse_text_encoder_outputs(result.encoder_attention_mask)
            if encoder_attention_mask is not None:
                mask_list.append(encoder_attention_mask.detach().cpu())

            negative_prompt_embeds = fuse_text_encoder_outputs(result.negative_prompt_embeds)
            if negative_prompt_embeds is not None:
                negative_prompt_list.append(negative_prompt_embeds.detach().cpu())

            negative_pooled_prompt_embeds = fuse_text_encoder_outputs(result.neg_pooled_prompt_embeds)
            if negative_pooled_prompt_embeds is not None:
                negative_pooled_list.append(negative_pooled_prompt_embeds.detach().cpu())

        prompt_embeds_cat = torch.cat(prompt_embeds_list, dim=0) if prompt_embeds_list else None
        if prompt_embeds_cat is None:
            return None

        raw_tensors: Dict[str, Any] = {
            "prompt_embeds": prompt_embeds_cat,
            "pooled_prompt_embeds": torch.cat(pooled_list, dim=0) if pooled_list else None,
            "encoder_attention_mask": torch.cat(mask_list, dim=0) if mask_list else None,
            "negative_prompt_embeds": torch.cat(negative_prompt_list, dim=0) if negative_prompt_list else None,
            "negative_pooled_prompt_embeds": torch.cat(negative_pooled_list, dim=0) if negative_pooled_list else None,
            "guidance_scale": guidance_scale,
        }

        # ``model_type`` is passed in from ``SGLangRolloutEngine._infer_model_type``,
        # which raises on unknown — so a KeyError here means the registry
        # genuinely lacks a ForwardContext for a model type the engine claimed
        # to support. Let it propagate rather than silently swap to default
        # and lose model-specific fields downstream.
        target_model_type = model_type or "default"
        ctx_cls = get_forward_context_cls(target_model_type)

        valid_fields = {f.name for f in dataclass_fields(ctx_cls)}

        if "text_ids" in valid_fields and prompt_embeds_cat is not None:
            raw_tensors["text_ids"] = build_flux_text_ids(prompt_embeds_cat)
        if "image_ids" in valid_fields and prompt_embeds_cat is not None:
            raw_tensors["image_ids"] = build_flux_image_ids(
                height=height,
                width=width,
                device=prompt_embeds_cat.device,
                dtype=prompt_embeds_cat.dtype,
            )

        filtered = {k: v for k, v in raw_tensors.items() if k in valid_fields and v is not None}
        return ctx_cls(**filtered)
