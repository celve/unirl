"""
GRPO Algorithm Implementation — algorithm-owned loss and training logic.

Standard GRPO with group normalization for advantages. The algorithm file is
the single source of truth for rollout requirements, loss creation, and the
backward training step.
"""

import logging
import math
import os
import time as _time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from diffusionrl.types.sampling import RolloutRequest

import torch
import torch.nn as nn

from diffusionrl.config.build_domain_args import resolve_sde_config
from diffusionrl.samplers.schedulers import create_indices_scheduler
from diffusionrl.types import PromptEmbeddings, SDEConfig, TimestepData
from diffusionrl.utils.misc import aggregate_numeric_metrics

from .base import BaseAlgorithm, EMASpec, SamplingRequirements
from .forward_context import ForwardContext

logger = logging.getLogger(__name__)


def _resolve_algorithm_sde_config(config: Dict[str, Any]) -> SDEConfig:
    return resolve_sde_config(config)


def _save_training_debug_tensor(
    base_dir: str, step_idx: int, name: str, tensor: torch.Tensor,
    rank: int = 0, *, append: bool = False,
) -> None:
    """Save a debug tensor from training path to disk. Only rank 0 saves.

    When *append=True*, concatenates the new tensor with any existing file
    along dim-0 (batch) so that multiple micro-batches accumulate into one
    file covering the full local update batch.
    """
    if rank != 0:
        return
    step_dir = os.path.join(base_dir, f"step_{step_idx:03d}")
    os.makedirs(step_dir, exist_ok=True)
    path = os.path.join(step_dir, f"{name}.pt")
    new_tensor = tensor.detach().cpu().float()
    if append and os.path.exists(path):
        try:
            existing = torch.load(path, map_location="cpu", weights_only=True)
            if existing.ndim >= 1 and new_tensor.ndim >= 1 and existing.shape[1:] == new_tensor.shape[1:]:
                new_tensor = torch.cat([existing, new_tensor], dim=0)
        except Exception:
            pass
    torch.save(new_tensor, path)


class GRPOAlgorithm(BaseAlgorithm):
    """
    Standard GRPO Algorithm — algorithm owns loss and training step.

    This class handles:
    1. Sampling requirements (get_sampling_requirements)
    2. Advantage computation (compute_advantages, inherited)
    3. Loss computation (compute_loss)
    4. Gradient computation (compute_loss_and_backward)

    Features:
    - Group normalization for advantages (within prompt groups)
    - PPO-style clipped objective
    - Optional KL penalty
    - Support for both SDE and Mixed sampling modes

    Reference: DanceGRPO
    """

    @classmethod
    def from_config(cls, config: dict) -> "GRPOAlgorithm":
        """Create GRPOAlgorithm from an algorithm_config dictionary.

        Reads GRPO-specific extension keys from ``algorithm_kwargs`` and shared
        framework-owned fields from the top-level algorithm_config surface.
        Rollout-side and train-side should instantiate from the same
        algorithm_config surface.
        """
        extra = cls.resolve_config_kwargs(config)
        sde_config = _resolve_algorithm_sde_config(config)
        known_keys = {
            "clip_range",
            "clip_schedule",
            "use_kl_penalty",
            "kl_coef",
            "ratio_reg_coef",
            "skip_last_timestep",
            "skip_initial_timesteps",
            "model_type",
        }
        unknown = sorted(key for key in extra.keys() if key not in known_keys)
        if unknown:
            algorithm_label = str(config.get("algorithm_type", "grpo"))
            raise ValueError(
                "algorithm.algorithm_kwargs contains unsupported keys for "
                f"algorithm_type={algorithm_label!r}: "
                f"{unknown}."
            )

        return cls(
            clip_range=float(extra.get("clip_range", 1e-4)),
            clip_schedule=str(extra.get("clip_schedule", "constant")),
            use_kl_penalty=bool(extra.get("use_kl_penalty", True)),
            kl_coef=float(extra.get("kl_coef", 0.01)),
            component_mix_stage=str(config.get("component_mix_stage", "reward")),
            samples_per_prompt=int(config.get("samples_per_prompt", 1)),
            num_inference_steps=int(config.get("num_inference_steps", 0)),
            eval_ema_decay=float(config.get("eval_ema_decay", 0.9)),
            eval_ema_update_interval=int(config.get("eval_ema_update_interval", 1)),
            ratio_reg_coef=float(extra.get("ratio_reg_coef", 0.0)),
            sde_config=sde_config,
            training_share_rollout_indices=bool(
                config.get("training_share_rollout_indices", True)
            ),
            rollout_scheduler_config=dict(config.get("rollout_scheduler") or {}),
            training_scheduler_config=dict(config.get("training_scheduler") or {}),
            skip_last_timestep=bool(extra.get("skip_last_timestep", False)),
            skip_initial_timesteps=int(extra.get("skip_initial_timesteps", 0)),
            model_type=str(extra.get("model_type", "default")),
            adv_normalization=str(config.get("adv_normalization", "group")),
            epsilon=float(config.get("adv_norm_eps", 1e-8)),
            clip_max=config.get("adv_clip_abs", 5.0),
            use_global_std=bool(config.get("use_global_std", False)),
            trimmed_ratio=float(config.get("trimmed_ratio", 0.0)),
        )

    def __init__(
        self,
        clip_range: float = 1e-4,
        clip_schedule: str = "constant",
        use_kl_penalty: bool = True,
        kl_coef: float = 0.01,
        component_mix_stage: str = "reward",
        ratio_reg_coef: float = 0.0,
        sde_config: Optional[SDEConfig] = None,
        training_share_rollout_indices: bool = True,
        rollout_scheduler_config: Optional[Dict[str, Any]] = None,
        training_scheduler_config: Optional[Dict[str, Any]] = None,
        skip_last_timestep: bool = False,
        skip_initial_timesteps: int = 0,
        model_type: str = "default",
        # BaseAlgorithm params
        adv_normalization: str = "group",
        samples_per_prompt: int = 1,
        num_inference_steps: int = 0,
        eval_ema_decay: float = 0.9,
        eval_ema_update_interval: int = 1,
        epsilon: float = 1e-8,
        clip_max: float = 5.0,
        use_global_std: bool = False,
        trimmed_ratio: float = 0.0,
        **kwargs,
    ):
        """
        Initialize GRPO algorithm.

        Args:
            clip_range: PPO clip range (epsilon)
            clip_schedule: Clip range schedule ("constant", "linear_decay", "cosine_decay")
            use_kl_penalty: Whether to add KL penalty
            kl_coef: KL penalty coefficient
            component_mix_stage: Multi-component reward mixing stage
            ratio_reg_coef: Coefficient for ratio regularization
            sde_config: Shared SDE config consumed by rollout and training math
            skip_last_timestep: Skip the last timestep (t->0) in loss computation (MixGRPO).
                The last step has very low noise level, causing unstable log_prob.
            skip_initial_timesteps: Skip the first N timesteps in loss computation (MixGRPO).
                Early timesteps may have high variance.
            model_type: Model type for forward plugin selection ("flux", "sd3", "hunyuan", "default").
                If not specified, model type will be auto-detected from the model class name.
            adv_normalization: Advantage normalization type ("global" or "group")
            samples_per_prompt: Number of rollout samples to generate per prompt
            eval_ema_decay: Eval-time EMA decay
            eval_ema_update_interval: Eval-time EMA update interval
            epsilon: Small value for numerical stability
            clip_max: Maximum advantage clip value (optional)
            use_global_std: Use global std instead of per-group std
            **kwargs: Additional arguments
        """
        super().__init__(
            kl_coef=kl_coef,
            component_mix_stage=component_mix_stage,
            adv_normalization=adv_normalization,
            samples_per_prompt=samples_per_prompt,
            num_inference_steps=num_inference_steps,
            eval_ema_decay=eval_ema_decay,
            eval_ema_update_interval=eval_ema_update_interval,
            epsilon=epsilon,
            clip_max=clip_max,
            use_global_std=use_global_std,
            trimmed_ratio=trimmed_ratio,
            **kwargs,
        )
        self.clip_range = clip_range
        self.clip_schedule = clip_schedule
        self.use_kl_penalty = use_kl_penalty
        self.ratio_reg_coef = ratio_reg_coef
        self.sde_config = sde_config or SDEConfig()
        self.model_type = model_type
        self.training_share_rollout_indices = bool(training_share_rollout_indices)
        self.rollout_scheduler_config = dict(rollout_scheduler_config or {})
        self.training_scheduler_config = dict(
            training_scheduler_config or self.rollout_scheduler_config
        )
        self.rollout_indices_scheduler = create_indices_scheduler(
            scheduler_config=self.rollout_scheduler_config,
            num_timesteps=self.num_inference_steps,
        )
        if self.training_share_rollout_indices:
            self.training_indices_scheduler = self.rollout_indices_scheduler
        else:
            self.training_indices_scheduler = create_indices_scheduler(
                scheduler_config=self.training_scheduler_config,
                num_timesteps=self.num_inference_steps,
            )

        # MixGRPO stability controls
        self.skip_last_timestep = skip_last_timestep
        self.skip_initial_timesteps = skip_initial_timesteps

        # Train-side objective state
        self._debug_output_dir = None  # Set externally for train-inference consistency debugging
        self._debug_dumped_steps: set = set()  # Track which steps already dumped (one-shot guard)

    @property
    def eta(self) -> float:
        return self.sde_config.eta

    @property
    def sde_type(self) -> str:
        return self.sde_config.sde_type

    @property
    def time_shift(self) -> float:
        return self.sde_config.shift

    def get_sampling_requirements(self) -> SamplingRequirements:
        """Return GRPO sampling requirements."""
        return SamplingRequirements(
            requires_trajectory=True,
            requires_log_prob=True,
            requires_embeddings=True,
        )

    def compute_advantages_with_components(
        self,
        *,
        rewards: torch.Tensor,
        group_ids: Optional[List[str]] = None,
        reward_components: Optional[Dict[str, List[float]]] = None,
        reward_component_weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        if self.component_mix_stage != "advantage" or not reward_components:
            return self.compute_advantages(rewards=rewards, group_ids=group_ids)

        default_weights = {name: 1.0 for name in reward_components}
        if reward_component_weights:
            for name, weight in reward_component_weights.items():
                if name in default_weights:
                    default_weights[name] = float(weight)

        weighted_advantages = torch.zeros_like(rewards)
        total_weight = 0.0
        for component_name, component_rewards in reward_components.items():
            component_tensor = torch.tensor(
                component_rewards,
                dtype=rewards.dtype,
                device=rewards.device,
            )
            if component_tensor.shape != rewards.shape:
                logger.warning(
                    "Skipping reward component %s due to shape mismatch: expected=%s got=%s",
                    component_name,
                    tuple(rewards.shape),
                    tuple(component_tensor.shape),
                )
                continue
            component_advantages = self.compute_advantages(
                rewards=component_tensor,
                group_ids=group_ids,
            )
            weight = float(default_weights.get(component_name, 1.0))
            weighted_advantages += component_advantages * weight
            total_weight += weight

        if total_weight <= 0:
            return self.compute_advantages(rewards=rewards, group_ids=group_ids)
        return weighted_advantages / total_weight

    def get_ema_spec(self) -> EMASpec:
        return EMASpec(
            enable_eval_ema=True,
            eval_decay=self.eval_ema_decay,
            eval_update_interval=self.eval_ema_update_interval,
        )

    def resolve_rollout_sde_indices(
        self,
        *,
        current_step: int,
    ) -> Optional[Set[int]]:
        if self.num_inference_steps < 1:
            raise ValueError(
                f"{type(self).__name__}.resolve_rollout_sde_indices requires "
                f"num_inference_steps >= 1, got {self.num_inference_steps}."
            )
        return set(self.rollout_indices_scheduler.get_sde_indices(current_step))

    def get_sampler_validation_config(self, *, args: Any) -> Dict[str, Any]:
        allow_replay = bool(getattr(args.sampling, "replay_log_probs", False))
        return {
            "allow_replay": allow_replay,
            "assert_step_alignment": True,
            "mode_label": "trajectory",
        }

    def get_filtered_training_indices(
        self,
        sde_indices: Set[int],
        num_steps: int,
    ) -> Set[int]:
        result = set(sde_indices)
        if self.skip_last_timestep and result:
            result.discard(max(result))
        if self.skip_initial_timesteps > 0:
            result = {i for i in result if i >= self.skip_initial_timesteps}
        return result

    def resolve_training_timesteps(
        self,
        *,
        batch: Any,
        current_step: int,
        **kwargs: Any,
    ) -> Any:
        from diffusionrl.types.training_batch import BackwardTrainingBatch

        del kwargs
        if not isinstance(batch, BackwardTrainingBatch):
            raise TypeError(
                f"{type(self).__name__} expects BackwardTrainingBatch, got {type(batch).__name__}"
            )

        step_labels = set(int(v) for v in batch.resolved_step_indices[:-1].tolist())
        if not step_labels:
            return tuple()

        requested_steps = set(
            int(i)
            for i in self.training_indices_scheduler.get_sde_indices(current_step)
        )
        filtered_steps = self.get_filtered_training_indices(
            requested_steps,
            len(step_labels),
        )
        missing_steps = sorted(
            int(i) for i in filtered_steps if int(i) not in step_labels
        )
        if missing_steps:
            raise ValueError(
                f"{type(self).__name__}.resolve_training_timesteps selected steps "
                f"not present in batch: missing={missing_steps}, "
                f"available={sorted(step_labels)}"
            )
        if not filtered_steps:
            return tuple()
        return tuple(sorted(int(i) for i in filtered_steps))

    def assemble_training_batch(
        self,
        *,
        request: "RolloutRequest",
        sampler_outputs: List[Any],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        sde_indices: Optional[Set[int]] = None,
    ) -> Any:
        return self._assemble_backward_batch(
            num_inference_steps=request.num_inference_steps,
            sampler_outputs=sampler_outputs,
            rewards=rewards,
            advantages=advantages,
            prompts=request.prompts,
            sde_indices=sde_indices,
        )

    def _assemble_backward_batch(
        self,
        *,
        num_inference_steps: int,
        sampler_outputs: List[Any],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        prompts: List[str],
        sde_indices: Optional[Set[int]] = None,
    ) -> Any:
        from diffusionrl.types.sampling import (
            LogProbData,
            PromptEmbeddings,
            RolloutSamples,
        )
        from diffusionrl.types.training_batch import BackwardTrainingBatch

        trajectories = []
        log_probs_dicts = []
        timesteps = None
        step_indices = None
        explicit_sde_indices = (
            {int(i) for i in sde_indices} if sde_indices is not None else None
        )
        collected_sde_indices: Set[int] = set()
        all_prompt_embeds = []
        all_pooled_prompt_embeds = []
        all_encoder_attention_mask = []
        all_negative_prompt_embeds = []
        all_negative_pooled_prompt_embeds = []
        all_text_ids = []
        all_image_ids = []

        for idx, output in enumerate(sampler_outputs):
            if not isinstance(output, RolloutSamples):
                raise TypeError(
                    f"Assemble stage expects RolloutSamples, got {type(output).__name__} at index={idx}."
                )
            _trajectories = output.aux.get("trajectories")
            _embeddings = output.aux.get("embeddings")
            _log_probs = output.aux.get("log_probs")
            if _trajectories is None:
                raise ValueError(f"RolloutSamples at index={idx} missing trajectories in backward path.")
            if _embeddings is None:
                raise ValueError(f"RolloutSamples at index={idx} missing embeddings in backward path.")

            trajectories.append(_trajectories)
            log_probs_dicts.append(_log_probs.to_dict() if _log_probs is not None else {})
            ts = output.timesteps
            steps = output.aux.get("step_indices")
            sde_idx = output.sde_indices

            if ts is not None and timesteps is None:
                timesteps = ts
            elif ts is not None and timesteps is not None and not torch.equal(timesteps.to(ts.device), ts):
                raise ValueError("Mismatched timesteps across sampler outputs")
            if steps is not None:
                if step_indices is None:
                    step_indices = steps
                elif not torch.equal(step_indices.to(steps.device), steps):
                    raise ValueError(
                        "Mismatched step_indices across sampler outputs: "
                        f"expected={step_indices.tolist()} got={steps.tolist()}"
                    )
            sample_sde_indices = set(int(i) for i in sde_idx)
            if explicit_sde_indices is not None and not explicit_sde_indices.issubset(
                sample_sde_indices
            ):
                missing = sorted(explicit_sde_indices - sample_sde_indices)
                raise ValueError(
                    "assemble_training_batch received explicit sde_indices that are missing "
                    f"from sampler output index={idx}: missing={missing}, "
                    f"available={sorted(sample_sde_indices)}"
                )
            collected_sde_indices.update(sample_sde_indices)

            emb = _embeddings
            all_prompt_embeds.append(emb.prompt_embeds)
            if emb.pooled_prompt_embeds is not None:
                all_pooled_prompt_embeds.append(emb.pooled_prompt_embeds)
            if emb.encoder_attention_mask is not None:
                all_encoder_attention_mask.append(emb.encoder_attention_mask)
            if emb.negative_prompt_embeds is not None:
                all_negative_prompt_embeds.append(emb.negative_prompt_embeds)
            if emb.negative_pooled_prompt_embeds is not None:
                all_negative_pooled_prompt_embeds.append(emb.negative_pooled_prompt_embeds)
            if emb.text_ids is not None:
                all_text_ids.append(emb.text_ids)
            if emb.image_ids is not None:
                all_image_ids.append(emb.image_ids)

        if not trajectories:
            raise ValueError("No trajectories found in sampler outputs")
        if timesteps is None:
            raise ValueError("No timesteps found in sampler outputs")
        if step_indices is None:
            step_indices = torch.arange(timesteps.shape[0], device=timesteps.device, dtype=torch.long)

        step_labels = [int(v) for v in step_indices[:-1].tolist()]
        step_label_set = set(step_labels)

        def _normalize_to_step_labels(indices: Set[int], *, source: str) -> Set[int]:
            if not indices:
                return set()
            if indices.issubset(step_label_set):
                return set(indices)
            mapped = {step_labels[i] for i in indices if 0 <= int(i) < len(step_labels)}
            if mapped:
                logger.debug(
                    "%s indices look positional; mapped to step labels raw=%s mapped=%s",
                    source,
                    sorted(indices),
                    sorted(mapped),
                )
                return mapped
            logger.warning(
                "%s indices do not match sampled step labels and could not be mapped: raw=%s available=%s",
                source,
                sorted(indices),
                sorted(step_labels),
            )
            return set()

        final_sde_indices = (
            explicit_sde_indices
            if explicit_sde_indices is not None
            else collected_sde_indices
        )
        final_sde_indices = _normalize_to_step_labels(
            set(int(i) for i in final_sde_indices), source="Assemble SDE"
        )

        merged_log_probs: Dict[int, torch.Tensor] = {}
        if log_probs_dicts:
            all_indices: Set[int] = set()
            for lpd in log_probs_dicts:
                all_indices.update(lpd.keys())
            for idx_key in all_indices:
                values = [lpd[idx_key] for lpd in log_probs_dicts if idx_key in lpd]
                if values:
                    merged_log_probs[idx_key] = torch.cat(values, dim=0)
        if final_sde_indices:
            merged_log_probs = {
                int(k): v for k, v in merged_log_probs.items()
                if int(k) in set(int(i) for i in final_sde_indices)
            }

        assemble_t0 = _time.perf_counter()
        traj_count = len(trajectories)
        traj_shapes = [tuple(t.shape) for t in trajectories[:3]]
        trajectories_tensor = torch.cat(trajectories, dim=0)
        assemble_t1 = _time.perf_counter()

        prompt_embeds = torch.cat(all_prompt_embeds, dim=0) if all_prompt_embeds else None
        if prompt_embeds is None:
            raise ValueError("No prompt embeddings found in sampler outputs")

        embeddings = PromptEmbeddings(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=torch.cat(all_pooled_prompt_embeds, dim=0) if all_pooled_prompt_embeds else None,
            encoder_attention_mask=torch.cat(all_encoder_attention_mask, dim=0) if all_encoder_attention_mask else None,
            negative_prompt_embeds=torch.cat(all_negative_prompt_embeds, dim=0) if all_negative_prompt_embeds else None,
            negative_pooled_prompt_embeds=torch.cat(all_negative_pooled_prompt_embeds, dim=0) if all_negative_pooled_prompt_embeds else None,
            text_ids=torch.cat(all_text_ids, dim=0) if all_text_ids else None,
            image_ids=all_image_ids[0] if all_image_ids else None,
        )

        batch = BackwardTrainingBatch(
            trajectories=trajectories_tensor,
            log_probs=LogProbData.from_dict(merged_log_probs),
            timesteps=timesteps,
            advantages=advantages,
            embeddings=embeddings,
            rewards=rewards,
            prompts=prompts,
            step_indices=step_indices,
            target_sde_indices=set(int(i) for i in final_sde_indices),
        )

        batch.validate()
        assemble_t2 = _time.perf_counter()
        traj_gb = trajectories_tensor.nelement() * trajectories_tensor.element_size() / 1e9
        logger.debug(
            "[TIMING] assemble_backward: cat_traj=%.2fs total=%.2fs n=%d shapes=%s traj_gb=%.2f",
            assemble_t1 - assemble_t0,
            assemble_t2 - assemble_t0,
            traj_count,
            traj_shapes,
            traj_gb,
        )
        return batch

    # ==================================================================
    # Objective computation
    # ==================================================================

    def get_clip_range(self, progress: float = 0.0) -> float:
        """
        Get the current clip range based on training progress.

        Args:
            progress: Training progress in [0, 1]

        Returns:
            Current clip range
        """
        if self.clip_schedule == "constant":
            return self.clip_range
        elif self.clip_schedule == "linear_decay":
            return self.clip_range * (1 - 0.5 * progress)
        elif self.clip_schedule == "cosine_decay":
            return self.clip_range * (0.5 * (1 + math.cos(math.pi * progress)))
        else:
            return self.clip_range

    def compute_log_prob(
        self,
        pred: torch.Tensor,
        sample: torch.Tensor,
        next_sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        sigma_max: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probability for SDE step.

        Delegates to the canonical implementation in diffusionrl.sde.runtime
        to ensure consistency between sampling and training paths.

        Args:
            pred: Model velocity prediction [B, C, H, W] or [B, C, T, H, W]
            sample: Current sample (x_t)
            next_sample: Next sample (x_{t-1})
            sigma: Current sigma
            sigma_next: Next sigma
            sigma_max: Maximum sigma value

        Returns:
            Tuple of (log_prob [B], prev_sample_mean [B, C, ...])
        """
        from diffusionrl.sde.runtime import denoising_step

        _, new_log_prob, prev_sample_mean = denoising_step(
            noise_pred=pred,
            sample=sample,
            sigma=sigma,
            sigma_next=sigma_next,
            eta=self.eta,
            prev_sample=next_sample,
            sde_type=self.sde_type,
            sigma_max=sigma_max,
        )
        return new_log_prob, prev_sample_mean

    def compute_loss(
        self,
        model: nn.Module,
        timestep_data: TimestepData,
        advantages: torch.Tensor,
        *,
        ctx: ForwardContext,
        sigmas: Optional[torch.Tensor] = None,
        ref_model: Optional[nn.Module] = None,
        training_progress: float = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Single GRPO loss entrypoint for debugging and training."""
        device = advantages.device

        latents = timestep_data.latents
        next_latents = timestep_data.next_latents
        old_log_probs = timestep_data.log_prob
        sigma = timestep_data.sigma
        sigma_next = timestep_data.sigma_next
        timestep_idx = timestep_data.timestep_idx

        if old_log_probs is None:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                "skip_reason": "ode_step",
                "timestep_idx": timestep_idx,
            }

        _sigmas = sigmas if sigmas is not None else timestep_data.sigmas
        if _sigmas is not None:
            sigma_max = _sigmas[1].item() if _sigmas[1].dim() == 0 else _sigmas[1][0].item()
        else:
            raise ValueError(
                "Cannot determine sigma_max: neither `sigmas` argument nor "
                "`timestep_data.sigmas` is provided. Ensure TimestepData is "
                "constructed with the full sigma schedule."
            )

        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma, device=device)
        if not isinstance(sigma_next, torch.Tensor):
            sigma_next = torch.tensor(sigma_next, device=device)

        pred = ctx.forward(model, latents=latents, sigma=sigma)

        new_log_prob, prev_sample_mean = self.compute_log_prob(
            pred=pred,
            sample=latents,
            next_sample=next_latents,
            sigma=sigma,
            sigma_next=sigma_next,
            sigma_max=sigma_max,
        )

        log_prob_diff = new_log_prob - old_log_probs
        ratio = torch.exp(log_prob_diff)

        _resolved_debug_dir = self._debug_output_dir or os.environ.get("DIFFUSIONRL_DEBUG_OUTPUT_DIR")
        if _resolved_debug_dir is not None:
            _append = timestep_idx in self._debug_dumped_steps
            self._debug_dumped_steps.add(timestep_idx)
            _rank = int(os.environ.get("RANK", 0))
            if _rank == 0:
                logger.info(
                    "Debug training: rank=%d timestep_idx=%d batch_size=%d latents=%s append=%s",
                    _rank, timestep_idx, latents.shape[0], list(latents.shape), _append,
                )
            _training_debug_dir = os.path.join(_resolved_debug_dir, "training")
            _save_training_debug_tensor(_training_debug_dir, timestep_idx, "noise_pred", pred, _rank, append=_append)
            _save_training_debug_tensor(_training_debug_dir, timestep_idx, "latents_input", latents, _rank, append=_append)
            _save_training_debug_tensor(_training_debug_dir, timestep_idx, "latents_output", next_latents, _rank, append=_append)
            _save_training_debug_tensor(_training_debug_dir, timestep_idx, "prev_sample_mean", prev_sample_mean, _rank, append=_append)
            _save_training_debug_tensor(_training_debug_dir, timestep_idx, "new_log_prob", new_log_prob, _rank, append=_append)
            _save_training_debug_tensor(_training_debug_dir, timestep_idx, "old_log_prob", old_log_probs, _rank, append=_append)
            _save_training_debug_tensor(_training_debug_dir, timestep_idx, "ratio", ratio, _rank, append=_append)
            _save_training_debug_tensor(_training_debug_dir, timestep_idx, "sigma", sigma.unsqueeze(0) if sigma.dim() == 0 else sigma, _rank)
            _save_training_debug_tensor(_training_debug_dir, timestep_idx, "sigma_next", sigma_next.unsqueeze(0) if sigma_next.dim() == 0 else sigma_next, _rank)
            _save_training_debug_tensor(_training_debug_dir, timestep_idx, "sigma_max", torch.tensor([sigma_max]), _rank)
            if not _append and _sigmas is not None and _rank == 0:
                _step_dir = os.path.join(_training_debug_dir, f"step_{timestep_idx:03d}")
                _sched_path = os.path.join(_step_dir, "sigmas_schedule.pt")
                if not os.path.exists(_sched_path):
                    torch.save(_sigmas.detach().cpu().float(), _sched_path)

        clip_range = self.get_clip_range(training_progress)

        adv = advantages.detach()
        unclipped_loss = -adv * ratio
        clipped_loss = -adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

        clip_fraction = ((ratio - 1.0).abs() > clip_range).float().mean()
        clipfrac_gt_one = (ratio - 1.0 > clip_range).float().mean()
        clipfrac_lt_one = (1.0 - ratio > clip_range).float().mean()
        approx_kl = 0.5 * torch.mean(log_prob_diff ** 2)

        loss_terms = {
            "policy_loss": policy_loss.detach(),
            "ratio_mean": ratio.mean().detach(),
            "ratio_std": ratio.std().detach(),
            "ratio_max": ratio.max().detach(),
            "ratio_min": ratio.min().detach(),
            "clip_fraction": clip_fraction.detach(),
            "clipfrac_gt_one": clipfrac_gt_one.detach(),
            "clipfrac_lt_one": clipfrac_lt_one.detach(),
            "approx_kl": approx_kl.detach(),
            "new_log_prob_mean": new_log_prob.mean().detach(),
            "old_log_prob_mean": old_log_probs.mean().detach(),
        }

        total_loss = policy_loss

        if self.use_kl_penalty and self.kl_coef > 0:
            kl_loss = self._compute_kl_penalty(
                model=model,
                latents=latents,
                next_latents=next_latents,
                sigma=sigma,
                sigma_next=sigma_next,
                sigma_max=sigma_max,
                prev_sample_mean=prev_sample_mean,
                ref_model=ref_model,
                ctx=ctx,
                **kwargs,
            )
            if kl_loss is not None:
                total_loss = total_loss + self.kl_coef * kl_loss
                loss_terms["kl_loss"] = kl_loss.detach()

        if self.ratio_reg_coef > 0:
            ratio_reg = torch.mean((new_log_prob - old_log_probs) ** 2)
            total_loss = total_loss + self.ratio_reg_coef * ratio_reg
            loss_terms["ratio_reg"] = ratio_reg.detach()

        loss_terms["total_loss"] = total_loss.detach()
        loss_terms["timestep_idx"] = timestep_idx

        return total_loss, loss_terms

    # ------------------------------------------------------------------
    # Algorithm-owned training step (Phase 2)
    # ------------------------------------------------------------------

    def compute_loss_and_backward(
        self,
        *,
        model: nn.Module,
        batch: Any,
        guidance_scale: float = 3.5,
        timesteps: Optional[Any] = None,
        loss_scale: float = 1.0,
        **kwargs: Any,
    ) -> tuple:
        """GRPO training step over one micro-batch.

        Here ``timesteps`` means the logical reverse-process step labels to
        optimize for this update chunk, for example ``(0, 1, 2, ...)`` after
        any algorithm-specific filtering. These are not continuous forward
        diffusion times; the algorithm maps each label back to the matching
        trajectory position via ``batch.get_timestep_data_by_step(...)``.

        Returns:
            ``(scaled_loss, metrics_dict, num_timesteps, has_backward)``
        """
        from diffusionrl.types.training_batch import BackwardTrainingBatch

        if not isinstance(batch, BackwardTrainingBatch):
            raise TypeError(
                f"{type(self).__name__} expects BackwardTrainingBatch, got {type(batch).__name__}"
            )

        model.train()

        available_steps = set(int(s) for s in batch.resolved_step_indices[:-1].tolist())
        if timesteps is None:
            timesteps = self.resolve_training_timesteps(batch=batch, current_step=0)
        if torch.is_tensor(timesteps):
            candidate_steps = [int(i) for i in timesteps.detach().flatten().tolist()]
        else:
            candidate_steps = [int(i) for i in timesteps]
        valid_step_indices = [
            step for step in candidate_steps if step in available_steps
        ]
        num_timesteps_per_sample = len(valid_step_indices)
        if num_timesteps_per_sample == 0:
            return 0.0, {}, 0, False

        ctx = self._make_forward_context(model, batch.embeddings, guidance_scale)

        total_loss_accum = 0.0
        has_backward = False
        all_metrics: Dict[str, Any] = {}
        timestep_metrics: List[Dict[str, Any]] = []

        for t_idx in valid_step_indices:
            timestep_data = batch.get_timestep_data_by_step(t_idx)
            loss_t, metrics_t = self.compute_loss(
                model=model,
                timestep_data=timestep_data,
                advantages=batch.advantages,
                ctx=ctx,
            )
            scaled_loss = loss_t * (loss_scale / num_timesteps_per_sample)
            scaled_loss.backward()
            has_backward = True
            total_loss_accum += scaled_loss.detach().item()
            timestep_metrics.append(metrics_t)

            for key, value in metrics_t.items():
                val = value.item() if isinstance(value, torch.Tensor) else value
                metric_key = f"t{t_idx}_{key}"
                if metric_key not in all_metrics:
                    all_metrics[metric_key] = val

        all_metrics.update(aggregate_numeric_metrics(timestep_metrics))

        return total_loss_accum, all_metrics, num_timesteps_per_sample, has_backward

    # ------------------------------------------------------------------
    # Forward plugin
    # ------------------------------------------------------------------

    def _compute_kl_penalty(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        next_latents: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        sigma_max: float,
        prev_sample_mean: torch.Tensor,
        ref_model: Optional[nn.Module],
        ctx: ForwardContext,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Compute KL penalty between policy and reference."""
        ref_prev_sample_mean = None

        # Try LoRA disable_adapter first (flow_grpo style)
        adapter_model = model.module if hasattr(model, 'module') else model
        if hasattr(adapter_model, 'disable_adapter'):
            try:
                with torch.no_grad():
                    with adapter_model.disable_adapter():
                        ref_pred = ctx.forward(model, latents=latents, sigma=sigma)

                _, ref_prev_sample_mean = self.compute_log_prob(
                    pred=ref_pred,
                    sample=latents,
                    next_sample=next_latents,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    sigma_max=sigma_max,
                )
            except Exception:
                pass

        # Fallback to ref_model if provided
        if ref_prev_sample_mean is None and ref_model is not None:
            with torch.no_grad():
                ref_pred = ctx.forward(ref_model, latents=latents, sigma=sigma)

            _, ref_prev_sample_mean = self.compute_log_prob(
                pred=ref_pred,
                sample=latents,
                next_sample=next_latents,
                sigma=sigma,
                sigma_next=sigma_next,
                sigma_max=sigma_max,
            )

        if ref_prev_sample_mean is not None:
            # KL loss calculation matching flow_grpo
            std_dev_t = torch.sqrt(
                sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))
            ) * self.eta

            kl_loss = ((prev_sample_mean - ref_prev_sample_mean) ** 2).mean(
                dim=tuple(range(1, prev_sample_mean.ndim))
            ) / (2 * std_dev_t ** 2 + 1e-12)
            kl_loss = kl_loss.mean()

            return kl_loss

        return None
