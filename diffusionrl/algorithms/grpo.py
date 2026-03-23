"""
GRPO Algorithm Implementation — algorithm-owned loss and training logic.

Standard GRPO with group normalization for advantages. The algorithm file is
the single source of truth for rollout requirements, loss creation, and the
backward training step.
"""

import math
import logging
import os
import time as _time
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from diffusionrl.types import PromptEmbeddings, SDEConfig, TimestepData
from .base import BaseAlgorithm, EMASpec, SamplingRequirements

logger = logging.getLogger(__name__)


def _resolve_algorithm_sde_config(config: Dict[str, Any]) -> SDEConfig:
    return SDEConfig.from_mapping(config.get("sde_config"))


def _save_training_debug_tensor(base_dir: str, step_idx: int, name: str, tensor: torch.Tensor, rank: int = 0) -> None:
    """Save a debug tensor from training path to disk. Only rank 0 saves."""
    if rank != 0:
        return
    step_dir = os.path.join(base_dir, f"step_{step_idx:03d}")
    os.makedirs(step_dir, exist_ok=True)
    path = os.path.join(step_dir, f"{name}.pt")
    torch.save(tensor.detach().cpu().float(), path)


class _GRPOLoss:
    """Private GRPO objective owned and created by GRPOAlgorithm."""

    def __init__(self, algorithm: "GRPOAlgorithm") -> None:
        self.algorithm = algorithm

    def compute_loss(
        self,
        model: nn.Module,
        timestep_data: TimestepData,
        advantages: torch.Tensor,
        embeddings: PromptEmbeddings,
        sigmas: Optional[torch.Tensor] = None,
        ref_model: Optional[nn.Module] = None,
        training_progress: float = 0.0,
        model_forward_fn: Optional[callable] = None,
        guidance_scale: float = 3.5,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        algorithm = self.algorithm
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

        prompt_embeds = embeddings.prompt_embeds
        pooled_prompt_embeds = embeddings.pooled_prompt_embeds
        negative_prompt_embeds = getattr(embeddings, "negative_prompt_embeds", None)
        negative_pooled_prompt_embeds = getattr(embeddings, "negative_pooled_prompt_embeds", None)

        if model_forward_fn is not None:
            pred = model_forward_fn(
                model=model,
                latents=latents,
                sigma=sigma,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                guidance_scale=guidance_scale,
                text_ids=embeddings.text_ids,
                image_ids=embeddings.image_ids,
                **kwargs,
            )
        else:
            pred = algorithm._default_forward(
                model=model,
                latents=latents,
                sigma=sigma,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                guidance_scale=guidance_scale,
                text_ids=embeddings.text_ids,
                image_ids=embeddings.image_ids,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            )

        pred_f = pred.float()
        latents_f = latents.float()
        next_latents_f = next_latents.float()

        new_log_prob, prev_sample_mean = algorithm.compute_log_prob(
            pred=pred_f,
            sample=latents_f,
            next_sample=next_latents_f,
            sigma=sigma,
            sigma_next=sigma_next,
            sigma_max=sigma_max,
        )

        log_prob_diff = new_log_prob - old_log_probs
        ratio = torch.exp(log_prob_diff)

        _resolved_debug_dir = algorithm._debug_output_dir or os.environ.get("DIFFUSIONRL_DEBUG_OUTPUT_DIR")
        if _resolved_debug_dir is not None and timestep_idx not in algorithm._debug_dumped_steps:
            algorithm._debug_dumped_steps.add(timestep_idx)
            _rank = int(os.environ.get("RANK", 0))
            _training_debug_dir = os.path.join(_resolved_debug_dir, "training")
            _save_training_debug_tensor(_training_debug_dir, timestep_idx, "noise_pred", pred, _rank)
            _save_training_debug_tensor(_training_debug_dir, timestep_idx, "latents_input", latents, _rank)
            _save_training_debug_tensor(_training_debug_dir, timestep_idx, "latents_output", next_latents, _rank)
            _save_training_debug_tensor(_training_debug_dir, timestep_idx, "prev_sample_mean", prev_sample_mean, _rank)
            _save_training_debug_tensor(_training_debug_dir, timestep_idx, "new_log_prob", new_log_prob, _rank)
            _save_training_debug_tensor(_training_debug_dir, timestep_idx, "old_log_prob", old_log_probs, _rank)
            _save_training_debug_tensor(_training_debug_dir, timestep_idx, "ratio", ratio, _rank)
            _save_training_debug_tensor(_training_debug_dir, timestep_idx, "sigma", sigma.unsqueeze(0) if sigma.dim() == 0 else sigma, _rank)
            _save_training_debug_tensor(_training_debug_dir, timestep_idx, "sigma_next", sigma_next.unsqueeze(0) if sigma_next.dim() == 0 else sigma_next, _rank)
            _save_training_debug_tensor(_training_debug_dir, timestep_idx, "sigma_max", torch.tensor([sigma_max]), _rank)
            if _sigmas is not None and _rank == 0:
                _step_dir = os.path.join(_training_debug_dir, f"step_{timestep_idx:03d}")
                _sched_path = os.path.join(_step_dir, "sigmas_schedule.pt")
                if not os.path.exists(_sched_path):
                    torch.save(_sigmas.detach().cpu().float(), _sched_path)

        clip_range = algorithm.get_clip_range(training_progress)

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

        if algorithm.use_kl_penalty and algorithm.kl_coef > 0:
            kl_loss = algorithm._compute_kl_penalty(
                model=model,
                latents=latents,
                latents_f=latents_f,
                next_latents_f=next_latents_f,
                sigma=sigma,
                sigma_next=sigma_next,
                sigma_max=sigma_max,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                prev_sample_mean=prev_sample_mean,
                ref_model=ref_model,
                guidance_scale=guidance_scale,
                model_forward_fn=model_forward_fn,
                **kwargs,
            )
            if kl_loss is not None:
                total_loss = total_loss + algorithm.kl_coef * kl_loss
                loss_terms["kl_loss"] = kl_loss.detach()

        if algorithm.ratio_reg_coef > 0:
            ratio_reg = torch.mean((new_log_prob - old_log_probs) ** 2)
            total_loss = total_loss + algorithm.ratio_reg_coef * ratio_reg
            loss_terms["ratio_reg"] = ratio_reg.detach()

        loss_terms["total_loss"] = total_loss.detach()
        loss_terms["timestep_idx"] = timestep_idx

        return total_loss, loss_terms


class GRPOAlgorithm(BaseAlgorithm):
    """
    Standard GRPO Algorithm — algorithm owns loss and training step.

    This class handles:
    1. Sampling requirements (get_sampling_requirements)
    2. Advantage computation (compute_advantages, inherited)
    3. Loss ownership / debug entry (compute_loss -> self.loss_fn)
    4. Gradient computation (compute_loss_and_backward)

    Features:
    - Group normalization for advantages (within prompt groups)
    - PPO-style clipped objective
    - Optional KL penalty
    - Support for both SDE and Mixed sampling modes

    Reference: DanceGRPO
    """
    _loss_cls = _GRPOLoss

    @classmethod
    def from_config(cls, config: dict) -> "GRPOAlgorithm":
        """Create GRPOAlgorithm from an algorithm_config dictionary.

        Reads constructor parameters from the normalized ``algorithm_kwargs``
        payload only. Rollout-side and train-side should instantiate from the
        same algorithm_config surface.
        """
        extra = dict(config.get("algorithm_kwargs") or {})
        sde_config = _resolve_algorithm_sde_config(config)
        known_keys = {
            "clip_range",
            "clip_schedule",
            "use_kl_penalty",
            "kl_coef",
            "component_mix_stage",
            "samples_per_prompt",
            "eval_ema_decay",
            "eval_ema_update_interval",
            "ratio_reg_coef",
            "skip_last_timestep",
            "skip_initial_timesteps",
            "window_training",
            "model_type",
            "adv_normalization",
            "adv_norm_eps",
            "adv_clip_abs",
            "use_global_std",
            "trimmed_ratio",
        }
        runtime_only_keys = {
            "shuffle_samples",
            "shuffle_seed",
        }
        unknown = sorted(key for key in extra.keys() if key not in known_keys and key not in runtime_only_keys)
        if unknown:
            warnings.warn(
                f"GRPOAlgorithm.from_config received unknown algorithm_kwargs keys: {unknown}. "
                "These keys are ignored by GRPO algorithm constructor.",
                stacklevel=3,
            )

        return cls(
            clip_range=float(extra.get("clip_range", 1e-4)),
            clip_schedule=str(extra.get("clip_schedule", "constant")),
            use_kl_penalty=bool(extra.get("use_kl_penalty", True)),
            kl_coef=float(extra.get("kl_coef", 0.01)),
            component_mix_stage=str(extra.get("component_mix_stage", "reward")),
            samples_per_prompt=int(extra.get("samples_per_prompt", 1)),
            eval_ema_decay=float(extra.get("eval_ema_decay", 0.9)),
            eval_ema_update_interval=int(extra.get("eval_ema_update_interval", 1)),
            ratio_reg_coef=float(extra.get("ratio_reg_coef", 0.0)),
            sde_config=sde_config,
            skip_last_timestep=bool(extra.get("skip_last_timestep", False)),
            skip_initial_timesteps=int(extra.get("skip_initial_timesteps", 0)),
            model_type=str(extra.get("model_type", "default")),
            adv_normalization=str(extra.get("adv_normalization", "group")),
            epsilon=float(extra.get("adv_norm_eps", 1e-8)),
            clip_max=extra.get("adv_clip_abs", 5.0),
            use_global_std=bool(extra.get("use_global_std", False)),
            trimmed_ratio=float(extra.get("trimmed_ratio", 0.0)),
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
        skip_last_timestep: bool = False,
        skip_initial_timesteps: int = 0,
        model_type: str = "default",
        # BaseAlgorithm params
        adv_normalization: str = "group",
        samples_per_prompt: int = 1,
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

        # MixGRPO stability controls
        self.skip_last_timestep = skip_last_timestep
        self.skip_initial_timesteps = skip_initial_timesteps

        # Train-side objective state
        self._forward_plugin = None  # Lazy loaded
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
        return EMASpec(enable_eval_ema=False)

    def resolve_rollout_sde_indices(
        self,
        *,
        timestep_scheduler: Optional[Any],
        current_step: int,
    ) -> Optional[Set[int]]:
        if timestep_scheduler is None:
            return None
        return set(int(i) for i in timestep_scheduler.get_sde_indices(current_step))

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

    def assemble_training_batch(
        self,
        *,
        num_inference_steps: int,
        sampler_outputs: List[Any],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        prompts: List[str],
        sde_indices: Optional[Set[int]] = None,
    ) -> Any:
        return self._assemble_backward_batch(
            num_inference_steps=num_inference_steps,
            sampler_outputs=sampler_outputs,
            rewards=rewards,
            advantages=advantages,
            prompts=prompts,
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
        from diffusionrl.types.sampling import RolloutOutput, LogProbData, PromptEmbeddings
        from diffusionrl.types.training_batch import BackwardTrainingBatch

        trajectories = []
        log_probs_dicts = []
        timesteps = None
        step_indices = None
        scheduler_was_provided = sde_indices is not None
        raw_scheduler_indices = {int(i) for i in sde_indices} if sde_indices is not None else None
        final_sde_indices: Set[int] = set(raw_scheduler_indices or set())
        all_prompt_embeds = []
        all_pooled_prompt_embeds = []
        all_encoder_attention_mask = []
        all_negative_prompt_embeds = []
        all_negative_pooled_prompt_embeds = []
        all_text_ids = []
        all_image_ids = []

        for idx, output in enumerate(sampler_outputs):
            if not isinstance(output, RolloutOutput):
                raise TypeError(
                    f"Assemble stage expects RolloutOutput, got {type(output).__name__} at index={idx}."
                )
            if output.trajectories is None:
                raise ValueError(f"RolloutOutput at index={idx} missing trajectories in backward path.")
            if output.embeddings is None:
                raise ValueError(f"RolloutOutput at index={idx} missing embeddings in backward path.")

            trajectories.append(output.trajectories)
            log_probs_dicts.append(output.log_probs.to_dict() if output.log_probs is not None else {})
            ts = output.timesteps
            steps = output.step_indices
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
            if sde_indices is None:
                final_sde_indices.update(int(i) for i in sde_idx)

            emb = output.embeddings
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

        if final_sde_indices:
            final_sde_indices = _normalize_to_step_labels(
                set(int(i) for i in final_sde_indices), source="Scheduler/Sampler SDE"
            )

        raw_train_indices = self.resolve_training_indices(
            num_steps=len(step_labels),
            sde_indices=set(final_sde_indices) if scheduler_was_provided else None,
        )
        train_indices = _normalize_to_step_labels(
            set(int(i) for i in raw_train_indices),
            source=f"{type(self).__name__}.resolve_training_indices",
        )
        if not train_indices:
            train_indices = step_label_set
        if not scheduler_was_provided:
            final_sde_indices = train_indices
        else:
            final_sde_indices = final_sde_indices & train_indices if final_sde_indices else train_indices
        if not final_sde_indices:
            final_sde_indices = train_indices if train_indices else set(step_labels)

        num_steps = len(step_labels)
        final_sde_indices = self.get_filtered_training_indices(final_sde_indices, num_steps)
        if not final_sde_indices:
            final_sde_indices = set(step_labels)

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
        from diffusionrl.sde.runtime import compute_sde_log_prob

        return compute_sde_log_prob(
            noise_pred=pred,
            sample=sample,
            prev_sample=next_sample,
            sigma=sigma,
            sigma_next=sigma_next,
            eta=self.eta,
            sde_type=self.sde_type,
            sigma_max=sigma_max,
        )

    def compute_loss(
        self,
        model: nn.Module,
        timestep_data: TimestepData,
        advantages: torch.Tensor,
        embeddings: PromptEmbeddings,
        sigmas: Optional[torch.Tensor] = None,
        ref_model: Optional[nn.Module] = None,
        training_progress: float = 0.0,
        model_forward_fn: Optional[callable] = None,
        guidance_scale: float = 3.5,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Single GRPO loss entrypoint for debugging and training."""
        if self.loss_fn is None:
            raise RuntimeError(f"{type(self).__name__} loss_fn is not initialized.")
        return self.loss_fn.compute_loss(
            model=model,
            timestep_data=timestep_data,
            advantages=advantages,
            embeddings=embeddings,
            sigmas=sigmas,
            ref_model=ref_model,
            training_progress=training_progress,
            model_forward_fn=model_forward_fn,
            guidance_scale=guidance_scale,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Algorithm-owned training step (Phase 2)
    # ------------------------------------------------------------------

    def compute_loss_and_backward(
        self,
        *,
        model: nn.Module,
        batch: Any,
        mini_batch_slices: Tuple[Tuple[int, int], ...],
        guidance_scale: float = 3.5,
        **kwargs: Any,
    ) -> tuple:
        """GRPO training step: iterate over SDE timesteps with gradient accumulation.

        The algorithm itself decides which timesteps to train on and how to accumulate.

        Returns:
            ``(avg_loss, metrics_dict, num_timesteps, actual_mini_batches, has_backward)``
        """
        from diffusionrl.types.training_batch import BackwardTrainingBatch

        if not isinstance(batch, BackwardTrainingBatch):
            raise TypeError(
                f"{type(self).__name__} expects BackwardTrainingBatch, got {type(batch).__name__}"
            )

        mini_batches = tuple((int(start), int(end)) for start, end in mini_batch_slices)
        if not mini_batches:
            raise ValueError(f"{type(self).__name__} requires non-empty mini_batch_slices.")
        actual_mini_batches = len(mini_batches)
        num_mini_batches = len(mini_batches)

        available_steps = set(int(s) for s in batch.resolved_step_indices[:-1].tolist())
        valid_step_indices = sorted(int(i) for i in batch.sde_indices if int(i) in available_steps)
        num_timesteps_per_sample = len(valid_step_indices)
        if num_timesteps_per_sample == 0:
            return 0.0, {}, 0, actual_mini_batches, False

        total_loss_accum = 0.0
        has_backward = False
        mini_batch_metrics_list = []

        for start, end in mini_batches:
            mini_batch = batch.slice(start, end)
            mini_loss_sum = 0.0
            mini_metrics: Dict[str, Any] = {}
            metric_sums: Dict[str, float] = {}
            metric_counts: Dict[str, int] = {}

            for t_idx in valid_step_indices:
                timestep_data = mini_batch.get_timestep_data_by_step(t_idx)
                loss_t, metrics_t = self.compute_loss(
                    model=model,
                    timestep_data=timestep_data,
                    advantages=mini_batch.advantages,
                    embeddings=mini_batch.embeddings,
                    guidance_scale=guidance_scale,
                )
                scaled_loss = loss_t / (num_mini_batches * num_timesteps_per_sample)
                scaled_loss.backward()
                has_backward = True
                mini_loss_sum += scaled_loss.detach().item()

                for key, value in metrics_t.items():
                    val = value.item() if isinstance(value, torch.Tensor) else value
                    metric_key = f"t{t_idx}_{key}"
                    if metric_key not in mini_metrics:
                        mini_metrics[metric_key] = val
                    if isinstance(val, (int, float)):
                        metric_sums[key] = metric_sums.get(key, 0.0) + float(val)
                        metric_counts[key] = metric_counts.get(key, 0) + 1

            for key, total in metric_sums.items():
                count = metric_counts.get(key, 0)
                if count > 0:
                    mini_metrics[key] = total / count

            mini_batch_metrics_list.append(mini_metrics)
            total_loss_accum += mini_loss_sum

        all_metrics: Dict[str, Any] = {}
        if mini_batch_metrics_list:
            keys = mini_batch_metrics_list[0].keys()
            for key in keys:
                values = [m.get(key) for m in mini_batch_metrics_list if m.get(key) is not None]
                if values and isinstance(values[0], (int, float)):
                    all_metrics[key] = sum(values) / len(values)
                else:
                    all_metrics[key] = mini_batch_metrics_list[-1].get(key)

        return total_loss_accum, all_metrics, num_timesteps_per_sample, actual_mini_batches, has_backward

    # ------------------------------------------------------------------
    # Forward plugin
    # ------------------------------------------------------------------

    def _get_forward_plugin(self, model: nn.Module):
        """Get the forward plugin for this model.

        The plugin should be set by the training actor from model_bundle.forward_plugin().
        Falls back to DefaultForwardPlugin if not set.
        """
        if self._forward_plugin is None:
            if self.model_type != "default":
                raise RuntimeError(
                    "No forward_plugin set on algorithm for model_type="
                    f"{self.model_type!r}. TrainingActor must inject plugin via "
                    "model_bundle.forward_plugin()."
                )
            from diffusionrl.models.forward_plugins import DefaultForwardPlugin
            self._forward_plugin = DefaultForwardPlugin()
            logger.warning(
                "No forward_plugin set on algorithm (model_type=%s). "
                "Using DefaultForwardPlugin. Set algorithm._forward_plugin from "
                "model_bundle.forward_plugin() for model-specific behavior.",
                self.model_type,
            )
        return self._forward_plugin

    def _default_forward(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor],
        guidance_scale: float,
        text_ids: Optional[torch.Tensor] = None,
        image_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Default forward pass using model-specific plugin.

        Delegates to the appropriate forward plugin based on model_type.
        """
        plugin = self._get_forward_plugin(model)

        return plugin.forward(
            model=model,
            latents=latents,
            sigma=sigma,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            guidance_scale=guidance_scale,
            text_ids=text_ids,
            image_ids=image_ids,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            **kwargs,
        )

    def _compute_kl_penalty(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        latents_f: torch.Tensor,
        next_latents_f: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        sigma_max: float,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor],
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_pooled_prompt_embeds: Optional[torch.Tensor],
        prev_sample_mean: torch.Tensor,
        ref_model: Optional[nn.Module],
        guidance_scale: float,
        model_forward_fn: Optional[callable],
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
                        if model_forward_fn is not None:
                            ref_pred = model_forward_fn(
                                model=model,
                                latents=latents,
                                sigma=sigma,
                                prompt_embeds=prompt_embeds,
                                pooled_prompt_embeds=pooled_prompt_embeds,
                                guidance_scale=guidance_scale,
                                **kwargs,
                            )
                        else:
                            ref_pred = self._default_forward(
                                model=model,
                                latents=latents,
                                sigma=sigma,
                                prompt_embeds=prompt_embeds,
                                pooled_prompt_embeds=pooled_prompt_embeds,
                                guidance_scale=guidance_scale,
                                negative_prompt_embeds=negative_prompt_embeds,
                                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                                **kwargs,
                            )

                _, ref_prev_sample_mean = self.compute_log_prob(
                    pred=ref_pred.float(),
                    sample=latents_f,
                    next_sample=next_latents_f,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    sigma_max=sigma_max,
                )
            except Exception:
                pass

        # Fallback to ref_model if provided
        if ref_prev_sample_mean is None and ref_model is not None:
            with torch.no_grad():
                if model_forward_fn is not None:
                    ref_pred = model_forward_fn(
                        model=ref_model,
                        latents=latents,
                        sigma=sigma,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        guidance_scale=guidance_scale,
                        **kwargs,
                    )
                else:
                    ref_pred = self._default_forward(
                        model=ref_model,
                        latents=latents,
                        sigma=sigma,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        guidance_scale=guidance_scale,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                        **kwargs,
                    )

            _, ref_prev_sample_mean = self.compute_log_prob(
                pred=ref_pred.float(),
                sample=latents_f,
                next_sample=next_latents_f,
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
