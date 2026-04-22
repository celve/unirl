"""
GRPO Algorithm Implementation — algorithm-owned loss and training logic.

Standard GRPO with group normalization for advantages. The algorithm file is
the single source of truth for rollout requirements, loss creation, and the
backward training step.
"""

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from diffusionrl.algorithms.registry import register_algorithm
from diffusionrl.models.base import ModelBundle
from diffusionrl.types import TimestepData
from diffusionrl.types.forward_context import ForwardContext
from diffusionrl.utils.misc import aggregate_numeric_metrics
from diffusionrl.utils.scheduler_utils import create_indices_scheduler

from .base import BaseAlgorithm, BaseAlgorithmConfig, EMASpec, SamplingRequirements

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GRPOAlgorithmConfig(BaseAlgorithmConfig):
    clip_range: float = 1e-4
    clip_schedule: str = "constant"
    use_kl_penalty: bool = True
    ratio_reg_coef: float = 0.0
    eta: float = 1.0
    sde_type: str = "flow"
    shift: float = 3.0
    training_share_rollout_indices: bool = True
    rollout_scheduler_config: Dict[str, Any] = field(default_factory=dict)
    training_scheduler_config: Dict[str, Any] = field(default_factory=dict)
    skip_last_timestep: bool = False
    skip_initial_timesteps: int = 0
    model_type: str = "default"


def _save_training_debug_tensor(
    base_dir: str,
    step_idx: int,
    name: str,
    tensor: torch.Tensor,
    rank: int = 0,
    *,
    append: bool = False,
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


@register_algorithm(component_name="grpo", component_cfg=GRPOAlgorithmConfig)
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

    def __init__(
        self,
        *,
        config: GRPOAlgorithmConfig,
        **kwargs,
    ):
        if not isinstance(config, GRPOAlgorithmConfig):
            raise TypeError(f"{type(self).__name__} expects GRPOAlgorithmConfig, got {type(config).__name__}.")
        super().__init__(
            kl_coef=config.kl_coef,
            component_mix_stage=config.component_mix_stage,
            adv_normalization_scope=config.adv_normalization_scope,
            samples_per_prompt=config.samples_per_prompt,
            num_inference_steps=config.num_inference_steps,
            eval_ema_decay=config.eval_ema_decay,
            eval_ema_update_interval=config.eval_ema_update_interval,
            epsilon=config.epsilon,
            clip_max=config.clip_max,
            use_global_std=config.use_global_std,
            trim_outliers_ratio=config.trim_outliers_ratio,
            **kwargs,
        )
        self.config = config
        self.clip_range = config.clip_range
        self.clip_schedule = config.clip_schedule
        self.use_kl_penalty = config.use_kl_penalty
        self.ratio_reg_coef = config.ratio_reg_coef
        self._eta = config.eta
        self._sde_type = config.sde_type
        self._shift = config.shift
        self.model_type = config.model_type
        self.training_share_rollout_indices = bool(config.training_share_rollout_indices)
        self.rollout_scheduler_config = dict(config.rollout_scheduler_config)
        self.training_scheduler_config = dict(config.training_scheduler_config or self.rollout_scheduler_config)
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
        self.skip_last_timestep = config.skip_last_timestep
        self.skip_initial_timesteps = config.skip_initial_timesteps

        # Train-side objective state
        self._debug_output_dir = None  # Set externally for train-inference consistency debugging
        self._debug_dumped_steps: set = set()  # Track which steps already dumped (one-shot guard)

    @property
    def eta(self) -> float:
        return self._eta

    @property
    def sde_type(self) -> str:
        return self._sde_type

    @property
    def time_shift(self) -> float:
        return self._shift

    def get_sampling_requirements(self) -> SamplingRequirements:
        """Return GRPO sampling requirements."""
        return SamplingRequirements(
            requires_trajectory=True,
            requires_log_prob=True,
            requires_embeddings=True,
        )

    def get_ema_spec(self) -> EMASpec:
        return EMASpec(
            enable_eval_ema=True,
            eval_ema_decay=self.eval_ema_decay,
            eval_ema_update_interval=self.eval_ema_update_interval,
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

    def get_sampler_validation_config(self, *, allow_replay: bool) -> Dict[str, Any]:
        return {
            "allow_replay": bool(allow_replay),
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
        from diffusionrl.types.training_batch import TrainingBatch

        del kwargs
        if not isinstance(batch, TrainingBatch):
            raise TypeError(f"{type(self).__name__} expects TrainingBatch, got {type(batch).__name__}")

        step_indices = batch.resolved_step_indices[:-1]
        step_labels = set(int(v) for v in step_indices.tolist())
        if not step_labels:
            return tuple()

        requested_steps = set(int(i) for i in self.training_indices_scheduler.get_sde_indices(current_step))
        filtered_steps = self.get_filtered_training_indices(
            requested_steps,
            len(step_labels),
        )
        missing_steps = sorted(int(i) for i in filtered_steps if int(i) not in step_labels)
        if missing_steps:
            raise ValueError(
                f"{type(self).__name__}.resolve_training_timesteps selected steps "
                f"not present in batch: missing={missing_steps}, "
                f"available={sorted(step_labels)}"
            )
        if not filtered_steps:
            return tuple()
        selected_positions = [
            pos for pos, step_label in enumerate(step_indices.tolist()) if int(step_label) in filtered_steps
        ]
        return batch.timesteps[selected_positions]

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
        model_bundle: ModelBundle,
        timestep_data: TimestepData,
        advantages: torch.Tensor,
        *,
        ctx: ForwardContext,
        sigmas: Optional[torch.Tensor] = None,
        ref_model_bundle: Optional[ModelBundle] = None,
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
            if not getattr(self, "_logged_old_log_probs_none", False):
                logger.info(
                    "GRPO loss: old_log_probs is None at timestep_idx=%s — skipping as ode_step (will suppress further messages)",
                    timestep_idx,
                )
                self._logged_old_log_probs_none = True
            return torch.tensor(0.0, device=device, requires_grad=True), {
                "skip_reason": "ode_step",
                "timestep_idx": timestep_idx,
            }
        if not getattr(self, "_logged_old_log_probs_attached", False):
            logger.info(
                "GRPO loss: old_log_probs present at timestep_idx=%s, shape=%s, mean=%.4E",
                timestep_idx,
                tuple(old_log_probs.shape),
                float(old_log_probs.mean().item()),
            )
            self._logged_old_log_probs_attached = True

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

        pred = model_bundle.forward_denoiser(
            latents=latents,
            sigma=sigma,
            ctx=ctx,
        )

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
                    _rank,
                    timestep_idx,
                    latents.shape[0],
                    list(latents.shape),
                    _append,
                )
            _training_debug_dir = os.path.join(_resolved_debug_dir, "training")
            _save_training_debug_tensor(_training_debug_dir, timestep_idx, "noise_pred", pred, _rank, append=_append)
            _save_training_debug_tensor(
                _training_debug_dir, timestep_idx, "latents_input", latents, _rank, append=_append
            )
            _save_training_debug_tensor(
                _training_debug_dir, timestep_idx, "latents_output", next_latents, _rank, append=_append
            )
            _save_training_debug_tensor(
                _training_debug_dir, timestep_idx, "prev_sample_mean", prev_sample_mean, _rank, append=_append
            )
            _save_training_debug_tensor(
                _training_debug_dir, timestep_idx, "new_log_prob", new_log_prob, _rank, append=_append
            )
            _save_training_debug_tensor(
                _training_debug_dir, timestep_idx, "old_log_prob", old_log_probs, _rank, append=_append
            )
            _save_training_debug_tensor(_training_debug_dir, timestep_idx, "ratio", ratio, _rank, append=_append)
            _save_training_debug_tensor(
                _training_debug_dir, timestep_idx, "advantages", advantages, _rank, append=_append
            )
            _save_training_debug_tensor(
                _training_debug_dir, timestep_idx, "sigma", sigma.unsqueeze(0) if sigma.dim() == 0 else sigma, _rank
            )
            _save_training_debug_tensor(
                _training_debug_dir,
                timestep_idx,
                "sigma_next",
                sigma_next.unsqueeze(0) if sigma_next.dim() == 0 else sigma_next,
                _rank,
            )
            _save_training_debug_tensor(
                _training_debug_dir, timestep_idx, "sigma_max", torch.tensor([sigma_max]), _rank
            )
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
        approx_kl = 0.5 * torch.mean(log_prob_diff**2)

        if not getattr(self, "_logged_loss_diag", False):
            logger.warning(
                "GRPO loss diag (one-shot): timestep_idx=%d adv[abs_mean=%.4E std=%.4E min=%.4E max=%.4E nonzero_frac=%.4f shape=%s] "
                "ratio[mean=%.4E std=%.4E min=%.4E max=%.4E] clip_fraction=%.4f approx_kl=%.4E policy_loss=%.4E",
                timestep_idx,
                float(advantages.abs().mean().item()),
                float(advantages.std().item()) if advantages.numel() > 1 else 0.0,
                float(advantages.min().item()),
                float(advantages.max().item()),
                float((advantages != 0).float().mean().item()),
                tuple(advantages.shape),
                float(ratio.mean().item()),
                float(ratio.std().item()) if ratio.numel() > 1 else 0.0,
                float(ratio.min().item()),
                float(ratio.max().item()),
                float(clip_fraction.item()),
                float(approx_kl.item()),
                float(policy_loss.item()),
            )
            self._logged_loss_diag = True

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
                model_bundle=model_bundle,
                latents=latents,
                next_latents=next_latents,
                sigma=sigma,
                sigma_next=sigma_next,
                sigma_max=sigma_max,
                prev_sample_mean=prev_sample_mean,
                ref_model_bundle=ref_model_bundle,
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
        model_bundle: ModelBundle,
        batch: Any,
        timesteps: Any = None,
        loss_scale: float = 1.0,
        **kwargs: Any,
    ) -> tuple:
        """GRPO loss + backward for a single micro-batch.

        Returns:
            ``(loss, metrics, num_timesteps, has_backward)``
        """
        from diffusionrl.types.training_batch import TrainingBatch

        if not isinstance(batch, TrainingBatch):
            raise TypeError(f"{type(self).__name__} expects TrainingBatch, got {type(batch).__name__}")

        model_bundle.transformer.train()

        available_steps = set(int(s) for s in batch.resolved_step_indices[:-1].tolist())
        valid_step_indices = sorted(int(i) for i in batch.sde_indices if int(i) in available_steps)
        num_timesteps = len(valid_step_indices)
        if num_timesteps == 0:
            return 0.0, {}, 0, False

        ctx = batch.forward_context
        total_loss = 0.0
        has_backward = False
        all_metrics: Dict[str, Any] = {}
        timestep_metrics: List[Dict[str, Any]] = []

        for t_idx in valid_step_indices:
            timestep_data = batch.get_timestep_data_by_step(t_idx)
            loss_t, metrics_t = self.compute_loss(
                model_bundle=model_bundle,
                timestep_data=timestep_data,
                advantages=batch.advantages,
                ctx=ctx,
            )
            scaled_loss = loss_t * loss_scale / num_timesteps
            scaled_loss.backward()
            has_backward = True
            total_loss += scaled_loss.detach().item()

            for key, value in metrics_t.items():
                val = value.item() if isinstance(value, torch.Tensor) else value
                all_metrics[f"t{t_idx}_{key}"] = val
            timestep_metrics.append(metrics_t)

        all_metrics.update(aggregate_numeric_metrics(timestep_metrics))

        return total_loss, all_metrics, num_timesteps, has_backward

    # ------------------------------------------------------------------
    # Forward dispatch
    # ------------------------------------------------------------------

    def _compute_kl_penalty(
        self,
        model_bundle: ModelBundle,
        latents: torch.Tensor,
        next_latents: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        sigma_max: float,
        prev_sample_mean: torch.Tensor,
        ref_model_bundle: Optional[ModelBundle],
        ctx: ForwardContext,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Compute KL penalty between policy and reference."""
        ref_prev_sample_mean = None
        model = model_bundle.transformer

        adapter_model = model.module if hasattr(model, "module") else model
        if hasattr(adapter_model, "disable_adapter"):
            try:
                with torch.no_grad():
                    with adapter_model.disable_adapter():
                        ref_pred = model_bundle.forward_denoiser(
                            latents=latents,
                            sigma=sigma,
                            ctx=ctx,
                        )

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

        if ref_prev_sample_mean is None and ref_model_bundle is not None:
            with torch.no_grad():
                ref_pred = ref_model_bundle.forward_denoiser(
                    latents=latents,
                    sigma=sigma,
                    ctx=ctx,
                )

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
            std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * self.eta

            kl_loss = ((prev_sample_mean - ref_prev_sample_mean) ** 2).mean(
                dim=tuple(range(1, prev_sample_mean.ndim))
            ) / (2 * std_dev_t**2 + 1e-12)
            kl_loss = kl_loss.mean()

            return kl_loss

        return None
