"""
GRPO Algorithm Implementation — unified algorithm + loss.

Standard GRPO with group normalization for advantages.  This class is the
single source of truth for both rollout-side requirements (sampling, advantages)
and training-side gradient computation (the former ``GRPOLoss``).

Key Features:
- PPO-style clipped policy gradient
- KL regularization (flow_grpo style with LoRA disable_adapter)
- Support for both SDE and Mixed sampling modes
- Per-timestep or aggregated loss
- Typed interface via compute_timestep() for type-safe training

Formula:
    ratio = exp(new_log_prob - old_log_prob)
    L = -E[min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)]
"""

import math
import logging
import os
import warnings
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from diffusionrl.types import TimestepData, PromptEmbeddings
from .base import BaseAlgorithm, SamplingRequirements

logger = logging.getLogger(__name__)


def _save_training_debug_tensor(base_dir: str, step_idx: int, name: str, tensor: torch.Tensor, rank: int = 0) -> None:
    """Save a debug tensor from training path to disk. Only rank 0 saves."""
    if rank != 0:
        return
    step_dir = os.path.join(base_dir, f"step_{step_idx:03d}")
    os.makedirs(step_dir, exist_ok=True)
    path = os.path.join(step_dir, f"{name}.pt")
    torch.save(tensor.detach().cpu().float(), path)


class GRPOAlgorithm(BaseAlgorithm):
    """
    Standard GRPO Algorithm — unified algorithm + loss.

    This class handles:
    1. Sampling requirements (get_sampling_requirements)
    2. Advantage computation (compute_advantages, inherited)
    3. Loss / gradient computation (compute_timestep)
    4. Static requirements declaration (declared_requirements)

    Features:
    - Group normalization for advantages (within prompt groups)
    - PPO-style clipped objective
    - Optional KL penalty
    - Support for both SDE and Mixed sampling modes

    Reference: DanceGRPO
    """

    @classmethod
    def declared_requirements(cls) -> Dict[str, bool]:
        """Declare data requirements for the contracts / validation pipeline."""
        return {
            "requires_trajectory": True,
            "requires_log_prob": True,
            "requires_embeddings": True,
        }

    @classmethod
    def from_config(cls, config: dict) -> "GRPOAlgorithm":
        """Create GRPOAlgorithm from a loss_config dictionary.

        Reads constructor parameters from config top-level keys,
        with ``loss_kwargs`` sub-dict taking precedence for overrides.
        """
        extra = config.get("loss_kwargs") or {}
        known_keys = {
            "clip_range",
            "clip_range_mode",
            "use_kl_penalty",
            "kl_coef",
            "ratio_reg_coef",
            "eta",
            "sde_type",
            "ignore_last",
            "frozen_init_timesteps",
            "model_type",
        }
        runtime_only_keys = {
            "use_ema",
            "ema_decay",
            "eval_ema_decay",
            "eval_ema_update_interval",
            "nft_timestep_mode",
            "nft_shuffle_timesteps",
            "nft_apply_shift",
            "decay_type",
            "ema_flat_steps",
            "ema_uprate",
            "ema_uphold",
            "old_adapter_name",
            "new_adapter_name",
            "shuffle_samples",
            "shuffle_seed",
        }
        unknown = sorted(key for key in extra.keys() if key not in known_keys and key not in runtime_only_keys)
        if unknown:
            warnings.warn(
                f"GRPOAlgorithm.from_config received unknown loss_kwargs keys: {unknown}. "
                "These keys are ignored by GRPO algorithm constructor.",
                stacklevel=3,
            )

        def _get(key, default):
            return extra.get(key, config.get(key, default))

        return cls(
            clip_range=float(_get("clip_range", 1e-4)),
            clip_range_mode=str(_get("clip_range_mode", "constant")),
            use_kl_penalty=bool(_get("use_kl_penalty", True)),
            kl_coef=float(_get("kl_coef", 0.01)),
            ratio_reg_coef=float(_get("ratio_reg_coef", 0.0)),
            eta=float(_get("eta", 0.7)),
            sde_type=str(_get("sde_type", "sde")),
            ignore_last=bool(_get("ignore_last", False)),
            frozen_init_timesteps=int(_get("frozen_init_timesteps", 0)),
            model_type=str(_get("model_type", "default")),
        )

    def __init__(
        self,
        clip_range: float = 1e-4,
        clip_range_mode: str = "constant",
        use_kl_penalty: bool = True,
        kl_coef: float = 0.01,
        ratio_reg_coef: float = 0.0,
        eta: float = 1.0,
        sde_type: str = "sde",
        ignore_last: bool = False,
        frozen_init_timesteps: int = 0,
        model_type: str = "default",
        # BaseAlgorithm params
        advantage_type: str = "group",
        epsilon: float = 1e-4,
        clip_max: float = 5.0,
        use_per_prompt_tracker: bool = False,
        per_prompt_mode: str = "batch",
        per_prompt_buffer_size: int = 16,
        per_prompt_min_count: int = 2,
        use_running_stats: bool = False,
        running_stats_warmup: int = 0,
        use_global_std: bool = False,
        **kwargs,
    ):
        """
        Initialize GRPO algorithm.

        Args:
            clip_range: PPO clip range (epsilon)
            clip_range_mode: Clip range schedule ("constant", "linear_decay", "cosine_decay")
            use_kl_penalty: Whether to add KL penalty
            kl_coef: KL penalty coefficient
            ratio_reg_coef: Coefficient for ratio regularization
            eta: SDE noise coefficient
            sde_type: Type of SDE ("sde", "cps", "dance")
            ignore_last: Skip the last timestep (t->0) in loss computation (MixGRPO).
                The last step has very low noise level, causing unstable log_prob.
            frozen_init_timesteps: Skip the first N timesteps in loss computation (MixGRPO).
                Early timesteps may have high variance.
            model_type: Model type for forward plugin selection ("flux", "sd3", "hunyuan", "default").
                If not specified, model type will be auto-detected from the model class name.
            advantage_type: Advantage normalization type ("global", "group", "per_prompt")
            epsilon: Small value for numerical stability
            clip_max: Maximum advantage clip value (optional)
            use_per_prompt_tracker: Use PerPromptStatTracker for cross-batch stats
            per_prompt_mode: "running" (tracker) or "batch" (per-batch stats)
            per_prompt_buffer_size: Buffer size for per-prompt tracker
            per_prompt_min_count: Min samples before using per-prompt stats
            use_running_stats: Use RunningMeanStd for cross-batch global normalization (DanceGRPO)
            running_stats_warmup: Warmup batches before using running stats
            use_global_std: Use global std instead of per-group std
            **kwargs: Additional arguments
        """
        super().__init__(
            clip_range=clip_range,
            kl_coef=kl_coef,
            advantage_type=advantage_type,
            epsilon=epsilon,
            clip_max=clip_max,
            use_per_prompt_tracker=use_per_prompt_tracker,
            per_prompt_mode=per_prompt_mode,
            per_prompt_buffer_size=per_prompt_buffer_size,
            per_prompt_min_count=per_prompt_min_count,
            use_running_stats=use_running_stats,
            running_stats_warmup=running_stats_warmup,
            use_global_std=use_global_std,
            **kwargs,
        )
        self.clip_range_mode = clip_range_mode
        self.use_kl_penalty = use_kl_penalty
        self.ratio_reg_coef = ratio_reg_coef
        self.eta = eta
        self.sde_type = sde_type
        self.model_type = model_type

        # MixGRPO stability controls
        self.ignore_last = ignore_last
        self.frozen_init_timesteps = frozen_init_timesteps

        # Loss-side instance vars (formerly on GRPOLoss)
        self._forward_plugin = None  # Lazy loaded
        self._debug_output_dir = None  # Set externally for train-inference consistency debugging
        self._debug_dumped_steps: set = set()  # Track which steps already dumped (one-shot guard)

    @classmethod
    def _grpo_kwargs_from_args(cls, args: Any) -> Dict[str, Any]:
        kwargs = cls._base_kwargs_from_args(args)
        kwargs.update(
            {
                "eta": getattr(args.sampling, "eta", 1.0),
                "sde_type": getattr(args.sampling, "sde_type", "sde"),
                "use_per_prompt_tracker": getattr(args.algorithm, "use_per_prompt_stat_tracker", False),
                "per_prompt_mode": getattr(args.algorithm, "per_prompt_mode", "batch"),
                "per_prompt_buffer_size": getattr(args.algorithm, "per_prompt_buffer_size", 16),
                "per_prompt_min_count": getattr(args.algorithm, "per_prompt_min_count", 2),
                "use_running_stats": getattr(args.algorithm, "use_running_stats", False),
                "running_stats_warmup": getattr(args.algorithm, "running_stats_warmup", 0),
                "use_global_std": getattr(args.algorithm, "use_global_std", False),
                "trimmed_ratio": getattr(args.algorithm, "trimmed_ratio", 0.0),
                "ignore_last": getattr(args.algorithm, "ignore_last", False),
                "frozen_init_timesteps": getattr(args.algorithm, "frozen_init_timesteps", 0),
            }
        )
        return kwargs

    @classmethod
    def from_args(cls, args: Any) -> "GRPOAlgorithm":
        """Construct GRPO algorithm from runtime args."""
        kwargs = cls._grpo_kwargs_from_args(args)
        kwargs.update(cls._algorithm_kwargs_from_args(args))
        return cls(**kwargs)

    def get_sampling_requirements(self) -> SamplingRequirements:
        """Return GRPO sampling requirements."""
        return SamplingRequirements(
            requires_trajectory=True,
            requires_log_prob=True,
        )

    # ==================================================================
    # Loss computation (formerly in GRPOLoss)
    # ==================================================================

    def get_clip_range(self, progress: float = 0.0) -> float:
        """
        Get the current clip range based on training progress.

        Args:
            progress: Training progress in [0, 1]

        Returns:
            Current clip range
        """
        if self.clip_range_mode == "constant":
            return self.clip_range
        elif self.clip_range_mode == "linear_decay":
            return self.clip_range * (1 - 0.5 * progress)
        elif self.clip_range_mode == "cosine_decay":
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

        Delegates to the canonical implementation in samplers/log_prob.py
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
        from diffusionrl.samplers.log_prob import compute_sde_log_prob

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

    def compute_timestep(
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
        """
        Compute GRPO loss for a single timestep using typed data structures.

        Args:
            model: The diffusion model being trained
            timestep_data: Typed container with latents, log_prob, sigmas for this step
            advantages: Per-sample advantages [B]
            embeddings: Typed container with prompt embeddings
            sigmas: Full sigma schedule (optional, uses timestep_data if not provided)
            ref_model: Reference model for KL penalty
            training_progress: Training progress in [0, 1]
            model_forward_fn: Custom forward function
            guidance_scale: CFG scale
            **kwargs: Additional model-specific arguments

        Returns:
            Tuple of (loss, metrics_dict)
        """
        device = advantages.device

        # Extract data from typed structures
        latents = timestep_data.latents
        next_latents = timestep_data.next_latents
        old_log_probs = timestep_data.log_prob
        sigma = timestep_data.sigma
        sigma_next = timestep_data.sigma_next
        timestep_idx = timestep_data.timestep_idx

        # Handle ODE step (no log_prob)
        if old_log_probs is None:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                "skip_reason": "ode_step",
                "timestep_idx": timestep_idx,
            }

        # Get sigma_max for log_prob computation
        _sigmas = sigmas if sigmas is not None else timestep_data.sigmas
        if _sigmas is not None:
            sigma_max = _sigmas[1].item() if _sigmas[1].dim() == 0 else _sigmas[1][0].item()
        else:
            raise ValueError(
                "Cannot determine sigma_max: neither `sigmas` argument nor "
                "`timestep_data.sigmas` is provided. Ensure TimestepData is "
                "constructed with the full sigma schedule (e.g. via "
                "BackwardTrainingBatch.get_timestep_data_by_step())."
            )

        # Convert sigma to tensor if needed
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma, device=device)
        if not isinstance(sigma_next, torch.Tensor):
            sigma_next = torch.tensor(sigma_next, device=device)

        # Extract embeddings
        prompt_embeds = embeddings.prompt_embeds
        pooled_prompt_embeds = embeddings.pooled_prompt_embeds
        negative_prompt_embeds = getattr(embeddings, "negative_prompt_embeds", None)
        negative_pooled_prompt_embeds = getattr(embeddings, "negative_pooled_prompt_embeds", None)

        # Forward pass through model
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
            pred = self._default_forward(
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

        # Compute new log probability
        pred_f = pred.float()
        latents_f = latents.float()
        next_latents_f = next_latents.float()

        new_log_prob, prev_sample_mean = self.compute_log_prob(
            pred=pred_f,
            sample=latents_f,
            next_sample=next_latents_f,
            sigma=sigma,
            sigma_next=sigma_next,
            sigma_max=sigma_max,
        )

        # Compute importance sampling ratio
        log_prob_diff = new_log_prob - old_log_probs
        ratio = torch.exp(log_prob_diff)

        # Debug: dump training-side tensors for consistency analysis.
        _resolved_debug_dir = self._debug_output_dir or os.environ.get("DIFFUSIONRL_DEBUG_OUTPUT_DIR")
        if _resolved_debug_dir is not None and timestep_idx not in self._debug_dumped_steps:
            self._debug_dumped_steps.add(timestep_idx)
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
            # Also save full sigma schedule once
            if _sigmas is not None and _rank == 0:
                _step_dir = os.path.join(_training_debug_dir, f"step_{timestep_idx:03d}")
                _sched_path = os.path.join(_step_dir, "sigmas_schedule.pt")
                if not os.path.exists(_sched_path):
                    torch.save(_sigmas.detach().cpu().float(), _sched_path)

        # Get clip range
        clip_range = self.get_clip_range(training_progress)

        # PPO-style clipped loss
        adv = advantages.detach()
        unclipped_loss = -adv * ratio
        clipped_loss = -adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

        # Compute metrics
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

        # KL penalty (optional)
        if self.use_kl_penalty and self.kl_coef > 0:
            kl_loss = self._compute_kl_penalty(
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
                total_loss = total_loss + self.kl_coef * kl_loss
                loss_terms["kl_loss"] = kl_loss.detach()

        # Ratio regularization
        if self.ratio_reg_coef > 0:
            ratio_reg = torch.mean((new_log_prob - old_log_probs) ** 2)
            total_loss = total_loss + self.ratio_reg_coef * ratio_reg
            loss_terms["ratio_reg"] = ratio_reg.detach()

        loss_terms["total_loss"] = total_loss.detach()
        loss_terms["timestep_idx"] = timestep_idx

        return total_loss, loss_terms

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
