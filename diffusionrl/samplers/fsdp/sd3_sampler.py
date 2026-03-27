"""
SD3 Image Sampler for GRPO Training (native direct sampling).

This sampler implements SDE sampling with log probability computation
for Stable Diffusion 3 image models using native PyTorch (FSDP-compatible).
    It supports:
    - Flow sampling (flow)
    - CPS sampling (cps) - recommended for flow_grpo
- Mixed ODE/SDE sampling for MixGRPO

Reference:
- flow_grpo: flow_grpo/diffusers_patch/sd3_sde_with_logprob.py
- DiffusionNFT: DiffusionNFT SD3 implementation
"""
import os
import logging
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Dict, List, Optional, Set, Any, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn

from ..base import BaseSampler, RolloutSamples
from diffusionrl.sde.kernels import get_sde_strategy
from diffusionrl.sde.runtime import (
    get_sigma_schedule,
    denoising_step,
)
from diffusionrl.types import LogProbData, PromptEmbeddings
from diffusionrl.utils.dtypes import parse_torch_dtype

logger = logging.getLogger(__name__)


def _save_debug_tensor(
    base_dir: str, step_idx: int, name: str, tensor: torch.Tensor,
    rank: int = 0, *, append: bool = False,
) -> None:
    """Save a debug tensor to disk. Only rank 0 saves to avoid conflicts.

    When *append=True* and a file already exists for this step+name, the new
    tensor is concatenated along dim-0 (batch) with the existing one.  This
    allows multiple sub-batch ``sample()`` calls within the same rollout to
    accumulate into a single file that covers the full local batch.
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


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Mirror diffusers SD3 dynamic-shift interpolation for image sequence length."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b



class SD3Sampler(BaseSampler):
    """
    SD3 image sampler with log probability computation for native direct sampling.

    This sampler is designed for Stable Diffusion 3 models and implements:
    - Standard SDE formulation
    - CPS (Coefficient-Preserving Sampling) - recommended for flow_grpo
    - Mixed ODE/SDE sampling for MixGRPO

    Example:
        sampler = SD3Sampler(
            model=sd3_model.transformer,
            text_encoder=sd3_model.text_encoder,
            sde_type="cps",
            eta=0.7,
            shift=3.0,
        )
        output = sampler.sample(
            prompts=["A beautiful sunset"],
            num_inference_steps=28,
            guidance_scale=7.0,
        )
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        text_encoder: Optional[Any] = None,
        text_encoder_2: Optional[Any] = None,
        text_encoder_3: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        tokenizer_2: Optional[Any] = None,
        tokenizer_3: Optional[Any] = None,
        vae: Optional[nn.Module] = None,
        scheduler: Optional[Any] = None,
        eta: float = 0.7,
        sde_type: str = "cps",  # SD3 typically uses "cps" or "flow"
        shift: float = 3.0,    # SD3 uses shift=3.0
        latent_channels: int = 16,  # SD3 uses 16 latent channels
        vae_scale_factor: int = 8,  # VAE 8x compression
        autocast_precision: Any = "bf16",
        trajectory_precision: Any = "fp16",
        logprob_precision: Any = "fp32",
    ):
        """
        Initialize SD3 sampler.

        Args:
            model: SD3 transformer model
            text_encoder: CLIP text encoder 1
            text_encoder_2: CLIP text encoder 2
            text_encoder_3: T5 text encoder
            tokenizer: CLIP tokenizer 1
            tokenizer_2: CLIP tokenizer 2
            tokenizer_3: T5 tokenizer
            vae: VAE for encoding/decoding
            eta: Noise level for SDE (controls stochasticity)
            sde_type: Transition rule ("flow", "cps", "dpm2")
            shift: Time shift parameter (SD3 uses 3.0)
            latent_channels: Number of latent channels (16 for SD3)
            vae_scale_factor: VAE spatial compression factor
        """
        super().__init__(eta=eta, sde_type=sde_type, shift=shift)
        self.model = model
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.text_encoder_3 = text_encoder_3
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.tokenizer_3 = tokenizer_3
        self.vae = vae
        self.scheduler = scheduler
        self.latent_channels = latent_channels
        self.vae_scale_factor = vae_scale_factor
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="autocast_precision")
        self.trajectory_dtype = parse_torch_dtype(trajectory_precision, field_name="trajectory_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="logprob_precision")
        # Track which steps have been dumped so subsequent sub-batch calls
        # use append mode to concatenate tensors along the batch dimension.
        self._debug_dumped_steps: set = set()

    def _predict_noise_with_cfg(
        self,
        *,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor],
        guidance_scale: float,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict noise with SD3 CFG using a single batched forward.

        Matches flow_grpo/DiffusionNFT SD3 behavior:
        - concat [uncond, cond] embeddings
        - concat [latents, latents]
        - single model forward
        - chunk(2) and apply CFG formula
        """
        if guidance_scale > 1.0:
            uncond_prompt_embeds = (
                negative_prompt_embeds
                if negative_prompt_embeds is not None
                else torch.zeros_like(prompt_embeds)
            )
            if pooled_prompt_embeds is not None:
                uncond_pooled_embeds = (
                    negative_pooled_prompt_embeds
                    if negative_pooled_prompt_embeds is not None
                    else torch.zeros_like(pooled_prompt_embeds)
                )
                pooled_batched = torch.cat(
                    [uncond_pooled_embeds, pooled_prompt_embeds], dim=0
                )
            else:
                pooled_batched = None

            noise_pred = self.model(
                hidden_states=torch.cat([latents, latents], dim=0),
                encoder_hidden_states=torch.cat(
                    [uncond_prompt_embeds, prompt_embeds], dim=0
                ),
                timestep=torch.cat([timestep, timestep], dim=0),
                pooled_projections=pooled_batched,
                return_dict=False,
            )[0]
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
            return noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

        return self.model(
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]

    def _resolve_runtime_device(
        self,
        prompt_embeds: Optional[torch.Tensor],
        latents: Optional[torch.Tensor],
    ) -> torch.device:
        """
        Resolve sampling compute device robustly under FSDP CPU offload.

        With FSDP cpu_offload, parameter.device can be CPU outside forward, so
        use runtime tensors first and fallback to current CUDA device.
        """
        if latents is not None and torch.is_tensor(latents) and latents.is_cuda:
            return latents.device
        if prompt_embeds is not None and torch.is_tensor(prompt_embeds) and prompt_embeds.is_cuda:
            return prompt_embeds.device
        if torch.cuda.is_available():
            try:
                return torch.device(f"cuda:{torch.cuda.current_device()}")
            except Exception:
                return torch.device("cuda")
        if latents is not None and torch.is_tensor(latents):
            return latents.device
        if prompt_embeds is not None and torch.is_tensor(prompt_embeds):
            return prompt_embeds.device
        return next(self.model.parameters()).device

    def sample(
        self,
        prompts: Optional[List[str]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,  # Not used by SD3 but kept for API compatibility
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
        height: int = 1024,
        width: int = 1024,
        latents: Optional[torch.Tensor] = None,
        base_seed: Optional[int] = None,
        sde_indices: Optional[Set[int]] = None,
        max_sequence_length: int = 256,
        init_same_noise: bool = False,
        samples_per_prompt: int = 1,
        noise_group_ids: Optional[List[str]] = None,
        debug_output_dir: Optional[str] = None,
        **kwargs,
    ) -> RolloutSamples:
        """
        Execute SDE sampling and return trajectories with log probabilities.

        Args:
            prompts: List of text prompts (used if prompt_embeds not provided)
            prompt_embeds: Pre-computed prompt embeddings [B, seq, hidden]
            pooled_prompt_embeds: Pooled prompt embeddings [B, hidden]
            text_ids: Not used by SD3 (kept for API compatibility)
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG guidance scale
            height: Output image height
            width: Output image width
            latents: Initial latents (if None, sampled from noise)
            generator: Random number generator
            sde_indices: Set of timestep indices to use SDE sampling.
                If None, all timesteps use SDE.
            max_sequence_length: Maximum T5 sequence length
            init_same_noise: Share initial noise across K samples for same prompt (DanceGRPO/MixGRPO)
            samples_per_prompt: Number of samples per prompt (for init_same_noise)
            debug_output_dir: If set, dump per-step SDE tensors to this directory for
                train-inference consistency debugging.

        Returns:
            RolloutSamples with trajectories, log_probs, etc.
        """
        if self.model is None:
            raise RuntimeError("Model not set. Initialize sampler with model parameter.")

        device = self._resolve_runtime_device(prompt_embeds, latents)
        embed_dtype = self.autocast_dtype
        latent_dtype = self.trajectory_dtype

        # Encode prompts if needed
        if prompt_embeds is None:
            if prompts is None:
                raise ValueError("Either prompts or prompt_embeds must be provided")
            prompt_embeds, pooled_prompt_embeds = self._encode_prompt(
                prompts, max_sequence_length, device, embed_dtype
            )
            if guidance_scale > 1.0 and negative_prompt_embeds is None:
                negative_prompt_embeds, negative_pooled_prompt_embeds = self._encode_prompt(
                    [""] * len(prompts), max_sequence_length, device, embed_dtype
                )

        batch_size = prompt_embeds.shape[0]

        # Embeddings use autocast_dtype (model compute precision);
        # latents use trajectory_dtype (compact storage precision).
        prompt_embeds = prompt_embeds.to(device=device, dtype=embed_dtype)
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=embed_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=embed_dtype)
        if negative_pooled_prompt_embeds is not None:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device=device, dtype=embed_dtype)
        if guidance_scale > 1.0 and negative_prompt_embeds is None:
            raise ValueError(
                "SD3 CFG requires negative_prompt_embeds when guidance_scale > 1.0."
            )

        # Calculate latent dimensions
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor

        # Initialize latents in trajectory_dtype (storage precision)
        if latents is None:
            from ..noise_utils import generate_latents
            latents = generate_latents(
                batch_size=batch_size,
                latent_shape=(self.latent_channels, latent_height, latent_width),
                device=device,
                dtype=latent_dtype,
                init_same_noise=init_same_noise,
                samples_per_prompt=samples_per_prompt,
                noise_group_ids=noise_group_ids,
                base_seed=base_seed,
            )
        else:
            latents = latents.to(device=device, dtype=latent_dtype)

        # Get sigma schedule (align with diffusers scheduler if available)
        patch_size = int(getattr(getattr(self.model, "config", None), "patch_size", 1) or 1)
        image_seq_len = (latents.shape[-2] // patch_size) * (latents.shape[-1] // patch_size)
        sigmas = self._get_sigma_schedule(
            num_inference_steps,
            device,
            image_seq_len=image_seq_len,
        )

        # Default: all timesteps use SDE; in deterministic (dpm2) mode, use ODE only.
        # When eta=0 the SDE degenerates to ODE (no noise), so skip SDE steps
        # to avoid division-by-zero in the log-prob variance term.
        if sde_indices is None:
            if self.uses_deterministic_solver:
                sde_indices = set()
            else:
                sde_indices = set(range(num_inference_steps))

        # Obtain a strategy instance once (important for stateful strategies like DPM2)
        strategy = get_sde_strategy(self.sde_type)
        if strategy.is_stateful:
            strategy.init_schedule(sigmas)

        # Storage for trajectory and log probs
        latents = latents.to(self.trajectory_dtype)
        trajectory: List[torch.Tensor] = [latents]
        log_probs_dict: Dict[int, torch.Tensor] = {}

        # Denoising loop
        autocast_ctx = (
            torch.autocast("cuda", self.autocast_dtype)
            if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        rank = int(os.environ.get("RANK", 0))
        _resolved_debug_dir_pre = debug_output_dir or os.environ.get("DIFFUSIONRL_DEBUG_OUTPUT_DIR")
        if _resolved_debug_dir_pre and rank == 0:
            logger.info(
                "Debug sampling: rank=%d batch_size=%d latents=%s sde_indices=%s "
                "already_dumped_steps=%s",
                rank, batch_size, list(latents.shape), sorted(sde_indices),
                sorted(self._debug_dumped_steps),
            )
        for i in range(num_inference_steps):
            sigma = sigmas[i].to(device)
            sigma_next = sigmas[i + 1].to(device)

            # Create timestep tensor (SD3 uses 0-1000 range)
            timestep = (sigma * 1000).expand(batch_size)

            # Forward pass with CFG
            with torch.no_grad():
                with autocast_ctx:
                    noise_pred = self._predict_noise_with_cfg(
                        latents=latents,
                        timestep=timestep,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        guidance_scale=guidance_scale,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    )

            # Debug: resolve sampling dump directory
            _resolved_debug_dir = debug_output_dir or os.environ.get("DIFFUSIONRL_DEBUG_OUTPUT_DIR")
            _sampling_debug_dir = None
            if _resolved_debug_dir is not None:
                _sampling_debug_dir = os.path.join(_resolved_debug_dir, "sampling")

            _pre_step_latents = latents.clone()

            # Unified step: eta controls SDE vs ODE behaviour.
            # For DPM2, sde_indices is empty so step_eta is always 0,
            # but DPM2Strategy.step() uses its own multi-step logic regardless.
            step_eta = self.eta if i in sde_indices else 0.0
            latents, log_prob, prev_sample_mean = denoising_step(
                noise_pred=noise_pred,
                sample=latents,
                sigma=sigma,
                sigma_next=sigma_next,
                eta=step_eta,
                sde_type=self.sde_type,
                sigma_max=sigmas[1].item(),
                strategy=strategy,
                step_index=i,
            )
            latents = latents.to(dtype=self.trajectory_dtype)
            trajectory.append(latents)

            if log_prob is not None:
                log_probs_dict[i] = log_prob.to(dtype=self.logprob_dtype)

                if _sampling_debug_dir is not None:
                    _append = i in self._debug_dumped_steps
                    self._debug_dumped_steps.add(i)
                    _save_debug_tensor(_sampling_debug_dir, i, "noise_pred", noise_pred, rank, append=_append)
                    _save_debug_tensor(_sampling_debug_dir, i, "latents_input", _pre_step_latents, rank, append=_append)
                    _save_debug_tensor(_sampling_debug_dir, i, "latents_output", latents, rank, append=_append)
                    _save_debug_tensor(_sampling_debug_dir, i, "prev_sample_mean", prev_sample_mean, rank, append=_append)
                    _save_debug_tensor(_sampling_debug_dir, i, "log_prob", log_prob, rank, append=_append)
                    _save_debug_tensor(_sampling_debug_dir, i, "sigma", sigma.unsqueeze(0) if sigma.dim() == 0 else sigma, rank)
                    _save_debug_tensor(_sampling_debug_dir, i, "sigma_next", sigma_next.unsqueeze(0) if sigma_next.dim() == 0 else sigma_next, rank)
                    _save_debug_tensor(_sampling_debug_dir, i, "timestep", timestep, rank)
                    if not _append and i == min(sde_indices):
                        _save_debug_tensor(_sampling_debug_dir, i, "sigmas_schedule", sigmas, rank)
                        if rank == 0:
                            import json
                            cfg_path = os.path.join(_sampling_debug_dir, "config.json")
                            with open(cfg_path, "w") as f:
                                json.dump({
                                    "sde_type": self.sde_type,
                                    "eta": self.eta,
                                    "shift": self.shift,
                                    "num_inference_steps": num_inference_steps,
                                    "guidance_scale": guidance_scale,
                                    "sde_indices": sorted(sde_indices),
                                    "batch_size": batch_size,
                                    "height": height,
                                    "width": width,
                                }, f, indent=2)

        # Stack trajectory
        trajectories = torch.stack(trajectory, dim=1)  # [B, T+1, C, H, W]

        # Create embeddings bundle
        embeddings = PromptEmbeddings(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )

        return RolloutSamples(
            latents=latents,
            timesteps=sigmas,
            aux={
                "trajectories": trajectories,
                "log_probs": LogProbData.from_dict(log_probs_dict),
                "embeddings": embeddings,
                "metadata": {
                    "sde_indices": sde_indices,
                    "engine_capabilities": {
                        "supports_logprob": True,
                        "supports_trajectory": True,
                        "supports_prompt_embeddings": True,
                    },
                    "trajectory_format": "dense_latent",
                    "timestep_type": "sigma",
                    "timestep_scale": 1.0,
                    "height": height,
                    "width": width,
                    "guidance_scale": guidance_scale,
                },
                "step_indices": torch.arange(
                    sigmas.shape[0], device=sigmas.device, dtype=torch.long
                ),
            },
        )

    def compute_log_prob_for_training(
        self,
        latents: torch.Tensor,
        prev_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor],
        timestep_index: int,
        sigma_schedule: torch.Tensor,
        guidance_scale: Optional[float] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute one-step log probability from replayed SD3 trajectory."""
        if self.model is None:
            raise RuntimeError("Model not set. Initialize sampler with model parameter.")

        device = latents.device
        batch_size = latents.shape[0]
        actual_guidance = float(guidance_scale if guidance_scale is not None else 1.0)

        sigma_schedule = sigma_schedule.to(device=device, dtype=torch.float32)
        if timestep_index < 0 or timestep_index >= int(sigma_schedule.shape[0]) - 1:
            raise ValueError(
                "timestep_index out of range for sigma_schedule: "
                f"index={timestep_index}, len={sigma_schedule.shape[0]}"
            )

        sigma = sigma_schedule[timestep_index]
        sigma_next = sigma_schedule[timestep_index + 1]
        # Keep timestep in float32 to avoid reduced-precision loss
        timestep = (sigma * 1000).expand(batch_size)

        # Embeddings use autocast_dtype (model compute precision), independent
        # of the trajectory storage dtype carried by latents.
        embed_dtype = self.autocast_dtype
        prompt_embeds = prompt_embeds.to(device=device, dtype=embed_dtype)
        pooled_prompt_embeds = (
            pooled_prompt_embeds.to(device=device, dtype=embed_dtype)
            if pooled_prompt_embeds is not None
            else None
        )
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=embed_dtype)
        if negative_pooled_prompt_embeds is not None:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(
                device=device, dtype=embed_dtype
            )

        autocast_ctx = (
            torch.autocast("cuda", self.autocast_dtype)
            if latents.is_cuda and self.autocast_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        self.model.train()
        with autocast_ctx:
            noise_pred = self._predict_noise_with_cfg(
                latents=latents,
                timestep=timestep,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                guidance_scale=actual_guidance,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            )

        if self.uses_deterministic_solver:
            raise ValueError(
                "Deterministic SD3 sampling does not define stochastic log-prob replay."
            )

        sigma_max = float(sigma_schedule[1].item()) if int(sigma_schedule.shape[0]) > 1 else 1.0
        _, log_prob, _ = denoising_step(
            noise_pred=noise_pred,
            sample=latents,
            sigma=sigma,
            sigma_next=sigma_next,
            eta=self.eta,
            prev_sample=prev_latents,
            sde_type=self.sde_type,
            sigma_max=sigma_max,
        )
        return log_prob.to(dtype=self.logprob_dtype)

    def _get_sigma_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
        image_seq_len: int = 256,
        sigmas: Optional[torch.Tensor] = None,
        mu: Optional[float] = None,
    ) -> torch.Tensor:
        if self.scheduler is None:
            return get_sigma_schedule(num_inference_steps, self.shift, device)
        try:
            from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

            scheduler_config = getattr(self.scheduler, "config", None)
            scheduler_kwargs: Dict[str, Any] = {}
            if scheduler_config is not None and bool(scheduler_config.get("use_dynamic_shifting", False)):
                if mu is None:
                    mu = calculate_shift(
                        image_seq_len,
                        int(scheduler_config.get("base_image_seq_len", 256)),
                        int(scheduler_config.get("max_image_seq_len", 4096)),
                        float(scheduler_config.get("base_shift", 0.5)),
                        float(scheduler_config.get("max_shift", 1.16)),
                    )
                scheduler_kwargs["mu"] = mu

            retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                sigmas=sigmas,
                **scheduler_kwargs,
            )
            sigmas = self.scheduler.sigmas
            if sigmas is None:
                raise ValueError("Scheduler did not provide sigmas")
            return sigmas.float().to(device)
        except Exception as e:
            raise RuntimeError(
                "Failed to use diffusers scheduler sigmas. "
                "Consider fixing the scheduler setup or explicitly switching "
                "to the internal sd3_time_shift schedule."
            ) from e

    def _encode_prompt(
        self,
        prompts: List[str],
        max_sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple:
        """Encode prompts using SD3's triple text encoder setup."""
        batch_size = len(prompts)

        # Check if we have all encoders
        if self.text_encoder is None or self.text_encoder_2 is None or self.text_encoder_3 is None:
            raise ValueError("All three text encoders required for SD3")
        if self.tokenizer is None or self.tokenizer_2 is None or self.tokenizer_3 is None:
            raise ValueError("All three tokenizers required for SD3")

        # CLIP 1 encoding
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)

        with torch.no_grad():
            clip_output_1 = self.text_encoder(
                text_input_ids,
                output_hidden_states=True,
            )
            clip_embeds_1 = clip_output_1.hidden_states[-2]
            pooled_1 = clip_output_1.text_embeds

        # CLIP 2 encoding
        text_inputs_2 = self.tokenizer_2(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs_2.input_ids.to(device)

        with torch.no_grad():
            clip_output_2 = self.text_encoder_2(
                text_input_ids_2,
                output_hidden_states=True,
            )
            clip_embeds_2 = clip_output_2.hidden_states[-2]
            pooled_2 = clip_output_2.text_embeds

        # Concatenate CLIP embeddings
        clip_embeds = torch.cat([clip_embeds_1, clip_embeds_2], dim=-1)
        pooled_embeds = torch.cat([pooled_1, pooled_2], dim=-1)

        # T5 encoding
        text_inputs_3 = self.tokenizer_3(
            prompts,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_3 = text_inputs_3.input_ids.to(device)

        with torch.no_grad():
            t5_output = self.text_encoder_3(text_input_ids_3)
            t5_embeds = t5_output.last_hidden_state

        # SD3 transformer expects:
        # - encoder_hidden_states: T5 embeddings only [B, seq_len, 4096]
        # - pooled_projections: CLIP pooled embeddings [B, 2048]
        prompt_embeds = t5_embeds

        return prompt_embeds.to(dtype=dtype), pooled_embeds.to(dtype=dtype)
