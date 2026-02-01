"""
FastVideo Inference Engine.

FastVideo-based inference engine for efficient video generation.
Uses FastVideo's parallel execution and sequence/tensor parallelism.

Key Architecture:
- FastVideo uses MultiprocExecutor internally to manage multiple GPU workers
- Each worker runs in its own process with proper WORLD_SIZE/LOCAL_RANK/RANK
- Engine creates VideoGenerator which spawns MultiprocExecutor workers

Ray Scheduling Requirements:
- Single GPU: Ray actor requests num_gpus=1.0, standard scheduling
- Multi GPU (Slime pattern): Ray actor requests num_gpus=0.5 (fractional),
  NOSET_VISIBLE_DEVICES prevents Ray from setting CUDA_VISIBLE_DEVICES,
  engine sets CUDA_VISIBLE_DEVICES manually based on base_gpu_id.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Set
import torch

from ..engine import BaseInferenceEngine, EngineConfig, EngineCapabilities, register_engine
from diffusionrl.types import SamplerOutput
from diffusionrl.utils import load_function

logger = logging.getLogger(__name__)


@register_engine("fastvideo")
class FastVideoInferenceEngine(BaseInferenceEngine):
    """
    FastVideo Inference Engine.

    Uses FastVideo framework for efficient video generation.
    Supports:
    - Sequence parallelism (SP) across multiple GPUs
    - Tensor parallelism (TP) for model sharding
    - Efficient batched inference via MultiprocExecutor
    - Trajectory tracking for GRPO

    Ray Scheduling Requirements:
    - Ray actor MUST request num_gpus = fastvideo_args.num_gpus
    - FastVideo's MultiprocExecutor spawns num_gpus worker processes
    - Each worker uses one GPU with proper LOCAL_RANK/RANK/WORLD_SIZE
    - sp_size must <= num_gpus and num_gpus % sp_size == 0

    Example:
        # For SP=4, Ray actor needs num_gpus=4
        engine_config = {
            "num_gpus": 4,
            "sp_size": 4,
            "tp_size": 1,
        }
    """

    def __init__(self, config: EngineConfig):
        """
        Initialize FastVideo engine.

        Args:
            config: Engine configuration including:
                - num_gpus: Total GPUs for this engine (for Ray allocation)
                - sp_size: Sequence parallelism size (default 1, not num_gpus)
                - tp_size: Tensor parallelism size
        """
        super().__init__(config)
        self.generator = None  # FastVideo VideoGenerator
        self.sampler = None
        self._device = None
        self._last_decoded_videos: Optional[torch.Tensor] = None

        # FastVideo parallelism config
        # NOTE: sp_size defaults to 1 (no SP), not num_gpus
        # This allows num_gpus>1 with sp_size=1 for data parallel within MultiprocExecutor
        self._num_gpus = config.engine_kwargs.get("num_gpus", 1)
        self._sp_size = config.engine_kwargs.get("sp_size", 1)  # Default to 1, not num_gpus
        self._tp_size = config.engine_kwargs.get("tp_size", 1)

        # Validate: sp_size must <= num_gpus and num_gpus % sp_size == 0
        if self._sp_size > self._num_gpus:
            raise ValueError(
                f"sp_size ({self._sp_size}) must be <= num_gpus ({self._num_gpus})"
            )
        if self._num_gpus % self._sp_size != 0:
            raise ValueError(
                f"num_gpus ({self._num_gpus}) must be divisible by sp_size ({self._sp_size})"
            )

    @property
    def required_num_gpus(self) -> int:
        """Number of GPUs required by this engine (for Ray scheduling)."""
        return self._num_gpus

    def initialize(self, device: torch.device) -> None:
        """
        Initialize FastVideo generator.

        This creates a VideoGenerator which internally spawns MultiprocExecutor
        with num_gpus worker processes. Each worker initializes its own
        distributed environment.

        Args:
            device: Target device (used for context, actual devices managed by FastVideo)
        """
        self._device = device

        # When using NOSET mode, manually set CUDA_VISIBLE_DEVICES so that
        # MultiprocExecutor's child processes inherit the correct GPU range.
        base_gpu_id = self.config.engine_kwargs.get("base_gpu_id", 0)
        force_set_visible = bool(self.config.engine_kwargs.get("force_set_cuda_visible_devices", False))
        if force_set_visible or base_gpu_id > 0:
            gpu_range = ",".join(str(base_gpu_id + i) for i in range(self._num_gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_range
            logger.info(f"FastVideo NOSET mode: CUDA_VISIBLE_DEVICES={gpu_range}")

        logger.info(
            f"Initializing FastVideo engine: num_gpus={self._num_gpus}, "
            f"sp_size={self._sp_size}, tp_size={self._tp_size}"
        )

        # Check for FastVideo availability
        try:
            from fastvideo.entrypoints.video_generator import VideoGenerator
            from fastvideo.fastvideo_args import FastVideoArgs
        except ImportError:
            raise ImportError(
                "FastVideo is required. Install it with: pip install fastvideo"
            )

        # Apply runtime patches (weight update utilities)
        self._apply_patches()

        # Build FastVideoArgs with parallelism settings
        fastvideo_kwargs = {
            "model_path": self.config.pretrained_model_path,
            "num_gpus": self._num_gpus,
            "sp_size": self._sp_size,
            "tp_size": self._tp_size,
            "inference_mode": True,
            "eta": self.config.eta,
            # CPU offload settings (can be overridden via engine_kwargs)
            "dit_cpu_offload": self.config.engine_kwargs.get("dit_cpu_offload", False),
            "dit_layerwise_offload": self.config.engine_kwargs.get("dit_layerwise_offload", False),
            "text_encoder_cpu_offload": self.config.engine_kwargs.get("text_encoder_cpu_offload", False),
            "image_encoder_cpu_offload": self.config.engine_kwargs.get("image_encoder_cpu_offload", False),
            "vae_cpu_offload": self.config.engine_kwargs.get("vae_cpu_offload", False),
            "pin_cpu_memory": self.config.engine_kwargs.get("pin_cpu_memory", True),
        }

        # Merge any additional FastVideo kwargs
        extra_kwargs = self.config.engine_kwargs.get("fastvideo_kwargs", {})
        fastvideo_kwargs.update(extra_kwargs)

        # Create FastVideoArgs.
        # Experimental/ad-hoc compatibility layer: FastVideo arg schemas differ
        # across revisions (e.g. `eta` may not exist on FastVideoArgs).
        creation_kwargs = dict(fastvideo_kwargs)
        stripped_kwargs: Dict[str, Any] = {}
        while True:
            try:
                fastvideo_args = FastVideoArgs.from_kwargs(**creation_kwargs)
                break
            except TypeError as e:
                msg = str(e)
                m = re.search(r"unexpected keyword argument '([^']+)'", msg)
                if not m:
                    raise
                bad_key = m.group(1)
                if bad_key not in creation_kwargs:
                    raise
                stripped_kwargs[bad_key] = creation_kwargs.pop(bad_key)
                logger.warning(
                    "FastVideoArgs does not accept '%s'; preserving as runtime extra only.",
                    bad_key,
                )
        # Re-attach stripped keys so monkey patches can still read runtime hints
        # (e.g. gpu_worker_patch checks fastvideo_args.eta when available).
        for key, value in stripped_kwargs.items():
            setattr(fastvideo_args, key, value)

        # Create VideoGenerator (this spawns MultiprocExecutor internally)
        # MultiprocExecutor will spawn num_gpus worker processes
        self.generator = VideoGenerator.from_fastvideo_args(fastvideo_args)
        self._fastvideo_args = fastvideo_args
        if hasattr(self.generator, "get_grpo_patch_status"):
            logger.info("FastVideo patch status: %s", self.generator.get_grpo_patch_status())

        # Initialize sampler for log_prob computation
        self._init_sampler()

        self._is_initialized = True
        logger.info(
            f"FastVideo engine initialized successfully. "
            f"Executor type: {type(self.generator.executor).__name__}"
        )

    def _apply_patches(self) -> None:
        from diffusionrl.patches.fastvideo import apply_all

        apply_all()

    def _init_sampler(self) -> None:
        """Initialize sampler for trajectory replay and log_prob computation."""
        sampler_path = self.config.engine_kwargs.get(
            "sampler_path",
            "diffusionrl.samplers.fastvideo.fastvideo_sampler.FastVideoSampler"
        )
        sampler_cls = load_function(sampler_path)

        # Note: For Phase 1, sampler needs model for log_prob computation
        # After FastVideo PR, log_prob will be computed during sampling
        self.sampler = sampler_cls(
            model=None,  # Model is inside FastVideo workers
            generator=self.generator,
            eta=self.config.eta,
            sde_type=self.config.sde_type,
            shift=self.config.shift,
        )

    def generate(
        self,
        prompts: Optional[List[str]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        latents: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        sde_indices: Optional[Set[int]] = None,
        return_trajectory: bool = True,
        **kwargs,
    ) -> SamplerOutput:
        """
        Generate videos with log probabilities using FastVideo.

        Uses VideoGenerator.generate_video() internally, which dispatches
        to MultiprocExecutor workers for parallel execution.

        Args:
            prompts: List of text prompts
            prompt_embeds: Pre-computed prompt embeddings (not used, FastVideo encodes internally)
            pooled_prompt_embeds: Pooled prompt embeddings (not used)
            encoder_attention_mask: Attention mask (not used)
            num_inference_steps: Override default steps
            guidance_scale: Override default guidance
            height: Override default height
            width: Override default width
            num_frames: Override default frames
            latents: Initial latents (not used, FastVideo samples internally)
            seed: Random seed
            sde_indices: SDE step indices for MixGRPO
            return_trajectory: Whether to return trajectory (default True for GRPO)
            **kwargs: Additional FastVideo arguments

        Returns:
            SamplerOutput with video trajectories and log_probs
        """
        if not self._is_initialized:
            raise RuntimeError("Engine not initialized")

        if self.generator is None:
            raise RuntimeError("Generator not loaded")

        self.ensure_ready_for_generate()
        self._last_decoded_videos = None

        # Use defaults if not specified
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        guidance_scale = guidance_scale or self.config.guidance_scale
        height = height or self.config.height
        width = width or self.config.width
        num_frames = num_frames or self.config.num_frames
        return_decoded_for_reward = bool(kwargs.pop("return_decoded_for_reward", False))

        # Build SamplingParam for FastVideo
        try:
            from fastvideo.configs.sample import SamplingParam
        except ImportError:
            raise ImportError("FastVideo is required")

        prompt_list: List[str] = []
        if isinstance(prompts, list):
            prompt_list = list(prompts)
        elif isinstance(prompts, str):
            prompt_list = [prompts]
        elif prompts is not None:
            prompt_list = [str(prompts)]
        if not prompt_list:
            raise ValueError("FastVideo engine requires non-empty prompts list.")

        sampling_param = SamplingParam.from_pretrained(self._fastvideo_args.model_path)
        sampling_param.update({
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "save_video": False,
            "return_frames": False,
            "return_trajectory_latents": bool(return_trajectory),
            "return_trajectory_decoded": return_decoded_for_reward,
        })

        # FastVideo's generate_video expects a single prompt string.
        # Keep this engine-side batching explicit and predictable.
        per_prompt_results: List[Dict[str, Any]] = []
        for idx, prompt in enumerate(prompt_list):
            local_kwargs = dict(kwargs)
            local_seed = seed
            if local_seed is not None:
                local_seed = int(local_seed) + idx
            if local_seed is not None:
                local_kwargs["seed"] = local_seed
            result = self.generator.generate_video(
                prompt=prompt,
                sampling_param=sampling_param,
                **local_kwargs,
            )
            if not isinstance(result, dict):
                raise RuntimeError(
                    f"Unexpected FastVideo output type: {type(result).__name__}. Expected dict."
                )
            per_prompt_results.append(result)
        result = self._merge_fastvideo_results(per_prompt_results)

        # Convert FastVideo output to SamplerOutput
        return self._convert_to_sampler_output(
            result,
            prompts=prompt_list,
            num_inference_steps=num_inference_steps,
            sde_indices=sde_indices,
            include_decoded_images=return_decoded_for_reward,
        )

    def _merge_fastvideo_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge per-prompt FastVideo outputs into a single batch-like dict."""
        if not results:
            raise ValueError("FastVideo result list is empty.")
        if len(results) == 1:
            return results[0]

        merged: Dict[str, Any] = {}

        # samples: [B, C, T, H, W]
        sample_tensors = [r.get("samples") for r in results if torch.is_tensor(r.get("samples"))]
        if len(sample_tensors) == len(results):
            merged["samples"] = torch.cat(sample_tensors, dim=0)

        # trajectory: [B, S, C, T, H, W]
        trajectory_tensors = [r.get("trajectory") for r in results if torch.is_tensor(r.get("trajectory"))]
        if len(trajectory_tensors) == len(results):
            merged["trajectory"] = torch.cat(trajectory_tensors, dim=0)
            merged["trajectory_timesteps"] = results[0].get("trajectory_timesteps")

        # trajectory_decoded: List[timestep] of [B, C, T, H, W]
        decoded_lists = [r.get("trajectory_decoded") for r in results]
        if all(isinstance(x, list) for x in decoded_lists):
            step_count = len(decoded_lists[0])
            if all(len(x) == step_count for x in decoded_lists):
                merged_decoded: List[torch.Tensor] = []
                for step_idx in range(step_count):
                    step_tensors = []
                    for per_prompt in decoded_lists:
                        step_tensor = per_prompt[step_idx]
                        if torch.is_tensor(step_tensor):
                            step_tensors.append(step_tensor)
                    if len(step_tensors) == len(decoded_lists):
                        merged_decoded.append(torch.cat(step_tensors, dim=0))
                if merged_decoded:
                    merged["trajectory_decoded"] = merged_decoded

        # prompt embeddings / attention masks (optional, patch-dependent)
        for field in ("prompt_embeds", "negative_prompt_embeds", "encoder_attention_mask"):
            values = [r.get(field) for r in results if torch.is_tensor(r.get(field))]
            if len(values) == len(results):
                try:
                    merged[field] = torch.cat(values, dim=0)
                except Exception:
                    # Keep non-concatenable payload as-is from the first item.
                    merged[field] = values[0]

        merged["generation_time"] = float(
            sum(float(r.get("generation_time", 0.0) or 0.0) for r in results)
        )
        merged["prompts"] = [r.get("prompts") for r in results]
        return merged

    def _convert_to_sampler_output(
        self,
        fastvideo_result: Dict[str, Any],
        prompts: Optional[List[str]],
        num_inference_steps: int,
        sde_indices: Optional[Set[int]],
        include_decoded_images: bool = False,
    ) -> SamplerOutput:
        """Convert FastVideo result dict to SamplerOutput."""
        from diffusionrl.types import LogProbData, PromptEmbeddings

        # Extract from FastVideo output
        samples = fastvideo_result.get("samples")  # [B, C, T, H, W]
        trajectory = fastvideo_result.get("trajectory")  # List or tensor
        trajectory_timesteps = fastvideo_result.get("trajectory_timesteps")

        # Stack trajectory if it's a list
        trajectories = None
        if trajectory is not None:
            if isinstance(trajectory, list):
                trajectories = torch.stack(trajectory, dim=1)
            else:
                trajectories = trajectory

        # Timesteps
        timesteps = None
        if trajectory_timesteps is not None:
            if isinstance(trajectory_timesteps, list):
                timesteps = torch.tensor(trajectory_timesteps, dtype=torch.float32)
            else:
                timesteps = trajectory_timesteps.float()
        elif trajectories is not None:
            timesteps = torch.arange(
                trajectories.shape[1],
                dtype=torch.float32,
                device=trajectories.device,
            )
        else:
            timesteps = torch.arange(num_inference_steps + 1, dtype=torch.float32)

        # For log_probs: compute using trajectory replay if trajectory available
        log_probs_dict = {}
        if trajectories is not None and self.sampler is not None:
            # Determine SDE indices
            if sde_indices is None:
                sde_indices = set(range(num_inference_steps))

            # Note: This is Phase 1 - extra forward for log_prob
            # After FastVideo PR, log_prob will come from fastvideo_result
            # For now, we skip log_prob computation here and let training handle it
            pass

        # Keep sampler contract semantics explicit: `latents` must represent
        # final latent state, not decoded pixel samples.
        final_latents = None
        if trajectories is not None:
            final_latents = trajectories[:, -1]
        elif torch.is_tensor(samples):
            final_latents = samples
        else:
            raise RuntimeError("FastVideo output missing both trajectory and samples tensors.")

        decoded_images = None
        if include_decoded_images:
            decoded_images = self._extract_decoded_images(fastvideo_result)
        decoded_videos = self._last_decoded_videos if include_decoded_images else None

        prompt_embeds = fastvideo_result.get("prompt_embeds")
        negative_prompt_embeds = fastvideo_result.get("negative_prompt_embeds")
        encoder_attention_mask = fastvideo_result.get("encoder_attention_mask")
        embeddings = None
        if torch.is_tensor(prompt_embeds):
            embeddings = PromptEmbeddings(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=fastvideo_result.get("pooled_prompt_embeds"),
                encoder_attention_mask=encoder_attention_mask
                if torch.is_tensor(encoder_attention_mask)
                else None,
                negative_prompt_embeds=negative_prompt_embeds
                if torch.is_tensor(negative_prompt_embeds)
                else None,
            )

        return SamplerOutput(
            latents=final_latents,
            timesteps=timesteps,
            trajectories=trajectories,
            log_probs=LogProbData.from_dict(log_probs_dict) if log_probs_dict else None,
            embeddings=embeddings,
            decoded_images=decoded_images,
            metadata={
                "generator_type": "fastvideo",
                "engine_capabilities": self.get_capabilities_dict(),
                "num_gpus": self._num_gpus,
                "sp_size": self._sp_size,
                "generation_time": fastvideo_result.get("generation_time"),
                "sde_indices": sorted(sde_indices) if sde_indices is not None else None,
                "trajectory_format": "video_dense_latent",
                "timestep_type": "sigma",
                "timestep_scale": 1.0,
                "has_decoded_images": bool(decoded_images),
                "has_prompt_embeds": embeddings is not None,
                "decoded_videos": decoded_videos,
            },
            contract_version="v1",
            step_indices=torch.arange(timesteps.shape[0], dtype=torch.long, device=timesteps.device),
        )

    def _extract_decoded_images(self, fastvideo_result: Dict[str, Any]) -> Optional[List[Any]]:
        """Extract middle-frame PIL images from decoded FastVideo outputs."""
        decoded_videos = None
        trajectory_decoded = fastvideo_result.get("trajectory_decoded")
        if isinstance(trajectory_decoded, list) and len(trajectory_decoded) > 0:
            decoded_videos = trajectory_decoded[-1]
        elif torch.is_tensor(fastvideo_result.get("samples")):
            # Fallback: FastVideo "samples" are already decoded pixel tensors.
            decoded_videos = fastvideo_result.get("samples")

        if not torch.is_tensor(decoded_videos):
            return None

        self._last_decoded_videos = decoded_videos
        return self._video_tensor_to_pil(decoded_videos)

    def _video_tensor_to_pil(self, videos: torch.Tensor) -> List[Any]:
        """Convert [B,C,T,H,W] or [B,C,H,W] tensors into PIL list (middle frame for video)."""
        from PIL import Image
        import numpy as np

        data = videos.detach().cpu()
        if data.ndim == 5:
            data = data[:, :, data.shape[2] // 2]
        if data.ndim != 4:
            raise ValueError(f"Expected 4D or 5D tensor for decode, got shape={tuple(data.shape)}")

        pil_images: List[Any] = []
        for img in data:
            img_np = img.permute(1, 2, 0).float().clamp(0, 1).mul(255).byte().numpy()
            pil_images.append(Image.fromarray(img_np))
        return pil_images

    def encode_prompt(
        self,
        prompts: List[str],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text prompts.

        Note: FastVideo handles text encoding internally during generate_video().
        This method is provided for compatibility but may not be needed.

        Args:
            prompts: List of text prompts
            **kwargs: Additional encoding arguments

        Returns:
            Dict with prompt info (actual encoding happens in workers)
        """
        # FastVideo encodes prompts internally via MultiprocExecutor workers
        # We just return the prompts for reference
        return {"prompts": prompts}

    def update_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Update model weights from training.

        For FastVideo with MultiprocExecutor, this is complex because
        the model lives inside worker processes. Options:
        1. Shutdown and reinitialize with new checkpoint
        2. Use FastVideo's checkpoint mechanism
        3. For GRPO, typically we save checkpoints and reload

        Args:
            state_dict: New model state dict
        """
        if self.generator is None:
            raise RuntimeError("FastVideo generator not initialized")

        import os
        import time

        # Save to shared filesystem (prefer /dev/shm for speed if available)
        weight_sync_dir = self.config.engine_kwargs.get(
            "weight_sync_dir",
            "/dev/shm/grpo_weight_sync",
        )
        os.makedirs(weight_sync_dir, exist_ok=True)

        use_safetensors = False
        try:
            from safetensors.torch import save_file

            use_safetensors = True
        except Exception:
            use_safetensors = False

        suffix = "safetensors" if use_safetensors else "pt"
        ckpt_path = os.path.join(
            weight_sync_dir,
            f"fastvideo_weights_{os.getpid()}_{int(time.time_ns())}.{suffix}",
        )

        cpu_state = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
        if use_safetensors:
            save_file(cpu_state, ckpt_path)
        else:
            import torch

            torch.save(cpu_state, ckpt_path)

        try:
            if not hasattr(self.generator, "update_weights_from_path"):
                raise RuntimeError("FastVideo update_weights_from_path patch not applied")
            self.generator.update_weights_from_path(ckpt_path, strict=False)
            logger.info("FastVideo engine weights updated from %s", ckpt_path)
        finally:
            # Best-effort cleanup
            try:
                os.remove(ckpt_path)
            except OSError:
                pass

    def update_weights_from_path(self, checkpoint_path: str, strict: bool = False) -> None:
        """Update FastVideo worker weights from a shared checkpoint path."""
        if self.generator is None:
            raise RuntimeError("FastVideo generator not initialized")
        if not hasattr(self.generator, "update_weights_from_path"):
            raise RuntimeError("FastVideo update_weights_from_path patch not applied")
        self.generator.update_weights_from_path(checkpoint_path, strict=strict)

    def offload(self) -> None:
        """
        Offload FastVideo engine.

        Offload FastVideo worker models to CPU without destroying workers.
        """
        if self.generator is None:
            raise RuntimeError("FastVideo generator not initialized")
        if not hasattr(self.generator, "offload_model"):
            raise RuntimeError("FastVideo offload patch not applied")
        self.generator.offload_model()
        self._is_offloaded = True
        logger.info("FastVideo engine offloaded (workers retained)")

    def onload(self) -> None:
        """
        Mark FastVideo engine as active again.
        """
        if not self._is_offloaded:
            return
        if self.generator is None:
            raise RuntimeError("FastVideo generator not initialized")
        if not hasattr(self.generator, "onload_model"):
            raise RuntimeError("FastVideo onload patch not applied")
        self.generator.onload_model()
        self._is_offloaded = False
        logger.info("FastVideo engine onloaded")

    def onload_weights(self) -> None:
        self.onload()

    def onload_post_update(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents for reward extraction.

        FastVideo's decoded videos are captured during generate() when
        return_decoded_for_reward=True. Reuse that cache here to avoid
        extra worker RPC.
        """
        if self._last_decoded_videos is not None:
            cached = self._last_decoded_videos
            if int(cached.shape[0]) == int(latents.shape[0]):
                return cached
        raise RuntimeError(
            "FastVideo decode cache is unavailable. "
            "Call generate(..., return_decoded_for_reward=True) before decode_latents()."
        )

    def shutdown(self) -> None:
        """Gracefully shutdown the engine and release resources."""
        if self.generator is not None:
            try:
                self.generator.shutdown()
            except Exception as e:
                logger.warning(f"Error during FastVideo shutdown: {e}")
            self.generator = None

        self.sampler = None
        self._is_initialized = False
        logger.info("FastVideo engine shutdown complete")

    @property
    def supports_distributed(self) -> bool:
        """FastVideo supports sequence/tensor parallelism."""
        return self._num_gpus > 1

    @property
    def requires_external_service(self) -> bool:
        """FastVideo runs embedded with MultiprocExecutor."""
        return False

    def get_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory info."""
        info = super().get_memory_info()
        info.update({
            "num_gpus": self._num_gpus,
            "sp_size": self._sp_size,
            "tp_size": self._tp_size,
        })
        return info

    def get_capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_logprob=False,
            supports_trajectory=True,
            supports_prompt_embeddings=False,
            supports_guidance_scale=True,
            supports_staged_onload=True,
            weight_sync_mode="checkpoint_path",
        )
