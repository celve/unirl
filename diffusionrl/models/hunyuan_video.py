"""
HunyuanVideo Model Bundle.

This bundle targets Tencent's HunyuanVideo (text-to-video) family
specifically — it does **not** cover Hunyuan-Image. The class /
``model_type`` naming is therefore explicitly ``HunyuanVideo`` /
``hunyuan_video``.

Reference: DanceGRPO implementation
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from diffusionrl.config.registration import register_config

from .base import ModelBundle
from .config import ModelBundleConfig

logger = logging.getLogger(__name__)


# Llava chat template + ``crop_start`` mirror diffusers'
# ``HunyuanVideoPipeline.DEFAULT_PROMPT_TEMPLATE`` exactly. Keeping the
# constant text in sync with upstream is what makes the produced
# ``prompt_embeds`` distribution match what the HunyuanVideo transformer
# was trained on (system header is dropped via ``crop_start``; only the
# user-prompt embeddings reach the transformer).
DEFAULT_PROMPT_TEMPLATE: Dict[str, Any] = {
    "template": (
        "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
        "1. The main content and theme of the video."
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
        "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
        "4. background environment, light, style and atmosphere."
        "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    ),
    "crop_start": 95,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@register_config(
    group="model",
    name="hunyuan_video",
    target="diffusionrl.models.hunyuan_video.HunyuanVideoModelBundle",
)
@dataclass
class HunyuanVideoModelBundleConfig(ModelBundleConfig):
    """Bundle config for the original HunyuanVideo (1.0).

    Mirrors ``ModelBundleConfig`` field-for-field — the legacy
    ``HunyuanVideoModelBundle.__init__`` only consumes
    ``pretrained_model_ckpt_path``, ``vae_ckpt_path``, and
    ``text_encoder_ckpt_path`` (all already on the base class).

    A dedicated subclass is registered (instead of reusing
    ``ModelBundleConfig`` directly) so Hydra's structured-config registry
    keeps a one-Spec-per-name invariant and yamls can reference this bundle
    via ``defaults: - override /model: hunyuan_video``. This registration
    was lost during the ``Hunyuan*`` → ``HunyuanVideo*`` rename refactor;
    re-introduced here so 1.0 experiment recipes can compose like the
    1.5 / SD3 / Flux / WAN bundles.
    """

    pass


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


class HunyuanVideoModelBundle(ModelBundle):
    """
    HunyuanVideo model bundle.

    Components:
    - transformer: HunyuanVideo transformer (dual-stream)
    - vae: 3D VAE for video encoding/decoding
    - text_encoder: LLAMA + CLIP text encoder wrapper
    """

    def __init__(
        self,
        config: HunyuanVideoModelBundleConfig,
    ):
        """
        Initialize HunyuanVideo model bundle.

        Args:
            config: Typed model-bundle config.
        """
        super().__init__(config)

        self.vae_ckpt_path = config.vae_ckpt_path or config.pretrained_model_ckpt_path
        self.text_encoder_ckpt_path = config.text_encoder_ckpt_path or config.pretrained_model_ckpt_path

        # Honor the same training_only / skip_device_move / use_lora protocol as
        # WAN / SD3 / Flux / HunyuanVideo-1.5: when initialized for a training
        # actor, ``training_only=True`` must keep VAE + text_encoder OFF the
        # GPU (they get lazy-loaded by ``FSDPSamplingEngine.bind_model`` after
        # FSDP wrap via ``ModelBundle.load_aux_components``). The previous 1.0
        # bundle ignored this and loaded LlamaModel + CLIP + VAE on every
        # train actor at init, which inflated per-GPU static weight pressure
        # (~55 GB before FSDP shard / Adam states / activations) and caused
        # silent OOMs in ``_load_transformer`` whenever the reward stack also
        # claimed serious GPU memory (e.g. a Qwen2-VL video reward model).
        self.use_lora = config.use_lora
        self.lora_rank = config.lora_rank
        self.lora_alpha = config.lora_alpha
        self.lora_target_modules = config.lora_target_modules
        self.training_only = config.training_only
        self.skip_device_move = config.skip_device_move

        self.load()

    @property
    def model_type(self) -> str:
        return "hunyuan_video"

    @property
    def media_type(self) -> str:
        return "t2v"

    @classmethod
    def default_sampler_dotpath(cls) -> Optional[str]:
        return "diffusionrl.samplers.fsdp.hunyuan_video_sampler.FSDPHunyuanVideoSampler"

    @classmethod
    def default_sampler_engine(cls) -> Optional[str]:
        return "fsdp"

    @classmethod
    def supports_sglang_prompt_mode(cls) -> bool:
        return True

    @classmethod
    def default_lora_target_modules(cls) -> Optional[List[str]]:
        """Canonical LoRA targets for HunyuanVideo-1.0 dual-stream blocks.

        HunyuanVideo-1.0's diffusers-converted transformer
        (``HunyuanVideoTransformer3DModel``) uses the same
        ``Attention``-with-added-KV joint-attention + FFN naming as the 1.5
        / SD3 / WAN bundles, so the canonical SD3-family target list works
        as-is. This was previously ``None`` (delegating to the sampler's
        "wrap everywhere" default), which silently no-op'd
        ``model.use_lora=true`` on every 1.0 recipe — the YAML accepted the
        flag but nothing in the bundle actually injected adapters.
        """
        return [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_q_proj",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]

    def forward_denoiser(
        self,
        *,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        ctx,
    ) -> torch.Tensor:
        prompt_embeds = ctx.prompt_embeds
        if prompt_embeds is None:
            raise ValueError("HunyuanVideoModelBundle.forward_denoiser requires ctx.prompt_embeds.")
        pooled_prompt_embeds = getattr(ctx, "pooled_prompt_embeds", None)
        encoder_attention_mask = getattr(ctx, "encoder_attention_mask", None)
        guidance_scale = float(getattr(ctx, "guidance_scale", 1.0))

        batch_size = latents.shape[0]
        device = latents.device
        dtype = prompt_embeds.dtype
        model = self.transformer
        if pooled_prompt_embeds is None:
            proj_dim = getattr(getattr(model, "config", None), "pooled_projection_dim", 768)
            pooled_prompt_embeds = torch.zeros(batch_size, int(proj_dim), device=device, dtype=dtype)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                batch_size,
                prompt_embeds.shape[1],
                device=device,
                dtype=torch.long,
            )
        timestep_1000 = self._prepare_training_timestep(sigma, batch_size, device) * 1000
        guidance = torch.tensor([guidance_scale], device=device, dtype=dtype)

        with self._build_training_autocast_ctx(device):
            return model(
                hidden_states=latents.to(dtype),
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep_1000,
                guidance=guidance,
                return_dict=False,
            )[0]

    def load(self) -> None:
        """Load model components with the standard training_only protocol.

        ``training_only=True`` (the default for ``TrainActor``-built bundles
        in direct-sampling mode) keeps VAE + text_encoder off the GPU at
        init time; they are lazily reloaded by
        ``ModelBundle.load_aux_components`` from inside
        ``FSDPSamplingEngine.bind_model``, AFTER the transformer has been
        FSDP-wrapped. This is the same protocol used by
        ``HunyuanVeido1p5ModelBundle`` / ``WANModelBundle`` /
        ``SD3ModelBundle`` / ``FluxModelBundle`` and is what keeps per-GPU
        static weight footprint to just the transformer (~26 GB bf16) so
        FSDP shard buffers + Adam states + reward weights all fit on the
        actor's GPU.
        """
        if self.training_only:
            logger.info("Loading HunyuanVideo transformer only (training_only mode)...")
        else:
            logger.info("Loading HunyuanVideo model bundle...")

        self._load_transformer()

        if self.use_lora:
            self._add_lora_adapters()

        if not self.training_only:
            self._load_vae()
            self._load_text_encoder()

        logger.info("HunyuanVideo model bundle loaded")

    def _load_transformer(self) -> None:
        """Load the transformer model.

        Honors ``skip_device_move`` (CPU-offload / FSDP2 cpu_offload mode keeps
        the transformer on CPU and lets FSDP2 manage placement) and aligns
        param dtype after ``from_pretrained`` (diffusers leaves a few buffers
        in fp32 even when ``torch_dtype`` is set; FSDP2 ``_init_mp_dtypes``
        asserts a uniform dtype across the wrapped module).

        Failures here used to be caught by a broad ``except Exception`` that
        only logged a warning and silently set ``self._transformer = None``,
        which surfaced downstream as the misleading ``"Transformer not
        loaded. Call load() first."`` from ``ModelBundle.transformer``. We
        now log with full traceback (``logger.exception``) AND re-raise, so
        the actor crashes with the real exception (e.g. CUDA OOM, HF Hub
        404) instead of leaking a half-initialized bundle into the FSDP
        backend.
        """
        try:
            from diffusers.models.transformers.transformer_hunyuan_video import (
                HunyuanVideoTransformer3DModel,
            )
        except ImportError as exc:
            # flash_attn_2_cuda.so can fail to load due to GLIBC version
            # mismatch on older container images (e.g. "GLIBC_2.32 not found").
            # This poisons the entire diffusers import chain because the
            # transformer module transitively touches flash-attention.  When
            # this happens, block flash_attn in sys.modules so that diffusers
            # falls back to SDPA attention, then retry the import once.
            if "flash_attn" in str(exc) or "GLIBC" in str(exc):
                import sys

                logger.warning(
                    "flash_attn import failed (%s); blocking flash_attn in sys.modules "
                    "so diffusers falls back to SDPA attention and retrying import.",
                    exc,
                )
                # Remove any partially-loaded flash_attn submodules and block
                # future imports by inserting None sentinels.
                fa_keys = [k for k in sys.modules if k == "flash_attn" or k.startswith("flash_attn.")]
                for k in fa_keys:
                    del sys.modules[k]
                sys.modules["flash_attn"] = None  # type: ignore[assignment]
                try:
                    from diffusers.models.transformers.transformer_hunyuan_video import (
                        HunyuanVideoTransformer3DModel,
                    )
                except ImportError as retry_exc:
                    logger.warning(
                        "Still could not import HunyuanVideoTransformer3DModel after blocking flash_attn (%s).",
                        retry_exc,
                    )
                    self._transformer = None
                    return
            else:
                logger.warning(
                    "Could not import HunyuanVideoTransformer3DModel from diffusers (%s). "
                    "Please install diffusers with HunyuanVideo support.",
                    exc,
                )
                self._transformer = None
                return

        try:
            self._transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                self.pretrained_path,
                subfolder="transformer",
                torch_dtype=self.dtype,
            )
            if self.skip_device_move:
                first_param_device = next(self._transformer.parameters()).device
                if str(first_param_device) != "cpu":
                    self._transformer.to("cpu")
            else:
                self._transformer.to(self.device, dtype=self.dtype)
            logger.info("Loaded HunyuanVideo transformer from %s", self.pretrained_path)
        except Exception:
            self._transformer = None
            logger.exception(
                "Failed to load HunyuanVideo transformer from %s (training_only=%s, "
                "skip_device_move=%s, dtype=%s, device=%s)",
                self.pretrained_path,
                self.training_only,
                self.skip_device_move,
                self.dtype,
                self.device,
            )
            raise

    def _add_lora_adapters(self) -> None:
        """Inject LoRA adapters into the HunyuanVideo-1.0 transformer.

        Mirrors the pattern used by ``WANModelBundle._add_lora_adapters`` /
        ``HunyuanVeido1p5ModelBundle._add_lora_adapters``: wrap the
        transformer with ``peft.get_peft_model``, attach a second ``"old"``
        adapter for the NFT dual-adapter mechanism, and re-cast trainable
        params back to ``self.dtype`` (peft drops them to fp32 by default).
        """
        if self._transformer is None:
            logger.warning("Cannot add LoRA: HunyuanVideo transformer not loaded")
            return

        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            logger.warning("peft not installed, skipping LoRA. Install with: pip install peft")
            return

        target_modules = self.lora_target_modules
        if target_modules is None:
            target_modules = type(self).default_lora_target_modules()
            if target_modules is None:
                raise RuntimeError(
                    "HunyuanVideo LoRA requested but lora_target_modules is unresolved. "
                    "Pass --training.lora-target-modules or ensure "
                    f"{type(self).__name__}.default_lora_target_modules() returns a list."
                )

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )

        self._transformer = get_peft_model(self._transformer, lora_config)
        self._transformer.add_adapter("old", lora_config)
        self._transformer.set_adapter("default")

        for param in self._transformer.parameters():
            if param.requires_grad and param.dtype != self.dtype:
                param.data = param.data.to(self.dtype)

        logger.info(
            "LoRA adapters added to HunyuanVideo (rank=%s, alpha=%s)",
            self.lora_rank,
            self.lora_alpha,
        )

    def _load_vae(self) -> None:
        """Load the VAE model.

        Aux-component loader: failures keep ``self._vae = None`` so that
        ``ModelBundle.load_aux_components`` can transparently retry on the
        next sampler bind, but we now log with full traceback
        (``logger.exception``) instead of the previous warning-with-no-stack
        so the actual exception (HF Hub error, OOM, dtype mismatch, ...)
        is recoverable from job logs.
        """
        try:
            from diffusers import AutoencoderKLHunyuanVideo
        except ImportError:
            logger.warning("Could not import AutoencoderKLHunyuanVideo from diffusers.")
            self._vae = None
            return

        try:
            self._vae = AutoencoderKLHunyuanVideo.from_pretrained(
                self.vae_ckpt_path,
                subfolder="vae",
                torch_dtype=self.vae_dtype,
            )
            self._vae.to(self.device)
            self._vae.eval()
            self._vae.requires_grad_(False)
            logger.info("Loaded HunyuanVideo VAE from %s", self.vae_ckpt_path)
        except Exception:
            self._vae = None
            logger.exception("Could not load HunyuanVideo VAE from %s", self.vae_ckpt_path)

    def _load_text_encoder(self) -> None:
        """Load the LLaMA + CLIP dual text encoder.

        Same lazy-retry semantics as ``_load_vae`` (failure → ``None`` so
        ``load_aux_components`` can re-attempt on next bind), with
        ``logger.exception`` stacktrace logging so the underlying cause is
        visible.
        """
        try:
            self._text_encoder = HunyuanVideoTextEncoderWrapper(
                pretrained_path=self.text_encoder_ckpt_path,
                device=self.device,
                dtype=self.text_encoder_dtype,
            )
            logger.info("Loaded HunyuanVideo text encoder from %s", self.text_encoder_ckpt_path)
        except Exception:
            self._text_encoder = None
            logger.exception(
                "Could not load HunyuanVideo text encoder from %s",
                self.text_encoder_ckpt_path,
            )

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode prompts using the dual text encoder, aligned with diffusers'
        ``HunyuanVideoPipeline.encode_prompt``.

        Args:
            prompt: Text prompt(s)
            negative_prompt: Optional negative prompt(s) — handled at the
                bundle level (``encode_inputs``), not by the wrapper.

        Returns:
            Tuple of ``(prompt_embeds, pooled_prompt_embeds,
            prompt_attention_mask)``. The third element is the **post-crop**
            LLaMA attention mask required by the HunyuanVideo transformer's
            ``encoder_attention_mask`` input — without it the transformer
            sees system-header padding tokens as valid prompt content.
        """
        if self._text_encoder is None:
            self._raise_aux_component_not_loaded("text encoder")

        # Handle string input
        if isinstance(prompt, str):
            prompt = [prompt]

        return self._text_encoder.encode_prompt(prompt, negative_prompt)

    def _encode_prompt_to_input_dict(
        self,
        prompts: Union[str, List[str]],
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Map the 3-tuple from :meth:`encode_prompt` into named fields.

        Override is required because the upstream tuple's third element is a
        **prompt attention mask** (LLaMA, post-crop), not the SD3/FLUX
        ``text_ids`` that the base class default would assume.

        The returned dict keys (``prompt_embeds`` / ``pooled_prompt_embeds`` /
        ``encoder_attention_mask``) match the HunyuanVideo FSDP sampler's
        :py:meth:`sample` parameter names and the
        :class:`HunyuanVideoForwardContext` field names — so the mask is
        threaded through the sampler / forward context / forward-denoiser
        without any sampler-side change.
        """
        del kwargs
        prompts_list = [prompts] if isinstance(prompts, str) else list(prompts)
        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.encode_prompt(prompts_list)
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "encoder_attention_mask": prompt_attention_mask,
        }

    def decode_latents(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latents to video frames.

        Args:
            latents: Latent tensor [B, C, T, H, W]

        Returns:
            Video tensor [B, C, T, H', W']
        """
        if self._vae is None:
            self._raise_aux_component_not_loaded("VAE")

        with torch.no_grad():
            # Scale latents
            latents = latents / self._vae.config.scaling_factor

            # Decode
            video = self._vae.decode(latents).sample

        return video

    def encode_images(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode images/video to latents.

        Args:
            images: Video tensor [B, C, T, H, W]

        Returns:
            Latent tensor [B, C', T', H', W']
        """
        if self._vae is None:
            self._raise_aux_component_not_loaded("VAE")

        with torch.no_grad():
            # Encode
            latents = self._vae.encode(images).latent_dist.sample()

            # Scale
            latents = latents * self._vae.config.scaling_factor

        return latents

    def get_no_split_modules(self) -> Tuple[Type[nn.Module], ...]:
        """
        Get module types that should not be split in FSDP.

        For HunyuanVideo, we don't split the transformer blocks.
        """
        try:
            from diffusers.models.transformers.transformer_hunyuan_video import (
                HunyuanVideoSingleTransformerBlock,
                HunyuanVideoTransformerBlock,
            )

            return (HunyuanVideoTransformerBlock, HunyuanVideoSingleTransformerBlock)
        except ImportError:
            return ()


class HunyuanVideoTextEncoderWrapper:
    """
    Wrapper for HunyuanVideo's dual text encoder (LLAMA + CLIP).
    """

    def __init__(
        self,
        pretrained_path: str,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize text encoder wrapper.

        Args:
            pretrained_path: Path to pretrained weights
            device: Device to load on
            dtype: Data type
        """
        self.pretrained_path = pretrained_path
        self.device = device
        self.dtype = dtype

        self.llama_encoder = None
        self.clip_encoder = None
        self.tokenizer = None
        self.clip_tokenizer = None

        self._load_encoders()

    def _load_encoders(self) -> None:
        """Load the LLAMA and CLIP encoders.

        Uses ``AutoTokenizer`` for both tokenizer subfolders so the right
        class is auto-selected from each ``tokenizer_config.json``. The
        diffusers-converted HunyuanVideo 1.0 release ships a fast
        LLaVA-Llama-3 tokenizer (``tokenizer.json``, no SentencePiece
        ``tokenizer.model``) — using the slow ``LlamaTokenizer`` here
        therefore raises a ``TypeError: not a string`` from SentencePiece.
        """
        try:
            from transformers import AutoTokenizer, CLIPTextModel, LlamaModel

            self.llama_encoder = LlamaModel.from_pretrained(
                self.pretrained_path,
                subfolder="text_encoder",
                torch_dtype=self.dtype,
            )
            self.llama_encoder.to(self.device)
            self.llama_encoder.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_path,
                subfolder="tokenizer",
            )

            self.clip_encoder = CLIPTextModel.from_pretrained(
                self.pretrained_path,
                subfolder="text_encoder_2",
                torch_dtype=self.dtype,
            )
            self.clip_encoder.to(self.device)
            self.clip_encoder.eval()

            self.clip_tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_path,
                subfolder="tokenizer_2",
            )

        except Exception as e:
            logger.warning(f"Could not load text encoders: {e}")
            self.llama_encoder = None
            self.clip_encoder = None
            self.tokenizer = None
            self.clip_tokenizer = None

    def encode_prompt(
        self,
        prompt: List[str],
        negative_prompt: Optional[List[str]] = None,
        *,
        prompt_template: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 256,
        num_hidden_layers_to_skip: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode prompts to LLaMA + CLIP embeddings, aligned with upstream.

        Mirrors diffusers' ``HunyuanVideoPipeline._get_llama_prompt_embeds``
        and ``_get_clip_prompt_embeds`` so the produced ``prompt_embeds`` /
        ``prompt_attention_mask`` / ``pooled_prompt_embeds`` are drawn from
        the **same distribution the HunyuanVideo transformer was trained
        on**:

        * Wrap each prompt with the Llava system+user chat template.
        * Take ``hidden_states[-(num_hidden_layers_to_skip + 1)]`` (skip the
          top 2 LLaMA layers), not ``last_hidden_state``.
        * Crop the first ``crop_start`` tokens (system-header span) from
          both the embeddings and the attention mask.
        * Use the **real** attention mask (cropped), not an all-ones tensor.

        ``negative_prompt`` is accepted for signature stability with the
        bundle's call-site but is not consumed here — the negative branch is
        handled at the bundle level (``encode_inputs``) by re-invoking
        ``_encode_prompt_to_input_dict`` on the negative prompts.

        Returns:
            ``(prompt_embeds, pooled_prompt_embeds, prompt_attention_mask)``.
        """
        del negative_prompt
        if (
            self.llama_encoder is None
            or self.clip_encoder is None
            or self.tokenizer is None
            or self.clip_tokenizer is None
        ):
            raise RuntimeError(
                "HunyuanVideoTextEncoderWrapper is not initialized: LLAMA/CLIP encoders are unavailable."
            )
        if prompt_template is None:
            prompt_template = DEFAULT_PROMPT_TEMPLATE

        # ---- LLaMA branch (Llava-templated, layer-skipped, header-cropped) ----
        templated_prompts = [prompt_template["template"].format(p) for p in prompt]
        crop_start = prompt_template.get("crop_start", None)
        if crop_start is None:
            # Probe the tokenizer for the template's leading-header length so
            # we can slice it out post-encoding (matches upstream fallback at
            # ``pipeline_hunyuan_video.py:217-229``).
            probe = self.tokenizer(
                prompt_template["template"],
                padding="max_length",
                return_tensors="pt",
                return_length=False,
                return_overflowing_tokens=False,
                return_attention_mask=False,
            )
            # Subtract 2 for the trailing ``<|eot_id|>`` and the ``{}`` placeholder.
            crop_start = probe["input_ids"].shape[-1] - 2
        crop_start = int(crop_start)

        # Upstream tokenises with ``max_length = max_sequence_length + crop_start``
        # so that the post-crop sequence length is exactly ``max_sequence_length``.
        effective_max_length = max_sequence_length + crop_start
        text_inputs = self.tokenizer(
            templated_prompts,
            max_length=effective_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=True,
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        prompt_attention_mask = text_inputs.attention_mask.to(self.device)

        with torch.no_grad():
            llama_output = self.llama_encoder(
                input_ids=text_input_ids,
                attention_mask=prompt_attention_mask,
                output_hidden_states=True,
            )
            prompt_embeds = llama_output.hidden_states[-(num_hidden_layers_to_skip + 1)]
            prompt_embeds = prompt_embeds.to(dtype=self.dtype)

        if crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_attention_mask = prompt_attention_mask[:, crop_start:]

        # ---- CLIP branch (pooled, max_length=77) ----
        clip_inputs = self.clip_tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        clip_input_ids = clip_inputs.input_ids.to(self.device)

        with torch.no_grad():
            pooled_embeds = self.clip_encoder(
                clip_input_ids,
                output_hidden_states=False,
            ).pooler_output
            pooled_embeds = pooled_embeds.to(dtype=self.dtype)

        return prompt_embeds, pooled_embeds, prompt_attention_mask

    def to(self, device: Union[str, torch.device]) -> "HunyuanVideoTextEncoderWrapper":
        """Move encoders to device."""
        self.device = device
        if self.llama_encoder is not None:
            self.llama_encoder.to(device)
        if self.clip_encoder is not None:
            self.clip_encoder.to(device)
        return self
