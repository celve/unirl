"""BagelBundle — weights+params holder for BAGEL-7B-MoT (gen-only T2I).

Implements the empty :class:`Bundle` Protocol. Pure container of the modules
BAGEL ships with for text-to-image: one MoT transformer (``Bagel`` wrapping a
``Qwen2ForCausalLM`` whose ``Qwen2MoTDecoderLayer`` blocks hold both und and gen
experts) + one FLUX-style VAE + one tokenizer. The und ViT path is disabled
(``visual_und=False``) — T2I needs only the gen expert + VAE.

LoRA injection / FSDP wrap / autocast lifecycle are owned outside the bundle
(the train backend), so ``from_config`` only loads + freezes. The trainable
surface is ``model.language_model`` (the MoT, where the ``*_moe_gen`` experts
live); the FSDP block class is ``Qwen2MoTDecoderLayer``.

Construction mirrors flow_grpo's ``train_bagel.py`` setup so the vendored
``InterleaveInferencer`` / ``generate_image`` path the diffusion stage delegates
to behaves identically.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Sequence, Tuple

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from torch import nn

from unirl.models.types.bundle import Bundle
from unirl.utils.dtypes import parse_torch_dtype

from .config import BagelPipelineConfig
from .vendor.data.data_utils import add_special_tokens
from .vendor.data.transforms import ImageTransform
from .vendor.inferencer import InterleaveInferencer
from .vendor.modeling.autoencoder import load_ae
from .vendor.modeling.bagel import Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM
from .vendor.modeling.qwen2 import Qwen2Tokenizer

logger = logging.getLogger(__name__)

# FSDP wrap block class for the MoT decoder (recipe backend.block_class_names).
BAGEL_FSDP_BLOCK_CLASS = "Qwen2MoTDecoderLayer"


class BagelBundle(Bundle):
    """BAGEL-7B-MoT bundle: MoT transformer + FLUX VAE + tokenizer + inferencer."""

    def __init__(
        self,
        *,
        model: Any,
        vae: Any,
        tokenizer: Any,
        new_token_ids: dict,
        vae_transform: Any,
        vit_transform: Any,
        inferencer: Any,
        dtype: torch.dtype,
        vae_dtype: torch.dtype,
        device: torch.device,
        pretrained_path: str,
        latent_patch_size: int,
        latent_channels: int,
        latent_downsample: int,
    ) -> None:
        super().__init__()
        self.model = model
        # The trainable MoT (where the *_moe_gen experts live). Same object the
        # vendored generate_image / _forward_flow run on, so FSDP2 fully_shard
        # (in-place) on this reference shards the gen forward too. Named
        # ``transformer`` so recipes can set backend.trainable_attr: transformer.
        self.transformer = model.language_model
        self.vae = vae
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.inferencer = inferencer
        self.dtype = dtype
        self.vae_dtype = vae_dtype
        self.device = device
        self.pretrained_path = pretrained_path
        self.latent_patch_size = latent_patch_size
        self.latent_channels = latent_channels
        self.latent_downsample = latent_downsample

    @classmethod
    def from_config(cls, config: BagelPipelineConfig) -> "BagelBundle":
        """Load BAGEL-7B-MoT (gen-only) from a local checkpoint directory.

        Replicates flow_grpo/train_bagel.py:316-414 minus LoRA/optimizer (which
        the train backend owns). Loads the EMA weights via
        ``load_checkpoint_and_dispatch`` onto a single device; the FSDP wrap and
        LoRA injection run later in :class:`FSDPBackend`.

        Note: ``load_checkpoint_and_dispatch`` attaches accelerate device hooks.
        For the dedicated FSDP path (Phase 6) those may need removal via
        ``accelerate.hooks.remove_hook_from_module`` before ``fully_shard``; for
        the standalone bundle smoke they are harmless.
        """
        if config.meta_init_transformer:
            # VeOmniBackend lifecycle: build on meta, load weights post-shard.
            return cls.from_meta_config(config)

        device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)
        dtype = parse_torch_dtype(config.model_precision, field_name="model_precision")
        vae_raw = config.vae_dtype if config.vae_dtype is not None else config.model_precision
        vae_dtype = parse_torch_dtype(vae_raw, field_name="vae_dtype")

        model_dir = config.pretrained_model_ckpt_path

        llm_config = Qwen2Config.from_json_file(os.path.join(model_dir, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vae_model, vae_config = load_ae(local_path=os.path.join(model_dir, "ae.safetensors"))

        bagel_config = BagelConfig(
            visual_gen=True,
            visual_und=False,
            llm_config=llm_config,
            vit_config=None,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=config.latent_patch_size,
            max_latent_size=config.max_latent_size,
        )

        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            model = Bagel(language_model, None, bagel_config)

        # force_hooks=True attaches accelerate AlignDevicesHooks (matching
        # flow_grpo/train_bagel.py) so the vendored inferencer / generate_image
        # path — which builds packed index tensors on CPU and calls submodule
        # forwards directly — has its inputs auto-moved to the model device.
        # Phase 6 (UniRL FSDP) must remove these hooks before fully_shard
        # (accelerate.hooks.remove_hook_from_module(model, recurse=True)).
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(model_dir, "ema.safetensors"),
            device_map={"": str(device)},
            dtype=dtype,
            offload_buffers=False,
            force_hooks=True,
            offload_folder="/tmp/bagel_offload",
        ).eval()

        tokenizer = Qwen2Tokenizer.from_pretrained(model_dir)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        # Image transforms match flow_grpo (vae 512/256/8, vit 490/112/7). Only
        # used for image-conditioned paths; pure T2I never exercises them, but
        # the inferencer constructor requires both.
        vae_transform = ImageTransform(512, 256, 8)
        vit_transform = ImageTransform(490, 112, 7)

        vae_model = vae_model.to(device=device, dtype=vae_dtype).eval()
        vae_model.requires_grad_(False)
        # Freeze the whole MoT here; the backend re-enables only the LoRA (or
        # moe_gen) params it injects/unfreezes.
        model.requires_grad_(False)

        inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )

        return cls(
            model=model,
            vae=vae_model,
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            inferencer=inferencer,
            dtype=dtype,
            vae_dtype=vae_dtype,
            device=device,
            pretrained_path=model_dir,
            latent_patch_size=int(model.latent_patch_size),
            latent_channels=int(model.latent_channel),
            latent_downsample=int(model.latent_downsample),
        )

    def trainable_module(self) -> "torch.nn.Module":
        """Return the MoT transformer — the FSDP wrap target / trainable root.

        ``model.language_model`` holds the ``Qwen2MoTDecoderLayer`` blocks whose
        ``*_moe_gen`` experts are the only trained params (via LoRA). The gen
        heads (``vae2llm`` / ``time_embedder`` / ``llm2vae`` / ``latent_pos_embed``)
        sit on the parent ``Bagel`` module and stay frozen in the LoRA setup.
        """
        return self.transformer

    @classmethod
    def from_meta_config(cls, config: BagelPipelineConfig) -> "BagelBundle":
        """Build the bundle with the MoT on the **meta** device for VeOmni FSDP2.

        Mirrors :meth:`from_config` but (1) skips ``load_checkpoint_and_dispatch``
        — the whole ``Bagel`` composite stays on meta until the backend
        materializes it (``veomni_parallelize`` ``to_empty``s + shards
        ``language_model``; the backend then calls :meth:`materialize`) — and (2)
        stamps the post-materialize deferred ops the VeOmni lifecycle needs:

        * ``language_model.init_weights`` → no-op: VeOmni calls ``init_weights()``
          on the wrapped trainable after ``to_empty``; ``Qwen2ForCausalLM`` has none.
        * ``_restore_rope``: recompute the rotary ``inv_freq`` (a non-persistent
          buffer clobbered by ``to_empty`` and absent from the checkpoint).
        * ``_unshard_root``: gather VeOmni's root-sharded leftovers
          (``embed_tokens`` / ``lm_head`` / norm) that Bagel's direct submodule
          calls never all-gather on their own.

        The FLUX-style VAE is a separate frozen module (not a ``Bagel`` child) and
        is loaded eagerly here, exactly as in :meth:`from_config`. Template:
        :meth:`unirl.models.hunyuan_image3.bundle.HunyuanImage3Bundle.from_meta_config`.
        """
        from unirl.train.deferred import _stamp

        device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)
        dtype = parse_torch_dtype(config.model_precision, field_name="model_precision")
        vae_raw = config.vae_dtype if config.vae_dtype is not None else config.model_precision
        vae_dtype = parse_torch_dtype(vae_raw, field_name="vae_dtype")

        model_dir = config.pretrained_model_ckpt_path

        llm_config = Qwen2Config.from_json_file(os.path.join(model_dir, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        # VAE: separate frozen module, loaded eagerly (not a Bagel child; small vs
        # the 7B MoT). Built before the meta block so it keeps real weights.
        vae_model, vae_config = load_ae(local_path=os.path.join(model_dir, "ae.safetensors"))

        bagel_config = BagelConfig(
            visual_gen=True,
            visual_und=False,
            llm_config=llm_config,
            vit_config=None,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=config.latent_patch_size,
            max_latent_size=config.max_latent_size,
        )

        # Whole composite on meta — no weight allocation. The backend materializes
        # language_model (FSDP) + the frozen heads from ema.safetensors post-shard.
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            model = Bagel(language_model, None, bagel_config)
        model.eval()

        # VeOmni calls init_weights() on the wrapped trainable after to_empty;
        # Qwen2ForCausalLM has no such method — stamp a no-op (real weights load in
        # materialize). Mirrors HunyuanImage3Bundle.from_meta_config.
        language_model.init_weights = lambda: None  # type: ignore[method-assign]

        # Post-materialize deferred ops, stamped on the trainable the backend
        # resolves; drained by ``apply_deferred_ops`` after the weight load.
        _stamp(language_model, _restore_rope)
        _stamp(language_model, _unshard_root)

        # Surface any non-persistent buffer the checkpoint won't restore — for
        # Bagel this names ``...rotary_emb.inv_freq``, which _restore_rope covers.
        non_persistent = sorted(set(n for n, _ in language_model.named_buffers()) - set(language_model.state_dict()))
        if non_persistent:
            logger.info(
                "meta_init_transformer: %d non-persistent buffer(s) absent from the "
                "checkpoint (restored post-materialize by _restore_rope): %s",
                len(non_persistent),
                non_persistent[:8],
            )

        tokenizer = Qwen2Tokenizer.from_pretrained(model_dir)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        vae_transform = ImageTransform(512, 256, 8)
        vit_transform = ImageTransform(490, 112, 7)

        vae_model = vae_model.to(device=device, dtype=vae_dtype).eval()
        vae_model.requires_grad_(False)
        # Freeze the whole MoT here (still on meta); the backend re-enables only
        # the LoRA params it injects. requires_grad survives to_empty.
        model.requires_grad_(False)

        inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )

        return cls(
            model=model,
            vae=vae_model,
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            inferencer=inferencer,
            dtype=dtype,
            vae_dtype=vae_dtype,
            device=device,
            pretrained_path=model_dir,
            latent_patch_size=int(model.latent_patch_size),
            latent_channels=int(model.latent_channel),
            latent_downsample=int(model.latent_downsample),
        )

    def materialize(self, *, device: torch.device, with_aux: Sequence[str] = ()) -> None:
        """Load real weights post-shard for the VeOmni meta-init path.

        Called by the backend's ``load_trainable_weights`` (Pattern A) after
        ``veomni_parallelize`` has ``to_empty``-materialized + sharded the
        trainable (``language_model``). Loads ``ema.safetensors`` into the
        ``Bagel`` composite in one DCP ``set_model_state_dict`` — covering the
        FSDP-sharded MoT (DTensor) AND the frozen parent gen-heads (plain)
        together. Mirrors :meth:`HunyuanImage3Bundle.materialize`.

        ``with_aux`` is accepted for backend signature parity and ignored — the
        VAE is a separate frozen module loaded eagerly in :meth:`from_meta_config`,
        not a ``Bagel`` child.
        """
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        # Plan: (wrapper-level prefix, module). The MoT (FSDP-wrapped) first, then
        # each present gen-head. Prefixes double as ema.safetensors key namespaces.
        plan: List[Tuple[str, nn.Module]] = [("language_model", self.transformer)]
        for attr in ("time_embedder", "vae2llm", "llm2vae", "latent_pos_embed"):
            head = getattr(self.model, attr, None)
            if isinstance(head, nn.Module):
                plan.append((attr, head))

        # 1. Allocate per-rank storage for anything still on meta. language_model
        #    was materialized by veomni_parallelize (gate skips it); the frozen
        #    parent heads are plain meta modules → to_empty here.
        for _attr, module in plan:
            if _module_has_meta_param(module):
                module.to_empty(device=device)

        # 2. rank-0 reads the matching keys; keys keep the wrapper-level namespace
        #    so they match Bagel.named_parameters() (language_model.* + bare heads).
        if _current_rank() == 0:
            prefixes = tuple(attr for attr, _ in plan)
            sd = _collect_filtered_state_dict(self.pretrained_path, prefixes=prefixes)
            # peft.inject_adapter_in_model wraps the LoRA-target Linears, moving the
            # base weight to ``<module>.base_layer.weight``; the checkpoint still
            # uses the original key. Rename ckpt keys into the LoRA-wrapped
            # namespace (else strict=False silently drops them → base_layer meta).
            expected = {n for n, _ in self.model.named_parameters(remove_duplicate=False)}
            rename = {}
            for name in expected:
                if name.endswith((".base_layer.weight", ".base_layer.bias")):
                    ck = name.replace(".base_layer.", ".")
                    if ck in sd:
                        rename[ck] = name
            for old_k, new_k in rename.items():
                sd[new_k] = sd.pop(old_k)
            if rename:
                logger.info(
                    "BagelBundle.materialize: renamed %d ckpt keys into the LoRA base_layer namespace",
                    len(rename),
                )
        else:
            sd = {}

        # 3. One DCP load into the composite — DTensor-aware (sharded MoT) + plain
        #    (heads) in a single rank-0 broadcast. strict=False: inv_freq
        #    (non-persistent) and injected LoRA adapters are legitimately absent.
        set_model_state_dict(
            self.model,
            sd,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=False),
        )

        # 4. rank-0 sanity: every LoRA base_layer param is finite & off-meta.
        if _current_rank() == 0:
            checked, bad = 0, 0
            for name, p in self.model.named_parameters(remove_duplicate=False):
                if ".base_layer." not in name:
                    continue
                checked += 1
                if p.is_meta or not p.data.isfinite().all():
                    bad += 1
            if checked and bad:
                raise RuntimeError(
                    f"BagelBundle.materialize: {bad}/{checked} LoRA base_layer params are "
                    "meta/non-finite after load — the base_layer key rename failed."
                )
        del sd

    def build_text_context(self, text: str, gen_context: dict) -> dict:
        """Device-safe variant of ``InterleaveInferencer.update_context_text``.

        ``Bagel.prepare_prompts`` builds the packed text-id tensors on CPU and
        ``forward_cache_update_text`` feeds them straight into ``embed_tokens``.
        On the eager FSDP path accelerate's ``AlignDevicesHook`` auto-moves them;
        the meta/VeOmni path attaches no hooks, so move them onto the trainable's
        device here. A no-op on the eager path (tensors already on device) → both
        backends share this one path.
        """
        model = self.model
        generation_input, kv_lens, ropes = model.prepare_prompts(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            prompts=[text],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        device = torch.device(self.device)
        generation_input = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in generation_input.items()}
        gen_context["past_key_values"] = model.forward_cache_update_text(
            gen_context["past_key_values"], **generation_input
        )
        gen_context["kv_lens"] = kv_lens
        gen_context["ropes"] = ropes
        return gen_context


def _restore_rope(module: nn.Module) -> None:
    """Recompute the rotary ``inv_freq`` clobbered by VeOmni's ``to_empty``.

    ``Qwen2RotaryEmbedding.inv_freq`` is a non-persistent buffer (absent from
    ``ema.safetensors``); ``to_empty`` re-allocates it as uninitialized memory and
    ``forward`` reads it without recomputing (rope_type=="default"). The plain
    attrs needed to rebuild it (``config`` / ``rope_init_fn`` / ``rope_kwargs``)
    survive ``to_empty``. Re-runs the ctor's inv_freq init
    (``modeling_qwen2.py:107-109``) on the materialized device. Stamped on
    ``language_model``; drained by ``apply_deferred_ops`` after the weight load.
    """
    re = module.model.rotary_emb
    device = re.inv_freq.device
    inv_freq, attention_scaling = re.rope_init_fn(re.config, device, **re.rope_kwargs)
    re.register_buffer("inv_freq", inv_freq, persistent=False)
    re.original_inv_freq = re.inv_freq
    re.attention_scaling = attention_scaling


def _unshard_root(module: nn.Module) -> None:
    """All-gather the root FSDP2 group once, post-load, and pin it resident.

    VeOmni root-shards the trainable (``embed_tokens`` / ``lm_head`` / final norm)
    with auto-no-reshard, expecting the root ``forward`` to all-gather them. Bagel's
    gen path calls ``language_model.forward_inference`` / ``.model.embed_tokens`` /
    ``.lm_head`` directly, never ``language_model.__call__``, so that gather never
    auto-fires and the params stay sharded DTensors at access. One explicit
    ``unshard()`` gathers them; auto-no-reshard then keeps them resident for the
    whole rollout+replay (the reshard hook is bound to a root forward that never
    runs). Unshards only this (root) group — child block groups manage themselves.
    No-op outside FSDP2. Stamped on ``language_model``; drained after the load.
    """
    from torch.distributed.fsdp import FSDPModule

    if isinstance(module, FSDPModule):
        module.unshard()


def _module_has_meta_param(module: nn.Module) -> bool:
    """True if any parameter of ``module`` (recursing) is on the meta device.
    Gates the per-module ``to_empty`` in :meth:`BagelBundle.materialize`."""
    for p in module.parameters(recurse=True):
        if p.is_meta:
            return True
    return False


def _current_rank() -> int:
    """Current torch.distributed rank, or 0 if not initialized."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _collect_filtered_state_dict(
    pretrained_path: str,
    *,
    prefixes: Sequence[str],
    filename: str = "ema.safetensors",
) -> Dict[str, torch.Tensor]:
    """Read keys from BAGEL's single ``ema.safetensors`` whose top-level prefix
    matches one of ``prefixes`` (matched as ``prefix + "."``). Keys are returned
    at the wrapper-level namespace (no stripping) so they match ``Bagel``'s
    parameter names directly: ``language_model.*`` for the MoT, bare
    ``time_embedder.*`` / ``vae2llm.*`` / ``llm2vae.*`` / ``latent_pos_embed.*``
    for the gen heads."""
    from safetensors.torch import safe_open

    ckpt_path = os.path.join(pretrained_path, filename)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"BagelBundle.materialize: checkpoint not found: {ckpt_path!r}. HF repo IDs "
            "are not supported here — point pretrained_model_ckpt_path at a local download."
        )
    prefix_dots = tuple(p + "." for p in prefixes)
    out: Dict[str, torch.Tensor] = {}
    with safe_open(ckpt_path, framework="pt") as f:
        for key in f.keys():
            if key.startswith(prefix_dots):
                out[key] = f.get_tensor(key)
    return out


__all__ = ["BAGEL_FSDP_BLOCK_CLASS", "BagelBundle"]
