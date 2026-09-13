"""Leo2 bundle -- bootstraps the external hymm stack and holds the weights; see README.md."""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from typing import Any

import torch
from torch import nn

from unirl.config.require import require
from unirl.models.types.bundle import Bundle

from .config import LEO2_CONFIG_RELPATH, Leo2PipelineConfig

_HYMM_BOOTSTRAPPED = False


def ensure_hy_parallel_state() -> bool:
    """Initialise hy_parallelism's global parallel state once per process; see README.md '## Gotchas'."""
    import torch.distributed as dist
    from hy_parallelism import parallel_states as hy_ps

    if hy_ps.is_parallel_state_initialized():
        return True
    if not dist.is_initialized():
        return False
    world = dist.get_world_size()
    dp_shard = min(8, world)
    hy_ps.init_parallel_state(
        dp_replicate=world // dp_shard,
        dp_shard=dp_shard,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        world_size=world,
    )
    return True


_HYMM_MODULES = ("hymm", "hy_parallelism", "index_kits")


def _resolve_hymm_paths(config: Leo2PipelineConfig) -> str:
    """Put hymm and its two deps on sys.path unless they already import; returns the hymm config yaml."""
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    if config.assets_base:
        os.environ.setdefault("ASSETS_BASE", config.assets_base)
    require(
        bool(os.environ.get("ASSETS_BASE")),
        "leo2: no hymm asset root -- set bundle.config.assets_base or export ASSETS_BASE.",
    )
    require(bool(config.ckpt_path), "leo2: no checkpoint -- set bundle.config.ckpt_path (recipe: $LEO2_CKPT_DIR).")
    require(
        bool(config.generation_config_path),
        "leo2: no generation config -- set bundle.config.generation_config_path (recipe: $LEO2_GENERATION_CONFIG).",
    )

    repo = config.hymm_repo_path
    # find_spec rather than an import: hymm.constants freezes the asset env vars at import time.
    if not all(importlib.util.find_spec(name) for name in _HYMM_MODULES):
        require(
            bool(repo),
            "leo2: hymm is not importable -- set bundle.config.hymm_repo_path (recipe: $HYMM_REPO_PATH) "
            "or put the repo and its deps/ on PYTHONPATH.",
        )
        for path in (repo, os.path.join(repo, "deps/hy_parallelism"), os.path.join(repo, "deps/IndexKits")):
            if path not in sys.path:
                sys.path.insert(0, path)

    config_yaml = config.config_yaml or (os.path.join(repo, LEO2_CONFIG_RELPATH) if repo else "")
    require(
        bool(config_yaml),
        "leo2: no hymm config yaml -- set bundle.config.config_yaml or hymm_repo_path to derive it.",
    )
    return config_yaml


def _bootstrap_hymm(config: Leo2PipelineConfig):
    """Import hymm, parse args from the yaml, set the per-process globals; idempotent."""
    global _HYMM_BOOTSTRAPPED

    config_yaml = _resolve_hymm_paths(config)

    import argparse

    from hymm.config import add_core_args, validate_args
    from hymm.core import global_vars
    from hymm.core.arguments import parse_argv_from_yaml

    if _HYMM_BOOTSTRAPPED:
        return global_vars.get_args()

    argv = [
        "--config-path",
        config_yaml,
        "--ckpt",
        config.ckpt_path,
        "--task-id",
        "unirl-leo2",
        "--framework",
        "fsdp",
        *config.extra_hymm_args,
    ]
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config-path", type=str, required=True)
    pre.add_argument("--framework", type=str, default="fsdp")
    known, remaining = pre.parse_known_args(argv)

    config_argv, frozen = parse_argv_from_yaml(known.config_path, allow_frozen=True, argv_overrides=remaining)
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0]] + config_argv + remaining
        parser = argparse.ArgumentParser(description="Leo2 UniRL bundle")
        parser = add_core_args(parser)
        args = parser.parse_args()
    finally:
        sys.argv = saved_argv
    args = validate_args(args, frozen)
    if not args.model_structure.endswith("HF"):
        args.model_structure += "HF"

    # per-process globals (Ray worker == fresh process, but stay idempotent)
    global_vars._GLOBAL_ARGS = None
    global_vars.set_args(args)

    from loguru import logger as _loguru_logger

    global_vars._GLOBAL_LOGGER = None
    global_vars.set_logger(_loguru_logger)

    from hymm.core.parallel_states import ParallelState

    ParallelState(dp_rank=0, dp_size=1)

    # The bundle may be built before torch.distributed is up (then this is a
    # no-op); predict_noise() re-checks right before the first forward.
    ensure_hy_parallel_state()

    _HYMM_BOOTSTRAPPED = True
    return args


def _dcp_load_into(model: nn.Module, weights_dir: str) -> None:
    """Fill the (empty) model from the torch-dcp checkpoint, dtype-exact."""
    import pickle

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    md = pickle.load(open(os.path.join(weights_dir, ".metadata"), "rb"))
    saved = md.state_dict_metadata

    dest = model.state_dict()
    # checkpoint keys are 'model.<k>'; nest one level so DCP fqns line up.
    wanted = {}
    missing_in_ckpt = []
    for k, v in dest.items():
        ck = f"model.{k}"
        if ck in saved:
            meta = saved[ck]
            props = getattr(meta, "properties", None)
            sdt = getattr(props, "dtype", None) if props is not None else None
            if isinstance(v, torch.Tensor) and sdt is not None and v.dtype != sdt:
                wanted[k] = torch.empty(tuple(meta.size), dtype=sdt)
            else:
                wanted[k] = v
        else:
            missing_in_ckpt.append(k)
    if missing_in_ckpt:
        print(
            f"[leo2 bundle] {len(missing_in_ckpt)} state keys not in ckpt (kept as built): {missing_in_ckpt[:8]}",
            flush=True,
        )

    dcp.load({"model": wanted}, storage_reader=FileSystemReader(weights_dir))
    result = model.load_state_dict(wanted, strict=False, assign=True)
    if result.unexpected_keys:
        raise RuntimeError(f"leo2 bundle: unexpected keys on load: {result.unexpected_keys[:8]}")
    print(
        f"[leo2 bundle] loaded {len(wanted)} tensors from {weights_dir}; missing(from ckpt)={len(missing_in_ckpt)}",
        flush=True,
    )


_BLOCK_CLASSES = ("LeoLayer", "LeoDualLayer", "LeoTripleLayer")


def _move_non_block_to_device(model: nn.Module, device) -> None:
    """Move every root-level child that does not contain a DiT block to device."""
    block_roots = set()
    for name, mod in model.named_modules():
        if type(mod).__name__ in _BLOCK_CLASSES:
            block_roots.add(name.split(".")[0])
    moved = 0
    for name, child in model.named_children():
        if name in block_roots:
            continue
        child.to(device)
        moved += sum(p.numel() for p in child.parameters())
    for pname, p in list(model._parameters.items()):
        if p is not None:
            model._parameters[pname] = nn.Parameter(p.to(device), requires_grad=p.requires_grad)
    for bname, b in list(model._buffers.items()):
        if b is not None:
            model._buffers[bname] = b.to(device)
    print(
        f"[leo2 bundle] moved non-block root modules to {device}: {moved / 1e9:.3f}B params; "
        f"block roots kept for FSDP: {sorted(block_roots)}",
        flush=True,
    )


def _patch_router_dtype(model: nn.Module) -> None:
    """Make every MoE router follow its input dtype; see README.md '## Gotchas'."""
    import torch.nn.functional as F

    n = 0
    for name, mod in model.named_modules():
        if name.endswith(".gate.wg") and isinstance(mod, nn.Linear):

            def _fwd(x, _m=mod):
                w = _m.weight
                b = _m.bias
                return F.linear(x, w.to(x.dtype), None if b is None else b.to(x.dtype))

            mod.forward = _fwd
            n += 1
    print(f"[leo2 bundle] router dtype-follow patch applied to {n} gate.wg modules", flush=True)


class Leo2Bundle(Bundle):
    """Loaded Leo2 components."""

    def __init__(
        self,
        *,
        model: nn.Module,
        hymm_args: Any,
        dtype: torch.dtype,
        device: torch.device,
        config: Leo2PipelineConfig,
    ) -> None:
        super().__init__()
        self.model = model
        self.hymm_args = hymm_args
        self.dtype = dtype
        self.device = device
        self.config = config
        self.pretrained_path = config.ckpt_path
        self.use_system_prompt = "li-dit-encode-visual-qwen-3.5"
        if "--use-system-prompt" in config.extra_hymm_args:
            i = config.extra_hymm_args.index("--use-system-prompt")
            self.use_system_prompt = config.extra_hymm_args[i + 1]

    @classmethod
    def from_config(cls, config: Leo2PipelineConfig) -> "Leo2Bundle":
        args = _bootstrap_hymm(config)

        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RAY_LOCAL_RANK", 0)))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank % max(1, torch.cuda.device_count()))
        device = (
            torch.device(config.device)
            if config.device
            else (
                torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            )
        )

        from hymm.models import build_model

        dtype = torch.bfloat16 if config.model_precision == "bf16" else torch.float32
        model, _model_config = build_model(args, dtype=dtype, device="cpu", initialize_weights=False)

        if not config.skip_load_ckpt:
            _dcp_load_into(model, config.ckpt_path)
        model.requires_grad_(False)
        model.eval()
        if config.uniform_bf16:
            model.to(dtype=torch.bfloat16)
            _patch_router_dtype(model)
        # FSDPBackend shards and moves only the block classes; the rest must already be on
        # the device or the first forward hits a cpu/cuda mismatch.
        _move_non_block_to_device(model, device)

        # tokenizer + frozen aux models + the hymm pipeline object
        from hymm.core.extra_model_provider import build_text_encoder, build_tkwrapper, build_vae

        model.tokenizer = build_tkwrapper()
        if getattr(args, "use_vae", False):
            vae = build_vae(dp_rank=0, only_encoder=False)
            model.model_dict["vae"] = vae
        text_encoder = build_text_encoder()
        model.model_dict["text_encoder"] = text_encoder
        try:
            text_encoder.to("cpu" if config.text_encoder_gpu_transient else device)
        except Exception:  # noqa: BLE001 -- placement is an optimisation; the encode ctx retries
            pass

        model.load_generation_config(config.generation_config_path)
        model.build_diffusion_pipeline()

        return cls(model=model, hymm_args=args, dtype=dtype, device=device, config=config)

    def trainable_module(self) -> nn.Module:
        return self.model

    @contextlib.contextmanager
    def text_encoder_ctx(self):
        """Transiently host the text encoder on GPU for an encode pass."""
        te = self.model.model_dict.get("text_encoder")
        moved = False
        if te is not None and self.config.text_encoder_gpu_transient:
            try:
                te.to(self.device)
                moved = True
            except Exception:
                moved = False
        try:
            yield
        finally:
            if moved:
                te.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


__all__ = ["Leo2Bundle"]
