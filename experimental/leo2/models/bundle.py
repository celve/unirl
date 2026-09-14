"""Leo2 bundle -- bootstraps the external hymm stack and holds the weights; see README.md."""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import time
from typing import Any, Dict, List

import torch
from torch import nn

from unirl.config.require import require
from unirl.models.types.bundle import Bundle

from .assets import export_local_asset_bases
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

    # Before any hymm import: hymm.constants freezes TEXT_ENCODER_BASE / VAE_BASE at import time.
    export_local_asset_bases(text_encoder_base=config.text_encoder_base, vae_base=config.vae_base)

    repo = config.hymm_repo_path
    # find_spec rather than an import, for the same reason.
    specs = {name: importlib.util.find_spec(name) for name in _HYMM_MODULES}
    if not all(specs.values()):
        require(
            bool(repo),
            "leo2: hymm is not importable -- set bundle.config.hymm_repo_path (recipe: $HYMM_REPO_PATH) "
            "or put the repo and its deps/ on PYTHONPATH.",
        )
        for path in (repo, os.path.join(repo, "deps/hy_parallelism"), os.path.join(repo, "deps/IndexKits")):
            if path not in sys.path:
                sys.path.insert(0, path)
    elif not repo:
        # hymm came from PYTHONPATH; it is a namespace package, so origin is None and the
        # search location is the only way back to the repo root the yaml sits under.
        locations = list(specs["hymm"].submodule_search_locations or [])
        repo = os.path.dirname(locations[0]) if locations else ""

    config_yaml = config.config_yaml or (os.path.join(repo, LEO2_CONFIG_RELPATH) if repo else "")
    require(
        bool(config_yaml) and os.path.isfile(config_yaml),
        f"leo2: hymm config yaml not found at {config_yaml!r} -- set bundle.config.config_yaml "
        "(recipe: $LEO2_HYMM_CONFIG).",
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


def _cast_params_dtype_no_copy(model: nn.Module, dtype: torch.dtype) -> int:
    """Re-declare every floating tensor of another dtype as an EMPTY one of ``dtype``; returns the count."""
    n = 0
    for module in model.modules():
        for name, p in list(module._parameters.items()):
            if p is not None and p.is_floating_point() and p.dtype != dtype:
                module._parameters[name] = nn.Parameter(torch.empty_like(p, dtype=dtype), requires_grad=p.requires_grad)
                n += 1
        for name, b in list(module._buffers.items()):
            if b is not None and b.is_floating_point() and b.dtype != dtype:
                module._buffers[name] = torch.empty_like(b, dtype=dtype)
                n += 1
    return n


def _dcp_load_sharded(model: nn.Module, weights_dir: str, *, key_prefix: str = "model.") -> Dict[str, Any]:
    """Fill an FSDP-wrapped model from a torch-DCP checkpoint, each rank reading only its shards."""
    import pickle

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict

    t0 = time.time()
    md = pickle.load(open(os.path.join(weights_dir, ".metadata"), "rb"))
    saved = md.state_dict_metadata
    live = get_model_state_dict(model, options=StateDictOptions(full_state_dict=False, cpu_offload=False))
    # peft exposes a base weight as ``<module>.base_layer.<leaf>`` and adds lora_A/lora_B
    # params the checkpoint lacks; map the former back, skip the latter.
    wanted: Dict[str, Any] = {}
    live_name: Dict[str, str] = {}
    skipped: List[str] = []
    lora_keys = 0
    for k, v in live.items():
        if ".lora_A." in k or ".lora_B." in k or ".lora_embedding_" in k:
            lora_keys += 1
            continue
        ck = k.replace(".base_layer.", ".")
        if f"{key_prefix}{ck}" in saved:
            wanted[ck] = v
            live_name[ck] = k
        else:
            skipped.append(k)
    unclaimed = [k for k in saved if k.startswith(key_prefix) and k[len(key_prefix) :] not in wanted]
    # Both counts are rank-invariant, so every rank raises together or none does.
    for names, what in ((skipped, "model keys absent from the checkpoint"), (unclaimed, "checkpoint keys unclaimed")):
        if names and len(names) > len(wanted) // 4:
            raise RuntimeError(
                f"[leo2 bundle] sharded DCP load: {len(names)} {what} against {len(wanted)} matched "
                f"in {weights_dir} (e.g. {names[:6]}); refusing to train on uninitialised weights"
            )
    dcp.load({key_prefix.rstrip("."): wanted}, storage_reader=FileSystemReader(weights_dir))
    # DTensor shards were filled in place; this covers plain tensors the planner copied.
    set_model_state_dict(
        model,
        {live_name[ck]: v for ck, v in wanted.items()},
        options=StateDictOptions(full_state_dict=False, strict=False),
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    nbytes = 0
    for v in wanted.values():
        local = v.to_local() if hasattr(v, "to_local") else v
        nbytes += local.numel() * local.element_size()
    print(
        f"[leo2 bundle] sharded DCP load: {len(wanted)} tensors, this rank {nbytes / 2**30:.1f} GB, "
        f"{lora_keys} LoRA keys left as initialised, {len(skipped)} not in ckpt, {len(unclaimed)} unclaimed, "
        f"{time.time() - t0:.1f}s",
        flush=True,
    )
    return {"loaded": len(wanted), "skipped": skipped, "unclaimed": unclaimed, "seconds": time.time() - t0}


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
        deferred = config.meta_init_transformer and not config.skip_load_ckpt
        captured_init = None
        if deferred:
            from accelerate import init_empty_weights

            from unirl.models.types.meta_init import capture_init_state

            t_build = time.time()
            # Parameters on meta so nn.Linear's reset_parameters costs nothing (the eager CPU
            # build of 75B took ~20 min per rank); buffers stay real on CPU to be captured.
            with init_empty_weights(include_buffers=False):
                model, _model_config = build_model(args, dtype=dtype, device="cpu", initialize_weights=False)
            captured_init = capture_init_state(model)
            require(
                not captured_init["buffers"] and not captured_init["attrs"],
                f"leo2: hymm's DiT now builds {len(captured_init['buffers'])} init-only buffer(s) and "
                f"{len(captured_init['attrs'])} tensor attr(s) that to_empty() destroys, and core "
                "restore_init_state silently skips owners it cannot resolve after the AC/LoRA wrap "
                "-- see the package README '## Gotchas'.",
            )
            print(f"[leo2 bundle] meta build in {time.time() - t_build:.1f}s", flush=True)
        else:
            model, _model_config = build_model(args, dtype=dtype, device="cpu", initialize_weights=False)

        if not config.skip_load_ckpt and not deferred:
            _dcp_load_into(model, config.ckpt_path)
        model.requires_grad_(False)
        model.eval()
        if config.uniform_bf16:
            if deferred:
                n_cast = _cast_params_dtype_no_copy(model, torch.bfloat16)
                print(f"[leo2 bundle] re-declared {n_cast} non-bf16 tensors as empty bf16", flush=True)
            else:
                model.to(dtype=torch.bfloat16)
            _patch_router_dtype(model)
        # FSDPBackend shards and moves only the block classes; the rest must already be on
        # the device or the first forward hits a cpu/cuda mismatch. Deferred: materialize() does it.
        if not deferred:
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

        bundle = cls(model=model, hymm_args=args, dtype=dtype, device=device, config=config)
        bundle._pending_dcp = config.ckpt_path if deferred else None
        bundle._meta_init_state = captured_init
        return bundle

    def trainable_module(self) -> nn.Module:
        return self.model

    def weights_pending(self) -> bool:
        """True while the DiT is still on meta, i.e. no backend has called materialize() yet."""
        return getattr(self, "_pending_dcp", None) is not None

    def materialize(self, *, device: Any = None, with_aux: tuple = ()) -> None:
        """Post-wrap weight load hook called by ``sharded_load.load_trainable_weights``; idempotent."""
        require(
            not with_aux,
            f"Leo2Bundle.materialize: with_aux={with_aux!r} is not supported -- the text encoder and "
            "VAE are eager modules in model_dict, outside the wrap.",
        )
        pending = getattr(self, "_pending_dcp", None)
        if not pending:
            return
        self._pending_dcp = None

        from unirl.models.types.meta_init import restore_init_state
        from unirl.models.types.post_materialize import apply_deferred_ops

        dev = torch.device(device) if device is not None else self.device
        t0 = time.time()
        if any(t.is_meta for t in self.model.parameters()) or any(t.is_meta for t in self.model.buffers()):
            self.model.to_empty(device=dev)
        n_restored = restore_init_state(self.model, getattr(self, "_meta_init_state", None))
        # Before the load, not after: it skips the LoRA keys, so they must already be reset.
        apply_deferred_ops(self.model)
        print(
            f"[leo2 bundle] materialised on {dev} (restored {n_restored} init tensors) in {time.time() - t0:.1f}s",
            flush=True,
        )
        _dcp_load_sharded(self.model, pending)

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
