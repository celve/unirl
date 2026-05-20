"""Shared SDE runtime entrypoints.

Three layers, all owned by this module:

1. **Pure math** — :func:`get_sigma_schedule` for the FlowMatch σ schedule.
   Static branch implemented here (diffusers' static path has issue #13243);
   dynamic branch delegates to diffusers (its dynamic path is bug-free).
   :func:`calculate_dynamic_mu` derives μ from image_seq_len.

2. **Schedule policy** — :class:`FlowMatchSchedulePolicy` is the *static*
   data the σ computation needs from a model: shift, the 5 dynamic-shift
   knobs, vae_scale_factor and patch_size. :meth:`from_pretrained` reads
   the three diffusers-standard JSONs (``scheduler/scheduler_config.json``,
   ``transformer/config.json``, ``vae/config.json``) under a model
   checkpoint directory and assembles a policy. The loader is **pure I/O
   on small JSONs** — no model weights, no Bundle, no Pipeline. Available
   main-side regardless of whether the actor loaded a full Bundle, which
   is what lets sglang / vllm-omni engines compute σ without holding the
   model in memory.

3. **Glue** — :func:`compute_flowmatch_sigma` applies a policy to the
   per-request (T, H, W) triple and produces the σ tensor.
   :func:`ensure_req_sigmas` pins the result onto ``RolloutReq.sigmas`` if
   not already set (every rollout engine calls it at the top of its
   ``generate``).

Ownership map (kept explicit so reading the code doesn't require
following six getattr chains)::

    Policy        owned by  MODEL CHECKPOINT (scheduler/transformer/vae JSONs)
    Params (T,H,W) owned by REQUEST (RolloutReq.stage_params["diffusion"])
    σ computation owned by THIS MODULE (pure function)
    σ flow        carried by RolloutReq.sigmas (set by engine, read by
                  pipeline / worker / replay; verified end-to-end by
                  diffusionrl.rollout.engine.sigma_verify)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ===========================================================================
# Layer 1 — pure math
# ===========================================================================


def _flowmatch_static_shift(shift: float, t: torch.Tensor) -> torch.Tensor:
    """SD3-paper time shift, applied exactly once to ``t`` in ``[0, 1]``::

        t' = (shift * t) / (1 + (shift - 1) * t)

    Why we own this instead of delegating to diffusers: the upstream
    ``FlowMatchEulerDiscreteScheduler`` applies this same formula in its
    ``use_dynamic_shifting=False`` branch but a confirmed bug (issue
    #13243) applies it twice. Dynamic branch is unaffected — see
    :func:`_flowmatch_dynamic_shift_via_diffusers`.
    """
    return (shift * t) / (1 + (shift - 1) * t)


def _flowmatch_dynamic_shift_via_diffusers(
    mu: float,
    num_steps: int,
    *,
    time_shift_type: str = "exponential",
    num_train_timesteps: int = 1000,
) -> torch.Tensor:
    """Dynamic-shift σ schedule, delegated to diffusers.

    Returns a ``[num_steps + 1]`` tensor matching diffusers'
    ``FlowMatchEulerDiscreteScheduler`` output when configured with
    ``use_dynamic_shifting=True`` and ``mu`` from the request's
    image_seq_len. Grid differs from our static Style A grid — callers
    asking for dynamic are asking for diffusers' reference behavior.

    Base sigmas are explicitly ``np.linspace(1.0, 1/num_steps, num_steps)``
    — the kickoff every upstream diffusers FlowMatch dynamic-shift
    pipeline uses (``QwenImagePipeline.__call__``, ``FluxPipeline.__call__``
    and friends all pass exactly this). Without passing them, diffusers'
    ``set_timesteps`` falls back to ``linspace(sigma_max, sigma_min, T)``
    with ``sigma_min ≈ 1/num_train_timesteps = 0.001`` — drifts the small-σ
    tail by ~0.13 at T=12 / Qwen-Image's μ. That's the σ the model was
    trained against; the fallback isn't.
    """
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=num_train_timesteps,
        use_dynamic_shifting=True,
        time_shift_type=time_shift_type,
    )
    base_sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
    scheduler.set_timesteps(num_inference_steps=num_steps, sigmas=base_sigmas, mu=mu)
    return scheduler.sigmas


def get_sigma_schedule(
    num_steps: int,
    shift: float = 3.0,
    device: Optional[torch.device] = None,
    *,
    mu: Optional[float] = None,
    time_shift_type: str = "exponential",
) -> torch.Tensor:
    """Compute the FlowMatch σ schedule of length ``num_steps + 1``.

    ``mu is None`` → static branch (own implementation). ``mu is not None``
    → dynamic branch (diffusers delegation). The "should I be dynamic?"
    decision belongs upstream — this is the math primitive.
    """
    if mu is None:
        t = torch.linspace(1.0, 0.0, num_steps + 1)
        sigmas = _flowmatch_static_shift(shift, t)
    else:
        sigmas = _flowmatch_dynamic_shift_via_diffusers(
            mu=mu,
            num_steps=num_steps,
            time_shift_type=time_shift_type,
        )
    if device is not None:
        sigmas = sigmas.to(device)
    return sigmas


def calculate_dynamic_mu(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Linear interpolation of dynamic-shift μ from image sequence length.

    Mirrors diffusers' ``calculate_shift`` used by SD3 / Flux pipelines.
    Feed into :func:`get_sigma_schedule` via ``mu=...``.
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


# ===========================================================================
# Layer 2 — schedule policy (static data owned by the model checkpoint)
# ===========================================================================


def _read_json(path: Path) -> Optional[dict]:
    """Read a JSON file; return ``None`` on any failure (missing / unreadable)."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, IsADirectoryError, json.JSONDecodeError, OSError):
        return None


def _vae_scale_factor_from_block_out_channels(block_out_channels: Any) -> Optional[int]:
    """Derive ``vae_scale_factor`` from ``block_out_channels`` length.

    Mirrors diffusers' convention
    (``2 ** (len(vae.config.block_out_channels) - 1)``, see
    ``pipeline_stable_diffusion_3.py:219`` and ``pipeline_flux.py:209``).
    Returns ``None`` for malformed inputs.
    """
    try:
        n = len(block_out_channels)
        if n < 1:
            return None
        return 2 ** (n - 1)
    except TypeError:
        return None


def _normalize_patch_size(value: Any, default: int) -> int:
    """Coerce a raw ``patch_size`` config value to a single spatial int.

    Some video transformers declare ``patch_size`` as a 3D
    ``[t_patch, h_patch, w_patch]`` list (e.g. diffusers'
    ``WanTransformer3DModel`` ships ``[1, 2, 2]``); the
    dynamic-shifting math here only consumes the spatial patch
    (``image_seq_len = (H // ... // patch_size) * (W // ... // patch_size)``,
    which assumes ``h_patch == w_patch``). Scalar inputs pass through
    unchanged; list/tuple inputs take the last element (the W patch),
    matching the canonical ``h == w`` convention. ``None`` falls back to
    ``default``.

    Without this normalization a checkpoint with list-valued
    ``patch_size`` would raise ``TypeError: int() argument must be a
    string, a bytes-like object or a real number, not 'list'`` at
    :meth:`FlowMatchSchedulePolicy.from_pretrained` time — even for
    static-only policies that never read the field at sample time.
    """
    if value is None:
        return int(default)
    if isinstance(value, (list, tuple)):
        if not value:
            return int(default)
        return int(value[-1])
    return int(value)


@dataclass
class FlowMatchSchedulePolicy:
    """The static σ recipe for a model. Loaded once per actor.

    Built either from a pretrained checkpoint directory
    (:meth:`from_pretrained`) or from explicit fields
    (:meth:`static_only`). The policy is **pure data** — pickleable,
    pass-by-value across Ray IPC, no Bundle required to construct it.

    Field semantics
    ---------------
    ``shift``: static FlowMatch time-shift. Per-model defaults: SD3=3.0,
    Flux=1.0, Wan=5.0, HunyuanVideo=1.0, HunyuanImage3=3.0. Always wins
    over ``scheduler_config.shift`` (user-configured override).

    ``use_dynamic_shifting``, ``base_shift``, ``max_shift``,
    ``base_image_seq_len``, ``max_image_seq_len``, ``time_shift_type``:
    dynamic-shift block. Sourced from
    ``<pretrained>/scheduler/scheduler_config.json``. When
    ``use_dynamic_shifting=True``, :func:`compute_flowmatch_sigma`
    derives μ from image_seq_len and delegates to diffusers' dynamic
    branch; otherwise these fields are ignored.

    ``vae_scale_factor``, ``patch_size``: latent-grid divisors used in
    image_seq_len = ``(H // vae_scale_factor // patch_size) * (W // ...)``.
    Sourced from ``<pretrained>/vae/config.json`` and
    ``<pretrained>/transformer/config.json``. Only used in dynamic
    branch.
    """

    shift: float = 3.0
    use_dynamic_shifting: bool = False
    base_shift: float = 0.5
    max_shift: float = 1.15
    base_image_seq_len: int = 256
    max_image_seq_len: int = 4096
    time_shift_type: str = "exponential"
    vae_scale_factor: int = 8
    patch_size: int = 2

    @classmethod
    def static_only(cls, shift: float) -> "FlowMatchSchedulePolicy":
        """Build a static-shift-only policy. Use when no pretrained dir is
        available (tests, ad-hoc smoke runs)."""
        return cls(shift=float(shift), use_dynamic_shifting=False)

    @classmethod
    def _dynamic_from_overrides(
        cls,
        shift: float,
        overrides: Optional[Dict[str, Any]],
        path: Any,
    ) -> "FlowMatchSchedulePolicy":
        """Construct a dynamic-shift policy from an explicit overrides dict.

        Helper for :meth:`from_pretrained` when ``require_dynamic=True``
        and the pretrained checkpoint isn't locally readable (e.g. HF
        repo ID like ``Qwen/Qwen-Image``). The model's Pipeline is
        responsible for passing its canonical dynamic-shift fields in
        ``overrides``; if it didn't, raise loudly so the σ contract
        bug surfaces at engine init instead of at first rollout.
        """
        if not overrides:
            raise RuntimeError(
                f"FlowMatchSchedulePolicy.from_pretrained: caller declared "
                f"require_dynamic=True for path={path!r} but provided no "
                f"dynamic_overrides. The checkpoint isn't locally readable "
                f"so we can't load scheduler_config.json, and without "
                f"explicit dynamic fields we'd silently produce a static "
                f"policy (which mis-shifts dynamic-shift models like "
                f"Qwen-Image). Pre-download the scheduler/scheduler_config.json "
                f"from HF Hub OR have the model's Pipeline.build_schedule_policy "
                f"pass dynamic_overrides with use_dynamic_shifting=True + "
                f"base_shift / max_shift / base_image_seq_len / max_image_seq_len / "
                f"time_shift_type fields."
            )
        defaults = cls()
        return cls(
            shift=float(shift),
            use_dynamic_shifting=True,
            base_shift=float(overrides.get("base_shift", defaults.base_shift)),
            max_shift=float(overrides.get("max_shift", defaults.max_shift)),
            base_image_seq_len=int(overrides.get("base_image_seq_len", defaults.base_image_seq_len)),
            max_image_seq_len=int(overrides.get("max_image_seq_len", defaults.max_image_seq_len)),
            time_shift_type=str(overrides.get("time_shift_type", defaults.time_shift_type)),
            vae_scale_factor=int(overrides.get("vae_scale_factor", defaults.vae_scale_factor)),
            patch_size=_normalize_patch_size(overrides.get("patch_size"), defaults.patch_size),
        )

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path, None],
        *,
        shift: float,
        require_dynamic: bool = False,
        dynamic_overrides: Optional[Dict[str, Any]] = None,
    ) -> "FlowMatchSchedulePolicy":
        """Build a policy by reading the diffusers-standard JSON layout.

        Tries three files under ``path``::

            <path>/scheduler/scheduler_config.json   → dynamic-shift fields
            <path>/transformer/config.json           → patch_size
            <path>/vae/config.json                   → vae_scale_factor

        Missing files / missing keys fall back to dataclass defaults;
        the scheduler JSON specifically gets a ``logger.warning`` (it
        carries the dynamic-shift block, so silent fallback there
        would be a real bug for dynamic-shift models). The ``shift``
        arg always wins over any ``scheduler_config.shift`` (some
        checkpoints ship with stale shift values).

        Path resolution
        ---------------
        - ``path is None`` → :meth:`static_only` (explicit opt-in).
        - ``path`` doesn't exist locally:
            - If ``require_dynamic=False`` (default): fall back to
              :meth:`static_only` with a debug log. **Correct for
              static-shift HF repo IDs** like
              ``stabilityai/stable-diffusion-3.5-medium``.
            - If ``require_dynamic=True``: caller has declared this
              model NEEDS dynamic shifting (e.g. Qwen-Image). Use
              ``dynamic_overrides`` if provided; otherwise RAISE so the
              error surfaces at engine init instead of silently shipping
              wrong σ schedules.
        - ``path`` is an existing local directory → read JSONs.

        ``require_dynamic`` + ``dynamic_overrides`` were added to fix the
        silent fallback-to-static for HF-repo-ID checkpoints whose model
        config declared dynamic shifting (the 2026-05-18 review's Phase
        I.4). Each Pipeline's ``build_schedule_policy()`` knows its own
        dynamic-shift posture and passes the right hints.
        """
        if path is None:
            if require_dynamic:
                return cls._dynamic_from_overrides(shift, dynamic_overrides, path)
            return cls.static_only(shift)
        root = Path(path)
        if not root.exists():
            if require_dynamic:
                return cls._dynamic_from_overrides(shift, dynamic_overrides, path)
            logger.debug(
                "FlowMatchSchedulePolicy.from_pretrained: %s does not exist "
                "locally (likely an HF repo ID — bundle.from_pretrained will "
                "resolve it). Falling back to static_only(shift=%s).",
                root,
                shift,
            )
            return cls.static_only(shift)

        defaults = cls()  # canonical default values
        sched_path = root / "scheduler" / "scheduler_config.json"
        sched = _read_json(sched_path)
        if sched is None:
            # Dynamic-shift information lives in this JSON; silent
            # fallback to static would mis-shift a dynamic-shift model
            # (caught by ``verify_engine_used_sigmas`` at rollout time
            # but worth surfacing here so the cause is obvious in
            # logs).
            logger.warning(
                "FlowMatchSchedulePolicy.from_pretrained: %s not found; "
                "dynamic-shift fields default to static-only behavior. "
                "If the model wants dynamic shift, σ will drift and the "
                "drift assert will raise at the first rollout.",
                sched_path,
            )
            sched = {}
        trans = _read_json(root / "transformer" / "config.json") or {}
        vae = _read_json(root / "vae" / "config.json") or {}

        vae_scale_factor = _vae_scale_factor_from_block_out_channels(vae.get("block_out_channels"))
        return cls(
            shift=float(shift),
            use_dynamic_shifting=bool(sched.get("use_dynamic_shifting", defaults.use_dynamic_shifting)),
            base_shift=float(sched.get("base_shift", defaults.base_shift)),
            max_shift=float(sched.get("max_shift", defaults.max_shift)),
            base_image_seq_len=int(sched.get("base_image_seq_len", defaults.base_image_seq_len)),
            max_image_seq_len=int(sched.get("max_image_seq_len", defaults.max_image_seq_len)),
            time_shift_type=str(sched.get("time_shift_type", defaults.time_shift_type)),
            vae_scale_factor=int(vae_scale_factor or defaults.vae_scale_factor),
            patch_size=_normalize_patch_size(trans.get("patch_size"), defaults.patch_size),
        )


# ===========================================================================
# Layer 3 — apply policy to a request → σ tensor
# ===========================================================================


def compute_flowmatch_sigma(
    policy: FlowMatchSchedulePolicy,
    *,
    num_inference_steps: int,
    height: int,
    width: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Apply ``policy`` to the per-request ``(T, H, W)`` triple → σ tensor.

    Static branch: ``get_sigma_schedule(T, policy.shift)``.
    Dynamic branch: derive ``image_seq_len`` from
    ``(H // vae_scale_factor // patch_size) * (W // ... // ...)``,
    compute μ via :func:`calculate_dynamic_mu`, delegate to diffusers.

    Returns ``Tensor[T+1]``, dtype float32, range ``[0, 1]``.
    """
    if not policy.use_dynamic_shifting:
        return get_sigma_schedule(num_inference_steps, policy.shift, device)
    latent_h = int(height) // int(policy.vae_scale_factor)
    latent_w = int(width) // int(policy.vae_scale_factor)
    image_seq_len = (latent_h // int(policy.patch_size)) * (latent_w // int(policy.patch_size))
    mu = calculate_dynamic_mu(
        image_seq_len,
        base_seq_len=policy.base_image_seq_len,
        max_seq_len=policy.max_image_seq_len,
        base_shift=policy.base_shift,
        max_shift=policy.max_shift,
    )
    return get_sigma_schedule(
        num_inference_steps,
        policy.shift,
        device,
        mu=mu,
        time_shift_type=policy.time_shift_type,
    )


def ensure_req_sigmas(req: Any, policy: FlowMatchSchedulePolicy) -> None:
    """Pin σ schedule onto ``req.sigmas`` if not already set.

    Every rollout engine calls this at the top of ``generate(req)``.
    Idempotent: if ``req.sigmas`` is already set (e.g. a caller pre-pinned
    a custom schedule for testing), this is a no-op.

    ``req`` must expose ``req.sigmas`` (read/write) and
    ``req.stage_params["diffusion"]`` with ``num_inference_steps`` /
    ``height`` / ``width`` keys (duck-typed to avoid importing
    ``RolloutReq`` here — keeps this module free of types/* deps).

    All three keys are **required** —  silent ``height=1024`` /
    ``width=1024`` defaults would mis-derive μ for dynamic-shift models
    when the request actually rendered at a different resolution
    (e.g. WAN T2V at 480×832). Drivers (``NewRolloutPipeline.plan_requests``)
    always set all three; absence means a wiring bug.
    """
    if req.sigmas is not None:
        return
    diffusion = dict(req.stage_params.get("diffusion") or {})
    missing = [k for k in ("num_inference_steps", "height", "width") if k not in diffusion]
    if missing:
        raise ValueError(
            f"ensure_req_sigmas: req.stage_params['diffusion'] is missing "
            f"required key(s) {missing} for σ schedule computation. The "
            f"driver (NewRolloutPipeline.plan_requests / _build_diffusion_"
            f"stage_params) must set num_inference_steps / height / width."
        )
    req.sigmas = compute_flowmatch_sigma(
        policy,
        num_inference_steps=int(diffusion["num_inference_steps"]),
        height=int(diffusion["height"]),
        width=int(diffusion["width"]),
    )


__all__ = [
    "get_sigma_schedule",
    "calculate_dynamic_mu",
    "FlowMatchSchedulePolicy",
    "compute_flowmatch_sigma",
    "ensure_req_sigmas",
]
