"""Driver-side rollout pipeline for the new ``RolloutReq`` / ``RolloutResp`` path.

Sibling of :class:`diffusionrl.rollout.pipeline.RolloutPipeline` (legacy types).
Composes the same five phases — load_prompts → plan_requests → exec_request →
aggregate → convert_training_data — for the new types contract. Driven by a
:class:`diffusionrl.ray.group.new_rollout.NewRolloutActorGroup` whose actors
expose ``run_rollout_pipeline(req: RolloutReq) → List[RolloutResp]``.

The trainer-facing ``TrainingBatch`` boundary is bridged via
:func:`diffusionrl.rollout.engine.types_compat.resp_to_samples`. The bridge
leaves ``forward_context`` as ``None``; ``convert_training_data`` here
synthesizes an empty ``ForwardContext()`` (batch_size 0) which passes
``TrainingBatch.validate`` (training_batch.py:289-291). The trainer-side
replay path (FSDP HI3, recent commit ``82d44d2``) populates the real
context before training; that wiring is out of scope for this pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import ray
import torch

from diffusionrl.rollout.engine.types_compat import resp_to_samples
from diffusionrl.rollout.request_builders import (
    build_eval_request_batch,
    load_prompt_batch_from_source,
)
from diffusionrl.sde.noise import (
    generate_shared_noise,
    mix_rollout_base_seed,
)
from diffusionrl.types.conditions.image import ImageLatentCondition
from diffusionrl.types.forward_context import ForwardContext
from diffusionrl.types.primitives import Images, Texts
from diffusionrl.types.prompts import Prompts
from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.response import RolloutResponse
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp
from diffusionrl.types.sample import MediaPreview
from diffusionrl.types.sampling import SamplingParams
from diffusionrl.types.training_batch import TrainingBatch

if TYPE_CHECKING:
    from diffusionrl.algorithms_new.rollout_control import GRPORolloutControl
    from diffusionrl.ray.group.new_rollout import NewRolloutActorGroup


def build_media_preview(
    response: RolloutResponse,
    *,
    max_items: int = 8,
) -> Optional[MediaPreview]:
    """Return a ``MediaPreview`` capped at *max_items* from an aggregated response.

    Primary path: actors build per-shard previews during ``attach_reward``
    and the result is carried on ``samples.media_preview``, so this driver
    helper simply slices that typed payload down to *max_items*.

    Fallback path: if ``samples.media_preview`` is unset (e.g. responses
    from legacy code paths that still carry raw ``decoded_images``), rebuild
    the preview from ``decoded_images`` via ``RolloutResponse.attach_media_preview``.

    Returns ``None`` when neither source has images.
    """
    limit = max(1, int(max_items))

    preview = response.samples.media_preview
    if isinstance(preview, MediaPreview):
        if preview.is_empty():
            return None
        return preview.slice(0, limit) if len(preview) > limit else preview

    response.attach_media_preview(max_items=limit)
    return response.samples.media_preview


def _build_diffusion_stage_params(
    *,
    sampling_params: SamplingParams,
    samples_per_prompt: int,
    sde_indices: Optional[List[int]],
) -> Dict[str, Any]:
    """Pack sampling-side knobs into ``stage_params['diffusion']`` for vllm-omni.

    This dict is what the rollout-side translator reads to build the
    worker-bound ``OmniDiffusionSamplingParams``. Driver does NO semantic
    re-interpretation: ``eta`` and ``sde_indices`` are passed through
    untouched. The two axes are independent and resolved separately on
    the worker:

    - ``eta`` → ``OmniDiffusionSamplingParams.eta``; consumed by our
      ``FlowMatchSDEDiscreteScheduler`` only when an SDE step actually
      runs. Our scheduler is always installed (so ``segment.latents``
      stays non-empty for the clean-latents replay path); when no SDE
      step fires, the scheduler degenerates to pure Euler ODE and
      ``eta`` is dormant. See ``RL*Pipeline._ensure_scheduler_for_eta``.

    - ``sde_indices`` → ``extra_args["sde_indices"]`` →
      ``FlowMatchSDEDiscreteScheduler._sde_indices_set``:
        · ``None`` (forward-process algorithms — NFT et al.): translator
          OMITS the key entirely, scheduler reads ``None``, no step
          runs SDE, no log-probs captured. ``segment.latents`` still
          dense.
        · non-empty subset of ``range(num_inference_steps)``: those
          steps run SDE + capture log_probs, the rest run Euler ODE.

    Per-sample x_T (``initial_noise``) does NOT live in this dict. It
    rides ``RolloutReq.request_conditions["initial_latents"]`` (CONCAT
    field), which is sliced correctly under multi-actor sharding by
    ``RolloutReq.select``. ``stage_params`` is a ``shared_field`` and
    would broadcast the full batch tensor to every shard, mis-slicing
    on per-shard ``request_id`` (Omni renumbers from 0 per shard).
    """
    # Passthrough all SamplingParams scalar / value fields. Each
    # receiving Pipeline's Params dataclass filters unknown kwargs at
    # construction time (e.g. SD3Pipeline.generate / WAN21Pipeline.generate
    # both use ``{k: v for k in dataclasses.fields(<ParamsCls>)}``). This
    # avoids the previous image-only short-circuit where ``num_frames``
    # / ``init_same_noise`` / ``noise_group_ids`` were silently dropped,
    # and any future per-Stage Params field is supported without changing
    # this builder.
    #
    # Explicitly excluded: ``sde_strategy`` (Spec class instance, not a
    # cross-process value), ``sampler_kwargs`` (already-merged escape
    # hatch routed via the engine itself, not the stage params).
    #
    # **Two-key compat for the per-prompt sample count**: ``samples_per_prompt``
    # is the field name on the v2 DiffusionParams dataclasses (SD3 /
    # WAN21 / WAN22); ``num_samples_per_prompt`` is the legacy key
    # consumed by ``samplers/sglang/engine.py:327`` and
    # ``rollout/pipeline.py:396``. Emitting both lets v2 Pipelines pull
    # ``samples_per_prompt`` through their ``dataclasses.fields()``
    # filter while the legacy engines keep reading the old key. Once
    # legacy goes away the ``num_samples_per_prompt`` mirror can be
    # dropped.
    #
    # **``noise_group_ids`` is intentionally NOT emitted here** — even
    # though it would let the per-sample groups reach the Stage. The
    # reason is ``stage_params`` rides ``RolloutReq``'s ``shared_field``
    # slot, and ``RolloutReq.slice`` / ``select`` does NOT slice shared
    # fields. Multi-actor sharding (``rollout/plan.py``) and actor-side
    # chunking (``rollout/engine/__init__.py``) both call
    # ``req.slice(...)``, so every shard / chunk would see the
    # full-batch ``noise_group_ids`` while its own batch is the chunk
    # subset, triggering
    # ``generate_shared_noise``'s ``len(noise_group_ids) == batch_size``
    # hard assert on the first chunk.
    #
    # **Consequence**: ``init_same_noise=True`` is currently NOT
    # supported on the new path. The Stage will hit
    # ``generate_latents``'s ``init_same_noise=True`` branch ->
    # ``assert noise_group_ids is not None`` -> ``AssertionError`` with
    # a clear message ("generate_latents requires both base_seed and
    # noise_group_ids when init_same_noise=True"). This is fail-fast by
    # design: a silently working single-actor case + crashing multi-
    # actor case is worse than a uniform fail.
    #
    # **The clean cross-process design** routes per-sample noise via
    # ``request_conditions["initial_latents"]`` (CONCAT field,
    # auto-sliced under ``select`` and chunk). Closing that loop
    # requires (a) ``TrainsideEngineConfig`` exposing a ``modality``
    # field so ``compute_initial_noise_for_request`` knows the latent
    # shape, and (b) WAN21 / WAN22 / SD3 / HI3 Pipelines reading
    # ``req.request_conditions.get("initial_latents")`` and skipping
    # their own ``generate_latents`` when present. That's a
    # framework-wide change affecting all v2 Pipelines and is a
    # separate follow-up PR.
    out: Dict[str, Any] = {
        "height": int(sampling_params.height),
        "width": int(sampling_params.width),
        "num_inference_steps": int(sampling_params.num_inference_steps),
        "guidance_scale": float(sampling_params.guidance_scale),
        "num_frames": int(sampling_params.num_frames),
        "seed": int(sampling_params.seed),
        "init_same_noise": bool(sampling_params.init_same_noise),
        "eta": float(sampling_params.sde_config.eta),
        # Pass ``sde_indices`` through unchanged; translator consumes None
        # as "no SDE step" via ``.get()``. We could omit the key when None,
        # but keeping it makes the wire schema fixed and easier to audit.
        "sde_indices": list(sde_indices) if sde_indices is not None else None,
        # Two-key compat: legacy + v2 (see comment above).
        "num_samples_per_prompt": int(samples_per_prompt),
        "samples_per_prompt": int(samples_per_prompt),
        # Numerical policy passthrough (Stage's Params filter what they
        # actually consume; e.g. legacy SamplingParams has these, future
        # vllm-omni Omni-side ignores).
        "autocast_precision": str(sampling_params.autocast_precision),
        "trajectory_precision": str(sampling_params.trajectory_precision),
        "logprob_precision": str(sampling_params.logprob_precision),
    }
    return out


_SUPPORTED_MEDIA_REF_ROLES: Set[Tuple[str, str]] = {("image", "condition")}


def _reject_unsupported_media_refs(batch: Dict[str, Any], *, context: str) -> None:
    """Fail loud when a dataset hands unsupported media_refs to the NEW driver.

    PR #100's ``media_refs`` channel (``Prompts.media_refs`` + the
    ``MediaRef(uri, modality, role)`` URI list) was originally OLD-only.
    The NEW driver now consumes the ``(image, condition)`` (modality,
    role) pair via :class:`MultimodalRLDataSource._load_condition_images`
    → :attr:`Prompts.images` → ``RolloutReq.primitives['image']: Images``;
    all other (modality, role) combinations are still untyped on NEW and
    would be silently dropped (degrading I2V/V2V/text-conditioned jobs
    into a misconfigured run).

    Supported set: see :data:`_SUPPORTED_MEDIA_REF_ROLES`. Anything else
    raises ``NotImplementedError`` with a per-prompt index of the first
    offending entry so debugging is straightforward.
    """
    refs = batch.get("media_refs")
    if not refs:
        return
    if not isinstance(refs, list):
        raise TypeError(
            f"{context}: media_refs must be a list of per-prompt MediaRef lists, got {type(refs).__name__}."
        )
    bad: List[Tuple[int, Any]] = []
    for i, per_prompt in enumerate(refs or []):
        for r in per_prompt or []:
            modality = getattr(r, "modality", None)
            role = getattr(r, "role", None)
            if (modality, role) not in _SUPPORTED_MEDIA_REF_ROLES:
                bad.append((i, r))
    if not bad:
        return
    raise NotImplementedError(
        f"{context}: media_refs include {len(bad)} unsupported (modality, role) "
        f"entries; the NEW driver currently consumes only (image, condition). "
        f"First bad entry: prompt={bad[0][0]}, ref={bad[0][1]!r}."
    )


def _build_images_primitive(
    images: List[Any],
    *,
    prompt_count: int,
    context: str,
) -> Optional[Images]:
    """Bundle a per-prompt list of ``Optional[Image]`` into a typed
    ``Images`` primitive, or return ``None`` when the batch is purely
    text-only.

    A heterogeneous batch (some prompts carry a conditioning image,
    some don't) raises ``ValueError``: WAN I2V's transformer
    ``in_channels`` jump from 16→36 is a batch-level switch; mixing
    T2V- and I2V-shaped samples in one request would require a per-
    sample channel-concat path we don't currently model.
    """
    if not images:
        return None
    if len(images) != prompt_count:
        raise ValueError(f"{context}: prompts.images length {len(images)} != prompt count {prompt_count}")
    populated = [img for img in images if img is not None]
    if not populated:
        return None
    if len(populated) != len(images):
        missing = [i for i, img in enumerate(images) if img is None]
        raise ValueError(
            f"{context}: heterogeneous I2V batch — {len(missing)}/{len(images)} prompts "
            f"are missing a condition image (e.g. prompt index {missing[0]}). "
            f"Split into separate requests so each batch is either fully T2V or "
            f"fully I2V; per-sample channel-concat is not supported."
        )
    return Images.from_list(populated)


def compute_initial_noise_for_request(
    *,
    cfg: Any,
    prompts: Prompts,
    sampling_spec: SamplingParams,
    samples_per_prompt: int,
    rollout_id: int,
) -> Optional[torch.Tensor]:
    """Pre-compute the per-sample x_T tensor on the driver.

    Resolves the latent shape via the Pipeline class's ``latent_shape``
    classmethod (see :class:`diffusionrl.models_new.types.pipeline
    .LatentShapeProvider`). Each NEW model Pipeline (SD3, Qwen-Image,
    HunyuanVideo-1.5, WAN21, WAN22) declares its per-sample latent
    geometry. Pipelines that don't implement it (e.g. HI3 t2i where AR
    token shape is per-prompt variable) fall back to the rollout
    engine's own RNG — explicit opt-out via ``NotImplementedError``
    rather than the old string-dispatch silent fall-through.

    Previously routed by ``cfg.rollout.engine.modality`` string,
    hard-coded for 3 models (sd35_t2i / wan21_t2v / wan22_t2v). That
    dispatcher silently returned ``None`` for ANY trainside config
    (``rollout.engine: {}`` has no ``modality`` field) AND for any
    model not in the literal triple (Qwen-Image / HV15 / future
    pipelines). The engine RNG fallback breaks GRPO group noise,
    resume determinism, and rollout/replay consistency.

    Per-sample diversity comes from ``noise_group_ids`` on :class:`Prompts`
    + the mixed rollout seed (see :func:`mix_rollout_base_seed`).
    """
    import hydra.utils  # local import — driver-side dispatch only

    # Resolve the Pipeline class from cfg.model._target_ (which points at
    # ``<...>.<Pipeline>.from_config``). Strip the ``.from_config`` suffix
    # to get the class dotpath.
    target = cfg.model._target_
    if not isinstance(target, str) or "." not in target:
        return None  # ill-formed cfg.model — engine RNG fallback
    pipeline_target = target.rsplit(".", 1)[0]
    try:
        pipeline_cls = hydra.utils.get_class(pipeline_target)
    except Exception:
        return None  # class not resolvable — engine RNG fallback
    if not hasattr(pipeline_cls, "latent_shape"):
        return None  # Pipeline opted out — engine RNG fallback
    try:
        latent_shape: Tuple[int, ...] = pipeline_cls.latent_shape(model_config=cfg.model, sampling_spec=sampling_spec)
    except NotImplementedError:
        # Explicit opt-out (e.g. HI3 AR where per-prompt token-shape
        # variability makes driver-side noise pre-computation ill-defined).
        return None

    base_seed = mix_rollout_base_seed(int(sampling_spec.seed), int(rollout_id))
    batch_size = len(prompts.sample_ids)
    if batch_size == 0:
        return None

    # Always go through ``generate_shared_noise`` so the noise is
    # deterministic on (base_seed, group_id) and reproducible across
    # restarts / multi-actor shards / re-runs. ``Prompts.expand`` already
    # produces the right shape of ``noise_group_ids`` for both modes:
    #
    # - ``init_same_noise=True``  → group ids are coarse (e.g. ['a','a','b','b'])
    #                                so samples within a group share noise.
    # - ``init_same_noise=False`` → group ids are per-sample unique
    #                                (e.g. ['prompt:a:sample:0', ...])
    #                                so each sample gets its own deterministic
    #                                draw — same noise across runs given the
    #                                same (seed, rollout_id, group_id) tuple.
    #
    # ``generate_latents`` would otherwise fall back to a non-seeded
    # ``torch.randn`` for ``init_same_noise=False``, which breaks resume
    # determinism. Driver-side we stay on CPU — Ray ships the tensor to
    # the rollout actor and the pipeline subclass moves it to the worker's
    # device before ``prepare_latents`` returns.
    group_ids = list(prompts.noise_group_ids) if prompts.noise_group_ids else [f"{sid}" for sid in prompts.sample_ids]
    return generate_shared_noise(
        batch_size=batch_size,
        latent_shape=latent_shape,
        device=torch.device("cpu"),
        dtype=torch.float32,
        noise_group_ids=group_ids,
        base_seed=base_seed,
    )


class NewRolloutPipeline:
    """Phase-decomposed driver-side rollout pipeline for the new-types path."""

    # ------------------------------------------------------------------
    # Phase sub-methods (override these to customize a single stage)
    # ------------------------------------------------------------------

    def load_prompts(
        self,
        *,
        data_source: Any,
        prompt_batch_size: int,
        samples_per_prompt: int,
        init_same_noise: bool = False,
    ) -> Prompts:
        """Fetch a raw prompt batch and expand to ``Prompts``.

        Identical to :meth:`RolloutPipeline.load_prompts` — the ``Prompts``
        type is shared across legacy and new paths and provides the only
        sample-aligned view we need (``prompts``, ``sample_ids``,
        ``group_ids``).
        """
        prompt_batch = load_prompt_batch_from_source(
            data_source=data_source,
            prompt_batch_size=int(prompt_batch_size),
        )
        raw_prompts = list(prompt_batch.get("prompts") or [])
        if not raw_prompts:
            raise RuntimeError("Data source returned an empty prompt batch.")
        _reject_unsupported_media_refs(prompt_batch, context="NewRolloutPipeline.load_prompts")
        raw_prompt_ids = prompt_batch.get("prompt_ids")
        prompt_ids = (
            [str(pid) for pid in raw_prompt_ids]
            if isinstance(raw_prompt_ids, list) and len(raw_prompt_ids) == len(raw_prompts)
            else None
        )
        raw_images = prompt_batch.get("images")
        images = list(raw_images) if isinstance(raw_images, list) and len(raw_images) == len(raw_prompts) else None
        prompts = Prompts.from_unique_prompts(raw_prompts, prompt_ids=prompt_ids, images=images)
        if int(samples_per_prompt) > 1:
            prompts = prompts.expand(
                int(samples_per_prompt),
                init_same_noise=bool(init_same_noise),
            )
        return prompts

    def plan_requests(
        self,
        *,
        prompts: Prompts,
        sampling_spec: SamplingParams,
        samples_per_prompt: int,
        control_algorithm: Any,
        rollout_id: int,
        collect_media_preview: bool = False,
        media_max_items: int = 8,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> Tuple[RolloutReq, Optional[Set[int]]]:
        """Build a typed ``RolloutReq`` and return ``(req, sde_indices_set)``.

        Translation logic mirrors
        :func:`diffusionrl.rollout.engine.types_compat.request_to_req`. The
        ``collect_media_preview`` / ``media_max_items`` knobs travel via
        ``stage_params`` (the new contract has no top-level fields for them);
        the actor-side mixin reads them from there.

        ``initial_latents`` (optional) — caller-supplied precomputed x_T noise
        (or VAE-encoded init image for it2i). When set, packed as
        ``request_conditions['initial_latents'] = ImageLatentCondition(...)``
        so the engine forwards it verbatim instead of drawing its own.
        Shape must match the per-sample latent shape stacked along dim 0
        (``[B, ...]`` where ``B == len(prompts.sample_ids)``).
        """
        sde_indices = control_algorithm.resolve_rollout_sde_indices(
            current_step=int(rollout_id),
        )
        sde_indices_list = list(sde_indices) if sde_indices is not None else None
        per_rollout_seed = mix_rollout_base_seed(int(sampling_spec.seed), int(rollout_id))

        # SamplingParams for downstream pack-into-stage_params; we reuse
        # dataclasses.replace to keep eta/sde_config/etc unchanged.
        import dataclasses

        sampling_params = dataclasses.replace(
            sampling_spec,
            num_samples_per_prompt=int(samples_per_prompt),
            sde_indices=sde_indices_list,
            seed=per_rollout_seed,
        )

        # Validate the caller-supplied initial noise tensor shape vs the
        # prompt batch so a misalignment surfaces here, not as an opaque
        # mis-slice inside the vllm-omni worker.
        if initial_latents is not None and int(initial_latents.shape[0]) != len(prompts.sample_ids):
            raise ValueError(
                f"plan_requests: initial_latents.shape[0]={int(initial_latents.shape[0])} "
                f"!= len(prompts.sample_ids)={len(prompts.sample_ids)}"
            )

        diffusion_params = _build_diffusion_stage_params(
            sampling_params=sampling_params,
            samples_per_prompt=int(samples_per_prompt),
            sde_indices=sde_indices_list,
        )

        # ``request_conditions["initial_latents"]`` is the canonical x_T
        # hand-off across all rollout engines (SGLang, vllm-omni, FSDP-side
        # direct sampling). It's a ``CONCAT`` field on ``RolloutReq``, so
        # ``RolloutReq.select`` slices it correctly under multi-actor
        # sharding. The vllm-omni request translator
        # (``rollout/engine/vllm_omni/request.py``) reads
        # ``req.request_conditions["initial_latents"].latents`` and re-routes
        # it into ``OmniDiffusionSamplingParams.extra_args["initial_noise_batch"]``
        # for the worker pipeline to consume per-request.
        request_conditions: Dict[str, Any] = {}
        if initial_latents is not None:
            request_conditions["initial_latents"] = ImageLatentCondition(latents=initial_latents)

        primitives: Dict[str, Any] = {"text": Texts(texts=list(prompts.prompts))}
        images_prim = _build_images_primitive(
            prompts.images, prompt_count=len(prompts.prompts), context="plan_requests"
        )
        if images_prim is not None:
            primitives["image"] = images_prim

        req = RolloutReq(
            sample_ids=list(prompts.sample_ids),
            group_ids=list(prompts.group_ids),
            primitives=primitives,
            request_conditions=request_conditions,
            stage_params={
                "diffusion": diffusion_params,
                "collect_media_preview": bool(collect_media_preview),
                "media_max_items": max(1, int(media_max_items)),
            },
        )
        sde_indices_set = set(int(i) for i in sde_indices_list) if sde_indices_list is not None else None
        return req, sde_indices_set

    def exec_request(
        self,
        *,
        req: RolloutReq,
        rollout_group: "NewRolloutActorGroup",
        samples_per_prompt: int,
    ) -> List[RolloutResp]:
        """Dispatch to actor(s) using the fused ``run_rollout_pipeline`` actor method.

        Single-actor fast path uses a direct ``ray.get`` on the one handle.
        Multi-actor path shards the request at group boundaries via
        ``rollout_group.rollout_plan.shard_grouped_req`` and scatter-gathers
        across all actors. Shards cover whole groups so per-actor advantage
        normalization (with ``use_global_std=True``) sees complete group
        populations; std is therefore scoped to one actor's shard in the
        multi-actor case.
        """
        if rollout_group.num_actors == 0:
            raise RuntimeError("NewRolloutPipeline.exec_request: no rollout actors.")
        if rollout_group.num_actors == 1:
            return ray.get(rollout_group.get_actors()[0].run_rollout_pipeline.remote(req))

        shards = rollout_group.rollout_plan.shard_grouped_req(
            req,
            num_actors=rollout_group.num_actors,
            samples_per_prompt=int(samples_per_prompt),
        )
        nested = rollout_group.scatter_gather("run_rollout_pipeline", shards)
        return [response for sub in nested for response in sub]

    def aggregate(
        self,
        *,
        responses: List[RolloutResp],
    ) -> RolloutResp:
        """Concat per-actor RolloutResps into a single merged response.

        Delegates to :meth:`RolloutResp.concat` (which performs the
        ``Segment.sample_indices`` offset remap). Caps the merged
        ``media_preview`` to ``stage_params['media_max_items']`` if the
        original request asked for previews.
        """
        if not responses:
            raise RuntimeError("Rollout produced no responses.")
        combined = RolloutResp.concat(responses)
        # Media-preview cap: the originating per-actor responses each cap
        # at media_max_items, but the merged container can hold more
        # since per-shard previews are extended. Cap once more here.
        # We don't carry the cap on the resp itself; the driver already
        # knows it from plan_requests so we accept a default of 8 if the
        # caller doesn't override via run_once.
        return combined

    def convert_training_data(
        self,
        *,
        combined_response: RolloutResp,
        request: RolloutReq,
        sampling_params: SamplingParams,
        sde_indices: Optional[Set[int]],
        control_algorithm: Optional["GRPORolloutControl"] = None,
    ) -> Tuple[TrainingBatch, int]:
        """Convert the merged ``RolloutResp`` into a validated ``TrainingBatch``.

        Bridges via :func:`resp_to_samples` for the segment-and-decode
        translation, then overrides ``advantages`` / ``rewards`` /
        ``component_rewards`` / ``forward_context`` from the merged response
        and a stub ``ForwardContext()`` (batch_size 0) so
        ``RolloutResponse.to_training_batch`` accepts the build. The trainer-
        side replay path is responsible for filling the real
        ``forward_context`` before training; see this module's docstring.
        """
        # Build a synthesized legacy request that carries the per-sample
        # prompts (the reward pipeline already attached rewards on the
        # merged response, but to_training_batch reads
        # ``request.prompts.{prompt_ids,sample_ids,group_ids,...}``).
        text_prim = request.primitives.get("text")
        if text_prim is None or not getattr(text_prim, "texts", None):
            raise RuntimeError("NewRolloutPipeline.convert_training_data: req.primitives['text'] missing.")
        sids = list(combined_response.sample_ids)
        gids = list(combined_response.group_ids)
        # Resp may have been reordered by per-group split→concat; rebuild
        # text alignment by walking the request's own sample_ids → prompts
        # map.
        request_sids = list(request.sample_ids)
        request_texts = list(text_prim.texts)
        if len(request_sids) != len(request_texts):
            raise RuntimeError(
                f"NewRolloutPipeline.convert_training_data: request sample_ids "
                f"count ({len(request_sids)}) != texts count ({len(request_texts)})."
            )
        sid_to_text = {sid: request_texts[i] for i, sid in enumerate(request_sids)}
        try:
            ordered_texts = [sid_to_text[sid] for sid in sids]
        except KeyError as missing:
            raise RuntimeError(
                f"NewRolloutPipeline.convert_training_data: response sample_id {missing!r} not present in request."
            ) from None

        prompt_ids = [str(sid) for sid in sids]
        prompts_obj = Prompts(
            prompts=ordered_texts,
            prompt_ids=prompt_ids,
            sample_ids=sids,
            group_ids=gids,
            noise_group_ids=list(prompt_ids),
            prompt_metadata=[{} for _ in sids],
        )
        legacy_request = RolloutRequest(
            prompts=prompts_obj,
            sampling_params=sampling_params,
            collect_media_preview=False,
            media_max_items=8,
        )
        legacy_samples = resp_to_samples(combined_response, request=legacy_request)
        # Override fields that resp_to_samples leaves blank: rewards,
        # advantages, component_rewards, forward_context. The trainer-side
        # replay populates real forward_context before training.
        legacy_samples.rewards = combined_response.rewards
        legacy_samples.advantages = combined_response.advantages
        legacy_samples.component_rewards = combined_response.component_rewards
        legacy_samples.forward_context = ForwardContext()  # batch_size 0 stub

        legacy_response = RolloutResponse(request=legacy_request, samples=legacy_samples)

        if sde_indices is not None and control_algorithm is not None:
            filtered = control_algorithm.get_filtered_training_indices(
                sde_indices,
                int(legacy_samples.timesteps.shape[0]),
            )
            if filtered:
                sde_indices = filtered

        training_batch = legacy_response.to_training_batch(sde_indices=sde_indices)
        return training_batch, int(training_batch.batch_size)

    # ------------------------------------------------------------------
    # All-in-one entrypoint composing the sub-methods
    # ------------------------------------------------------------------

    def run_once(
        self,
        *,
        rollout_group: "NewRolloutActorGroup",
        data_source: Any,
        prompt_batch_size: int,
        samples_per_prompt: int,
        sampling_spec: SamplingParams,
        control_algorithm: Any,
        rollout_id: int,
        collect_media_preview: bool = False,
        media_max_items: int = 8,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> Tuple[TrainingBatch, int, RolloutResp]:
        """Execute one full driver-side rollout step.

        Composition:
            load_prompts → plan_requests → exec_request → aggregate →
            convert_training_data

        Returns ``(training_batch, sample_count, combined_response)``. When
        ``collect_media_preview=True``, the aggregated response carries a
        capped preview on ``combined.media_preview``.

        ``initial_latents`` (optional) — forwarded to :meth:`plan_requests`
        and ultimately to the engine via
        ``req.request_conditions['initial_latents']``.
        """
        prompts = self.load_prompts(
            data_source=data_source,
            prompt_batch_size=prompt_batch_size,
            samples_per_prompt=samples_per_prompt,
            init_same_noise=bool(getattr(sampling_spec, "init_same_noise", False)),
        )
        req, sde_indices = self.plan_requests(
            prompts=prompts,
            sampling_spec=sampling_spec,
            samples_per_prompt=samples_per_prompt,
            control_algorithm=control_algorithm,
            rollout_id=rollout_id,
            collect_media_preview=collect_media_preview,
            media_max_items=media_max_items,
            initial_latents=initial_latents,
        )
        responses = self.exec_request(
            req=req,
            rollout_group=rollout_group,
            samples_per_prompt=samples_per_prompt,
        )
        combined = self.aggregate(responses=responses)
        if collect_media_preview:
            combined.cap_media_preview(int(media_max_items))

        # Resolve the resolved sampling_params (post-rollout-id seed mix).
        import dataclasses

        sde_indices_list = list(sde_indices) if sde_indices is not None else None
        sampling_params = dataclasses.replace(
            sampling_spec,
            num_samples_per_prompt=int(samples_per_prompt),
            sde_indices=sde_indices_list,
            seed=mix_rollout_base_seed(int(sampling_spec.seed), int(rollout_id)),
        )
        training_batch, sample_count = self.convert_training_data(
            combined_response=combined,
            request=req,
            sampling_params=sampling_params,
            sde_indices=sde_indices,
            control_algorithm=control_algorithm,
        )
        return training_batch, sample_count, combined

    # ------------------------------------------------------------------
    # Evaluation entrypoint
    # ------------------------------------------------------------------

    def run_eval(
        self,
        *,
        rollout_group: "NewRolloutActorGroup",
        data_source: Any,
        prompt_batch_size: int,
        samples_per_prompt: int,
        sampling_spec: SamplingParams,
        evaluation_settings: Any,
        rollout_id: int,
        collect_media_preview: bool = False,
        media_max_items: int = 8,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Run one evaluation pass and return metrics.

        Mirrors :meth:`RolloutPipeline.run_eval` for the new types path.
        Generates + scores via ``run_eval_pipeline`` (no advantages), aggregates,
        and computes ``mean_reward`` / ``std_reward``. When
        ``collect_media_preview=True``, the result dict also carries a
        ``MediaPreview`` ready for ``wandb_logger.log_generated_media``.
        """
        import dataclasses

        request_batch = build_eval_request_batch(
            data_source=data_source,
            prompt_batch_size=prompt_batch_size,
            evaluation_settings=evaluation_settings,
        )
        raw_prompts = list(request_batch.get("prompts", []) or [])
        if not raw_prompts:
            return {
                "rollout_id": int(rollout_id),
                "num_samples": 0,
                "mean_reward": 0.0,
                "std_reward": 0.0,
            }
        _reject_unsupported_media_refs(request_batch, context="NewRolloutPipeline.run_eval")

        raw_prompt_ids = request_batch.get("prompt_ids")
        prompt_ids_for_eval = (
            [str(pid) for pid in raw_prompt_ids]
            if isinstance(raw_prompt_ids, list) and len(raw_prompt_ids) == len(raw_prompts)
            else None
        )
        raw_eval_images = request_batch.get("images")
        eval_images = (
            list(raw_eval_images)
            if isinstance(raw_eval_images, list) and len(raw_eval_images) == len(raw_prompts)
            else None
        )
        prompts = Prompts.from_unique_prompts(raw_prompts, prompt_ids=prompt_ids_for_eval, images=eval_images)
        if int(samples_per_prompt) > 1:
            prompts = prompts.expand(
                int(samples_per_prompt),
                init_same_noise=bool(getattr(sampling_spec, "init_same_noise", False)),
            )

        # Build eval SamplingParams via dataclasses.replace. Eval is hard-wired
        # to ``sde_indices=None`` — the deterministic clean-latents path —
        # because ``EvaluationConfig`` (diffusionrl/config/evaluation_config.py)
        # has no fields for stochastic eval (no ``eta``, no
        # ``num_inference_steps`` override). ``sampling_spec.sde_config.eta``
        # rides through unchanged but is dormant in the scheduler since no
        # step is SDE-gated; this matches the per-stage contract documented
        # on ``_build_diffusion_stage_params``. If a future EvaluationConfig
        # grows a stochastic-eval mode, the right move is to add an
        # ``eval_sde_indices`` field on ``EvaluationConfig`` and resolve it
        # here — NOT to silently re-route the (still-dormant) ``eta`` field
        # back into a request that has no SDE step.
        eval_overrides: Dict[str, Any] = {
            "num_samples_per_prompt": int(samples_per_prompt),
            "sde_indices": None,
            "sampler_kwargs": dict(sampling_spec.sampler_kwargs or {}),
        }
        sampling_params = dataclasses.replace(sampling_spec, **eval_overrides)

        diffusion_params = _build_diffusion_stage_params(
            sampling_params=sampling_params,
            samples_per_prompt=int(samples_per_prompt),
            sde_indices=None,
        )
        request_conditions: Dict[str, Any] = {}
        if initial_latents is not None:
            if int(initial_latents.shape[0]) != len(prompts.sample_ids):
                raise ValueError(
                    f"run_eval: initial_latents.shape[0]={int(initial_latents.shape[0])} "
                    f"!= len(prompts.sample_ids)={len(prompts.sample_ids)}"
                )
            request_conditions["initial_latents"] = ImageLatentCondition(latents=initial_latents)

        primitives: Dict[str, Any] = {"text": Texts(texts=list(prompts.prompts))}
        images_prim = _build_images_primitive(prompts.images, prompt_count=len(prompts.prompts), context="run_eval")
        if images_prim is not None:
            primitives["image"] = images_prim

        req = RolloutReq(
            sample_ids=list(prompts.sample_ids),
            group_ids=list(prompts.group_ids),
            primitives=primitives,
            request_conditions=request_conditions,
            stage_params={
                "diffusion": diffusion_params,
                "collect_media_preview": bool(collect_media_preview),
                "media_max_items": max(1, int(media_max_items)),
            },
        )

        if rollout_group.num_actors == 0:
            raise RuntimeError("NewRolloutPipeline.run_eval: no rollout actors.")
        if rollout_group.num_actors == 1:
            responses = ray.get(rollout_group.get_actors()[0].run_eval_pipeline.remote(req))
        else:
            shards = rollout_group.rollout_plan.shard_grouped_req(
                req,
                num_actors=rollout_group.num_actors,
                samples_per_prompt=int(samples_per_prompt),
            )
            nested = rollout_group.scatter_gather("run_eval_pipeline", shards)
            responses = [resp for sub in nested for resp in sub]

        combined = self.aggregate(responses=responses)
        if collect_media_preview:
            combined.cap_media_preview(int(media_max_items))

        rewards = combined.rewards
        if rewards is None or rewards.numel() == 0:
            result: Dict[str, Any] = {
                "rollout_id": int(rollout_id),
                "num_samples": len(raw_prompts),
                "mean_reward": 0.0,
                "std_reward": 0.0,
            }
        else:
            result = {
                "rollout_id": int(rollout_id),
                "num_samples": len(raw_prompts),
                "mean_reward": float(rewards.mean().item()),
                "std_reward": float(rewards.std().item()),
            }

        if collect_media_preview:
            preview = combined.media_preview
            if isinstance(preview, MediaPreview) and not preview.is_empty():
                result["media_preview"] = preview
        return result


__all__ = ["NewRolloutPipeline", "build_media_preview"]
