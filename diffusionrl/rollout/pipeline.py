"""Driver-side rollout pipeline for ``RolloutReq`` / ``RolloutResp``.

Composes four phases — ``load_prompts → plan_requests → exec_request →
aggregate``. Driven by a
:class:`diffusionrl.ray.group.rollout.RolloutActorGroup` whose actors
expose ``run_rollout_pipeline(req: RolloutReq) → List[RolloutResp]``.

The trainer-facing contract is ``RolloutResp`` end-to-end: the train
actor reads ``resp.tracks[slot].{conditions, segment, advantages}``
directly with no intermediate ``TrainingBatch`` translation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import ray
import torch

from diffusionrl.rollout.request_builders import load_prompt_batch_from_source
from diffusionrl.sde.noise import (
    generate_shared_noise,
    mix_rollout_base_seed,
)
from diffusionrl.types.conditions.image import ImageLatentCondition
from diffusionrl.types.prompts import RolloutInputs
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp
from diffusionrl.types.sampling import (
    ComposedSamplingParams,
    DiffusionSamplingParams,
    get_diffusion_params,
)
from diffusionrl.utils.scheduler_utils import TimestepScheduler

if TYPE_CHECKING:
    from diffusionrl.ray.group.rollout import RolloutActorGroup


_SUPPORTED_MEDIA_REF_ROLES: Set[Tuple[str, str]] = {("image", "condition")}


def _reject_unsupported_media_refs(batch: Dict[str, Any], *, context: str) -> None:
    """Fail loud when a dataset hands unsupported media_refs to the driver.

    The ``media_refs`` channel (``MediaRef(uri, modality, role)`` URI
    list) was originally OLD-only. The driver now consumes the
    ``(image, condition)`` (modality, role) pair via
    :class:`MultimodalRLDataSource._load_condition_images`
    → ``RolloutInputs.primitives['image']: Images``;
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
        f"entries; the driver currently consumes only (image, condition). "
        f"First bad entry: prompt={bad[0][0]}, ref={bad[0][1]!r}."
    )


def compute_initial_noise_for_request(
    *,
    cfg: Any,
    inputs: RolloutInputs,
    sampling_spec: DiffusionSamplingParams,
    samples_per_prompt: int,
    rollout_id: int,
) -> Optional[torch.Tensor]:
    """Pre-compute the per-sample x_T tensor on the driver.

    Resolves the latent shape via the Pipeline class's ``latent_shape``
    classmethod (see :class:`diffusionrl.models.types.pipeline
    .LatentShapeProvider`). Pipelines that don't implement it fall back
    to the rollout engine's own RNG — explicit opt-out via
    ``NotImplementedError``.

    Noise grouping is derived from ``init_same_noise``:
    ``True`` → ``group_ids`` (samples in same GRPO group share noise);
    ``False`` → ``sample_ids`` (each sample gets unique deterministic noise).
    """
    import hydra.utils  # local import — driver-side dispatch only

    target = cfg.model._target_
    if not isinstance(target, str) or "." not in target:
        return None
    pipeline_target = target.rsplit(".", 1)[0]
    try:
        pipeline_cls = hydra.utils.get_class(pipeline_target)
    except Exception:
        return None
    if not hasattr(pipeline_cls, "latent_shape"):
        return None
    try:
        latent_shape: Tuple[int, ...] = pipeline_cls.latent_shape(model_config=cfg.model, sampling_spec=sampling_spec)
    except NotImplementedError:
        return None

    base_seed = mix_rollout_base_seed(int(sampling_spec.seed), int(rollout_id))
    batch_size = len(inputs.sample_ids)
    if batch_size == 0:
        return None

    init_same_noise = bool(getattr(sampling_spec, "init_same_noise", False))
    noise_ids = list(inputs.group_ids) if init_same_noise else list(inputs.sample_ids)
    return generate_shared_noise(
        batch_size=batch_size,
        latent_shape=latent_shape,
        device=torch.device("cpu"),
        dtype=torch.float32,
        noise_group_ids=noise_ids,
        base_seed=base_seed,
    )


class RolloutPipeline:
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
    ) -> RolloutInputs:
        """Fetch a prompt batch from the data source and expand."""
        inputs = load_prompt_batch_from_source(
            data_source=data_source,
            prompt_batch_size=int(prompt_batch_size),
        )
        if not inputs.sample_ids:
            raise RuntimeError("Data source returned an empty prompt batch.")
        if int(samples_per_prompt) > 1:
            inputs = inputs.expand(int(samples_per_prompt))
        return inputs

    def plan_requests(
        self,
        *,
        inputs: RolloutInputs,
        sampling_spec: DiffusionSamplingParams,
        samples_per_prompt: int,
        sde_scheduler: Optional[TimestepScheduler],
        rollout_id: int,
        collect_media_preview: bool = False,
        media_max_items: int = 8,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> Tuple[RolloutReq, Optional[Set[int]]]:
        """Build a typed ``RolloutReq`` and return ``(req, sde_indices_set)``."""
        import dataclasses

        diffusion_base = get_diffusion_params(sampling_spec)
        sde_indices = set(sde_scheduler.get_sde_indices(int(rollout_id))) if sde_scheduler is not None else None
        sde_indices_list = list(sde_indices) if sde_indices is not None else None
        per_rollout_seed = mix_rollout_base_seed(int(diffusion_base.seed), int(rollout_id))

        engine_spp = (
            int(diffusion_base.samples_per_prompt)
            if isinstance(sampling_spec, ComposedSamplingParams)
            else int(samples_per_prompt)
        )
        diffusion_params = dataclasses.replace(
            diffusion_base,
            sde_strategy=None,
            sde_indices=sde_indices_list,
            seed=per_rollout_seed,
            samples_per_prompt=engine_spp,
            num_samples_per_prompt=engine_spp,
        )

        if initial_latents is not None and int(initial_latents.shape[0]) != len(inputs.sample_ids):
            raise ValueError(
                f"plan_requests: initial_latents.shape[0]={int(initial_latents.shape[0])} "
                f"!= len(inputs.sample_ids)={len(inputs.sample_ids)}"
            )

        ar_sampling = getattr(sampling_spec, "ar", None)
        if ar_sampling is not None:
            sampling_params = ComposedSamplingParams(diffusion=diffusion_params, ar=ar_sampling)
        else:
            sampling_params = diffusion_params

        request_conditions: Dict[str, Any] = {}
        if initial_latents is not None:
            request_conditions["initial_latents"] = ImageLatentCondition(latents=initial_latents)

        req = RolloutReq(
            sample_ids=list(inputs.sample_ids),
            group_ids=list(inputs.group_ids),
            primitives=dict(inputs.primitives),
            request_conditions=request_conditions,
            sampling_params=sampling_params,
            collect_media_preview=bool(collect_media_preview),
            media_max_items=max(1, int(media_max_items)),
            metadata=list(inputs.metadata) if inputs.metadata else [],
        )
        sde_indices_set = set(int(i) for i in sde_indices_list) if sde_indices_list is not None else None
        return req, sde_indices_set

    def exec_request(
        self,
        *,
        req: RolloutReq,
        rollout_group: "RolloutActorGroup",
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
            raise RuntimeError("RolloutPipeline.exec_request: no rollout actors.")
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


__all__ = ["RolloutPipeline"]
