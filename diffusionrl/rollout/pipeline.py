"""Driver-side rollout pipeline used by ``diffusionrl.train``.

``RolloutPipeline`` exposes the rollout flow as a set of decomposed phase
sub-methods (``load_prompts`` → ``plan_requests`` → ``exec_request`` →
``aggregate`` → ``convert_training_data``) plus an all-in-one ``run_once``
composer that ``train.py`` calls inside its step loop. Subclasses can
override individual phases to customize one stage without re-implementing
the whole loop.

Reward scoring and advantage computation are handled **inside the rollout
actor** via ``RolloutActor.run_rollout_pipeline``; the ``score_samples`` and
``compute_advantages`` methods on this class are kept as override hooks that
raise ``NotImplementedError`` when the default ``run_once`` flow is used.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import ray

from diffusionrl.ray.group.base import ActorGroup
from diffusionrl.rollout.request_builders import (
    build_eval_request_batch,
    load_prompt_batch_from_source,
)
from diffusionrl.samplers.utils.noise import mix_rollout_base_seed
from diffusionrl.types.prompts import Prompts
from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.response import RolloutResponse
from diffusionrl.types.sample import MediaPreview
from diffusionrl.types.sampling import SamplingParams, SDEConfig
from diffusionrl.types.training_batch import TrainingBatch

if TYPE_CHECKING:
    from diffusionrl.algorithms.base import BaseAlgorithm


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


class RolloutPipeline:
    """Phase-decomposed driver-side rollout pipeline for the new-actor path."""

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
        """Fetch a raw prompt batch and expand to ``Prompts`` with samples_per_prompt.

        When the data source provides ``prompt_ids`` (e.g. dataset row
        identifiers like ``train.txt:14963``), thread them through so
        that initial-noise seeding via ``_derive_group_seed(base_seed,
        noise_group_id)`` matches the legacy pipeline. Without this,
        ``Prompts.from_unique_prompts`` synthesizes generic ``prompt:N``
        IDs, producing different blake2b hashes → different initial
        latents → different trajectories, even with deterministic kernels.

        ``init_same_noise`` is forwarded to ``Prompts.expand`` so the resulting
        ``noise_group_ids`` match the caller's intent (share initial noise
        across K samples of the same prompt vs. per-sample independent noise).
        Defaults to ``False`` to preserve the historical per-sample behaviour;
        ``run_once`` / ``run_eval`` pull the real value from the sampling
        spec and pass it through so SGLang (which receives the same
        ``noise_group_ids``) stays aligned with FSDP.
        """
        prompt_batch = load_prompt_batch_from_source(
            data_source=data_source,
            prompt_batch_size=int(prompt_batch_size),
        )
        raw_prompts = list(prompt_batch.get("prompts") or [])
        if not raw_prompts:
            raise RuntimeError("Data source returned an empty prompt batch.")
        raw_prompt_ids = prompt_batch.get("prompt_ids")
        prompt_ids = (
            [str(pid) for pid in raw_prompt_ids]
            if isinstance(raw_prompt_ids, list) and len(raw_prompt_ids) == len(raw_prompts)
            else None
        )
        # Pass metadata through so downstream I2V pipelines can access image paths
        raw_metadata = prompt_batch.get("metadata")
        prompt_metadata = (
            list(raw_metadata) if isinstance(raw_metadata, list) and len(raw_metadata) == len(raw_prompts) else None
        )
        prompts = Prompts.from_unique_prompts(raw_prompts, prompt_ids=prompt_ids, prompt_metadata=prompt_metadata)
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
    ) -> Tuple[RolloutRequest, Optional[Set[int]]]:
        """Build a typed RolloutRequest and return ``(request, sde_indices_set)``.

        ``collect_media_preview`` / ``media_max_items`` are threaded onto
        the resulting ``RolloutRequest`` so the actor-side rollout pipeline
        knows whether (and how many) PIL images to retain in
        ``samples.media_preview`` before dropping the full decoded lists.
        """
        import dataclasses

        sde_indices = control_algorithm.resolve_rollout_sde_indices(
            current_step=int(rollout_id),
        )
        sde_indices_list = list(sde_indices) if sde_indices is not None else None
        per_rollout_seed = mix_rollout_base_seed(int(sampling_spec.seed), int(rollout_id))
        sampling_params = dataclasses.replace(
            sampling_spec,
            num_samples_per_prompt=int(samples_per_prompt),
            sde_indices=sde_indices_list,
            seed=per_rollout_seed,
        )
        request = RolloutRequest(
            prompts=prompts,
            sampling_params=sampling_params,
            collect_media_preview=bool(collect_media_preview),
            media_max_items=max(1, int(media_max_items)),
        )
        sde_indices_set = set(int(i) for i in sde_indices_list) if sde_indices_list is not None else None
        return request, sde_indices_set

    def exec_request(
        self,
        *,
        request: RolloutRequest,
        rollout_group: ActorGroup,
        samples_per_prompt: int,
    ) -> List[RolloutResponse]:
        """Dispatch to actor(s) using the fused ``run_rollout_pipeline`` actor method.

        Single-actor fast path uses a direct ``ray.get`` on the one handle.
        Multi-actor path shards the request at group boundaries via
        ``rollout_group.rollout_plan.shard_grouped`` and scatter-gathers across all
        actors. Shards cover whole groups so that per-actor advantage
        normalization (with ``use_global_std=True``) always sees complete
        group populations; std is therefore scoped to one actor's shard in
        the multi-actor case.
        """
        if rollout_group.num_actors == 0:
            raise RuntimeError("RolloutPipeline.exec_request: no rollout actors.")
        if rollout_group.num_actors == 1:
            return ray.get(rollout_group.get_actors()[0].run_rollout_pipeline.remote(request))

        shards = rollout_group.rollout_plan.shard_grouped(
            request,
            num_actors=rollout_group.num_actors,
            samples_per_prompt=int(samples_per_prompt),
        )
        nested = rollout_group.scatter_gather("run_rollout_pipeline", shards)
        return [response for sub in nested for response in sub]

    def aggregate(
        self,
        *,
        responses: List[RolloutResponse],
    ) -> RolloutResponse:
        """Concat per-actor RolloutResponses into a single merged response.

        After concat, the per-shard ``samples.media_preview`` payloads are
        already merged by ``RolloutSamples.concat`` (lists extended). Cap
        the final preview list at ``request.media_max_items`` so the driver
        never holds more PIL images than the caller asked for regardless
        of how many shards contributed.
        """
        if not responses:
            raise RuntimeError("Rollout produced no responses.")
        combined = RolloutResponse.concat(responses)
        combined.samples.cap_media_preview(int(combined.request.media_max_items))
        return combined

    def score_samples(self, *args: Any, **kwargs: Any) -> None:
        """Driver-side reward scoring hook.

        Not used by the default new-actor ``run_once`` flow — reward scoring
        happens inside the actor via ``RolloutActor.run_rollout_pipeline`` →
        ``attach_reward``. Override this sub-method only when building a
        custom driver-side pipeline that needs driver-side scoring.
        """
        raise NotImplementedError(
            "RolloutPipeline.score_samples is handled actor-side in the new-actor "
            "flow; override only if you're building a custom pipeline that needs "
            "driver-side scoring."
        )

    def compute_advantages(self, *args: Any, **kwargs: Any) -> None:
        """Driver-side advantage computation hook.

        Not used by the default new-actor ``run_once`` flow — advantages are
        z-scored per group inside the actor via
        ``RolloutActor.run_rollout_pipeline`` → ``compute_advantages``.
        Override this sub-method only when building a custom driver-side
        pipeline.
        """
        raise NotImplementedError(
            "RolloutPipeline.compute_advantages is handled actor-side in the "
            "new-actor flow; override only if you're building a custom pipeline."
        )

    def convert_training_data(
        self,
        *,
        combined_response: RolloutResponse,
        sde_indices: Optional[Set[int]],
        control_algorithm: Optional[BaseAlgorithm] = None,
    ) -> Tuple[TrainingBatch, int]:
        """Assemble the final TrainingBatch and return ``(batch, sample_count)``.

        When *control_algorithm* is provided, applies
        ``get_filtered_training_indices`` to exclude boundary timesteps
        (e.g. ``skip_last_timestep``) whose log-probs may be numerically invalid.
        """
        if sde_indices is not None and control_algorithm is not None:
            filtered = control_algorithm.get_filtered_training_indices(
                sde_indices,
                int(combined_response.samples.timesteps.shape[0]),
            )
            if filtered:
                sde_indices = filtered
        training_batch = combined_response.to_training_batch(sde_indices=sde_indices)
        return training_batch, int(training_batch.batch_size)

    # ------------------------------------------------------------------
    # All-in-one entrypoint composing the sub-methods
    # ------------------------------------------------------------------

    def run_once(
        self,
        *,
        rollout_group: ActorGroup,
        data_source: Any,
        prompt_batch_size: int,
        samples_per_prompt: int,
        sampling_spec: SamplingParams,
        control_algorithm: Any,
        rollout_id: int,
        collect_media_preview: bool = False,
        media_max_items: int = 8,
    ) -> Tuple[TrainingBatch, int, RolloutResponse]:
        """Execute one full driver-side rollout step.

        Composition of the pipeline sub-methods:
            load_prompts → plan_requests → exec_request → aggregate → convert_training_data

        Returns ``(training_batch, sample_count, combined_response)``. When
        ``collect_media_preview=True``, the aggregated response carries a
        capped preview on ``combined.samples.media_preview`` and full
        decoded images are never returned across Ray.  Called by
        ``train.py`` inside its PHASE A step-loop block.
        """
        prompts = self.load_prompts(
            data_source=data_source,
            prompt_batch_size=prompt_batch_size,
            samples_per_prompt=samples_per_prompt,
            init_same_noise=bool(getattr(sampling_spec, "init_same_noise", False)),
        )
        request, sde_indices = self.plan_requests(
            prompts=prompts,
            sampling_spec=sampling_spec,
            samples_per_prompt=samples_per_prompt,
            control_algorithm=control_algorithm,
            rollout_id=rollout_id,
            collect_media_preview=collect_media_preview,
            media_max_items=media_max_items,
        )
        responses = self.exec_request(
            request=request,
            rollout_group=rollout_group,
            samples_per_prompt=samples_per_prompt,
        )
        combined = self.aggregate(responses=responses)
        training_batch, sample_count = self.convert_training_data(
            combined_response=combined,
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
        rollout_group: ActorGroup,
        data_source: Any,
        prompt_batch_size: int,
        samples_per_prompt: int,
        sampling_spec: SamplingParams,
        evaluation_settings: Any,
        rollout_id: int,
        collect_media_preview: bool = False,
        media_max_items: int = 8,
    ) -> Dict[str, Any]:
        """Run one evaluation pass and return metrics.

        Loads eval prompts via ``build_eval_request_batch``, dispatches
        ``run_eval_pipeline`` to actors (generate + reward, no advantages),
        and computes mean/std reward.

        When ``collect_media_preview=True``, the returned dict also
        includes a ``media_preview`` key carrying up to ``media_max_items``
        decoded media (PIL images and/or 4D ``(C, T, H, W)`` video tensors)
        paired with their prompts and reward scores — ready to hand to
        ``wandb_logger.log_generated_media(
        ..., image_key="eval/generated_images",
        video_key="eval/generated_videos")``. Image-only, video-only, and
        image+video previews are all supported by the ``MediaPreview`` payload.

        Returns dict with keys ``rollout_id``, ``num_samples``,
        ``mean_reward``, ``std_reward``, and optionally ``media_preview``.
        """
        # 1. Load eval prompts
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

        # 2. Build typed Prompts (thread prompt_ids through for seed parity)
        raw_prompt_ids = request_batch.get("prompt_ids")
        prompt_ids_for_eval = (
            [str(pid) for pid in raw_prompt_ids]
            if isinstance(raw_prompt_ids, list) and len(raw_prompt_ids) == len(raw_prompts)
            else None
        )
        prompts = Prompts.from_unique_prompts(raw_prompts, prompt_ids=prompt_ids_for_eval)
        if int(samples_per_prompt) > 1:
            prompts = prompts.expand(
                int(samples_per_prompt),
                init_same_noise=bool(getattr(sampling_spec, "init_same_noise", False)),
            )

        # 3. Build eval SamplingParams with config overrides. Strategy choice
        # inherits from the sampling-side cfg.sampling.sde_strategy (the
        # injected RolloutEngine.strategy on the actor); only eta and
        # num_inference_steps are eval-overridable.
        eval_num_steps = getattr(evaluation_settings, "num_inference_steps", None)
        eval_eta = getattr(evaluation_settings, "eta", None)

        eval_sde_config = SDEConfig(
            eta=float(eval_eta) if eval_eta is not None else float(sampling_spec.sde_config.eta),
            shift=float(sampling_spec.sde_config.shift),
        )
        # Build eval SamplingParams via dataclasses.replace so any non-eval
        # fields on the training-side spec (e.g. the three precision fields)
        # are preserved automatically. A hand-rolled SamplingParams(...)
        # silently drops any field not explicitly listed.
        eval_overrides: Dict[str, Any] = {
            "num_samples_per_prompt": int(samples_per_prompt),
            "sde_config": eval_sde_config,
            "sde_indices": None,
            "sampler_kwargs": dict(sampling_spec.sampler_kwargs or {}),
        }
        if eval_num_steps is not None:
            eval_overrides["num_inference_steps"] = int(eval_num_steps)
        sampling_params = dataclasses.replace(sampling_spec, **eval_overrides)
        request = RolloutRequest(
            prompts=prompts,
            sampling_params=sampling_params,
            collect_media_preview=bool(collect_media_preview),
            media_max_items=max(1, int(media_max_items)),
        )

        # 4. Dispatch to actors via run_eval_pipeline
        if rollout_group.num_actors == 0:
            raise RuntimeError("RolloutPipeline.run_eval: no rollout actors.")
        if rollout_group.num_actors == 1:
            responses = ray.get(rollout_group.get_actors()[0].run_eval_pipeline.remote(request))
        else:
            shards = rollout_group.rollout_plan.shard_grouped(
                request,
                num_actors=rollout_group.num_actors,
                samples_per_prompt=int(samples_per_prompt),
            )
            nested = rollout_group.scatter_gather("run_eval_pipeline", shards)
            responses = [resp for sub in nested for resp in sub]

        # 5. Aggregate and compute reward statistics
        combined = self.aggregate(responses=responses)
        rewards = combined.samples.rewards
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
            preview = combined.samples.media_preview
            if isinstance(preview, MediaPreview) and not preview.is_empty():
                result["media_preview"] = preview
        return result


__all__ = ["RolloutPipeline", "build_media_preview"]
