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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import ray
import torch
from ray.actor import ActorHandle

from diffusionrl.ray.generate_sharding import build_generate_shard_plan_grouped
from diffusionrl.ray.group_base import ActorHandleGroup
from diffusionrl.rollout.request_builders import (
    build_eval_request_batch,
    load_prompt_batch_from_source,
)
from diffusionrl.samplers.noise_utils import mix_rollout_base_seed
from diffusionrl.types.prompts import Prompts
from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.response import RolloutResponse
from diffusionrl.types.sampling import SamplingParams, SDEConfig
from diffusionrl.types.training_batch import TrainingBatch

if TYPE_CHECKING:
    from diffusionrl.algorithms.base import BaseAlgorithm


def build_media_preview(
    response: RolloutResponse,
    *,
    max_items: int = 8,
) -> Optional[Dict[str, Any]]:
    """Build a wandb media preview dict from an aggregated RolloutResponse.

    Returns ``{"images": [...], "prompts": [...], "rewards": [...]}`` capped
    at *max_items*, or ``None`` when no decoded images are available.
    """
    decoded_images = response.samples.decoded_images
    if decoded_images is None:
        return None

    limit = max(1, int(max_items))
    prompt_texts = response.request.prompts.prompts
    rewards_flat: List[float] = []
    if response.samples.rewards is not None and torch.is_tensor(response.samples.rewards):
        rewards_flat = [float(v) for v in response.samples.rewards.detach().cpu().reshape(-1).tolist()]

    images: List[Any] = []
    prompts: List[str] = []
    reward_values: List[float] = []

    for i, img in enumerate(decoded_images):
        if len(images) >= limit:
            break
        if not hasattr(img, "save"):
            continue
        images.append(img)
        prompts.append(str(prompt_texts[i]) if i < len(prompt_texts) else "")
        reward_values.append(float(rewards_flat[i]) if i < len(rewards_flat) else 0.0)

    if not images:
        return None

    return {"images": images, "prompts": prompts, "rewards": reward_values}


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
        prompts = Prompts.from_unique_prompts(raw_prompts, prompt_ids=prompt_ids)
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
    ) -> Tuple[RolloutRequest, Optional[Set[int]]]:
        """Build a typed RolloutRequest and return ``(request, sde_indices_set)``."""
        import dataclasses

        sde_indices = control_algorithm.resolve_rollout_sde_indices(
            current_step=int(rollout_id),
        )
        sde_indices_list = list(sde_indices) if sde_indices is not None else None
        per_rollout_seed = mix_rollout_base_seed(
            int(sampling_spec.seed), int(rollout_id)
        )
        sampling_params = dataclasses.replace(
            sampling_spec,
            num_samples_per_prompt=int(samples_per_prompt),
            sde_indices=sde_indices_list,
            seed=per_rollout_seed,
        )
        request = RolloutRequest(prompts=prompts, sampling_params=sampling_params)
        sde_indices_set = set(int(i) for i in (sde_indices_list or [])) or None
        return request, sde_indices_set

    def exec_request(
        self,
        *,
        request: RolloutRequest,
        rollout_actors: List[ActorHandle],
        rollout_handle_group: ActorHandleGroup,
        samples_per_prompt: int,
    ) -> List[RolloutResponse]:
        """Dispatch to actor(s) using the fused ``run_rollout_pipeline`` actor method.

        Single-actor fast path uses a direct ``ray.get`` on the one handle.
        Multi-actor path shards the request at group boundaries via
        ``build_generate_shard_plan_grouped`` and scatter-gathers across all
        actors. Shards cover whole groups so that per-actor advantage
        normalization (with ``use_global_std=True``) always sees complete
        group populations; std is therefore scoped to one actor's shard in
        the multi-actor case.
        """
        if not rollout_actors:
            raise RuntimeError("RolloutPipeline.exec_request: no rollout actors.")
        if len(rollout_actors) == 1:
            return ray.get(rollout_actors[0].run_rollout_pipeline.remote(request))

        shards = build_generate_shard_plan_grouped(
            request=request,
            num_actors=len(rollout_actors),
            samples_per_prompt=int(samples_per_prompt),
        )
        nested_refs = rollout_handle_group.scatter_gather_async("run_rollout_pipeline", shards)
        nested = ray.get(nested_refs)
        return [response for sub in nested for response in sub]

    def aggregate(
        self,
        *,
        responses: List[RolloutResponse],
    ) -> RolloutResponse:
        """Concat per-actor RolloutResponses into a single merged response."""
        if not responses:
            raise RuntimeError("Rollout produced no responses.")
        return RolloutResponse.concat(responses)

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
        rollout_actors: List[ActorHandle],
        rollout_handle_group: ActorHandleGroup,
        data_source: Any,
        prompt_batch_size: int,
        samples_per_prompt: int,
        sampling_spec: SamplingParams,
        control_algorithm: Any,
        rollout_id: int,
    ) -> Tuple[TrainingBatch, int, RolloutResponse]:
        """Execute one full driver-side rollout step.

        Composition of the pipeline sub-methods:
            load_prompts → plan_requests → exec_request → aggregate → convert_training_data

        Returns ``(training_batch, sample_count, combined_response)``.
        The caller may use the combined response for media preview logging
        before discarding it.  Called by ``train.py`` inside its
        PHASE A step-loop block.
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
        )
        responses = self.exec_request(
            request=request,
            rollout_actors=rollout_actors,
            rollout_handle_group=rollout_handle_group,
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
        rollout_actors: List[ActorHandle],
        rollout_handle_group: ActorHandleGroup,
        data_source: Any,
        prompt_batch_size: int,
        samples_per_prompt: int,
        sampling_spec: SamplingParams,
        evaluation_settings: Any,
        rollout_id: int,
    ) -> Dict[str, Any]:
        """Run one evaluation pass and return metrics.

        Loads eval prompts via ``build_eval_request_batch``, dispatches
        ``run_eval_pipeline`` to actors (generate + reward, no advantages),
        and computes mean/std reward.

        Returns dict with keys ``rollout_id``, ``num_samples``,
        ``mean_reward``, ``std_reward``.
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

        # 3. Build eval SamplingParams with config overrides
        eval_num_steps = getattr(evaluation_settings, "num_inference_steps", None)
        eval_eta = getattr(evaluation_settings, "eta", None)
        raw_sde_type = getattr(evaluation_settings, "sde_type", None)
        eval_sde_type = (
            str(raw_sde_type).strip()
            if raw_sde_type is not None and str(raw_sde_type).strip()
            else str(sampling_spec.sde_config.sde_type)
        )

        eval_sde_config = SDEConfig(
            eta=float(eval_eta) if eval_eta is not None else float(sampling_spec.sde_config.eta),
            sde_type=eval_sde_type,
            shift=float(sampling_spec.sde_config.shift),
        )
        sampling_params = SamplingParams(
            num_inference_steps=int(eval_num_steps)
            if eval_num_steps is not None
            else int(sampling_spec.num_inference_steps),
            guidance_scale=float(sampling_spec.guidance_scale),
            height=int(sampling_spec.height),
            width=int(sampling_spec.width),
            num_frames=int(sampling_spec.num_frames),
            seed=int(sampling_spec.seed),
            num_samples_per_prompt=int(samples_per_prompt),
            init_same_noise=bool(sampling_spec.init_same_noise),
            sde_config=eval_sde_config,
            sde_indices=None,
            sampler_kwargs=dict(sampling_spec.sampler_kwargs or {}),
        )
        request = RolloutRequest(prompts=prompts, sampling_params=sampling_params)

        # 4. Dispatch to actors via run_eval_pipeline
        if not rollout_actors:
            raise RuntimeError("RolloutPipeline.run_eval: no rollout actors.")
        if len(rollout_actors) == 1:
            responses = ray.get(rollout_actors[0].run_eval_pipeline.remote(request))
        else:
            shards = build_generate_shard_plan_grouped(
                request=request,
                num_actors=len(rollout_actors),
                samples_per_prompt=int(samples_per_prompt),
            )
            nested_refs = rollout_handle_group.scatter_gather_async("run_eval_pipeline", shards)
            nested = ray.get(nested_refs)
            responses = [resp for sub in nested for resp in sub]

        # 5. Aggregate and compute reward statistics
        combined = self.aggregate(responses=responses)
        rewards = combined.samples.rewards
        if rewards is None or rewards.numel() == 0:
            return {
                "rollout_id": int(rollout_id),
                "num_samples": len(raw_prompts),
                "mean_reward": 0.0,
                "std_reward": 0.0,
            }

        return {
            "rollout_id": int(rollout_id),
            "num_samples": len(raw_prompts),
            "mean_reward": float(rewards.mean().item()),
            "std_reward": float(rewards.std().item()),
        }


__all__ = ["RolloutPipeline", "build_media_preview"]
