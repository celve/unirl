"""Reward executors and aggregation helpers."""

from __future__ import annotations

from dataclasses import replace
import inspect
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from diffusionrl.reward.schema import RewardSchema
from diffusionrl.utils import load_function

from .base import (
    BaseRewardExecutor,
    BaseRewardScorer,
    RewardRequest,
    RewardResponse,
)
from .scorers.registry import resolve_builtin_reward_scorer_class

logger = logging.getLogger(__name__)

# Type alias for placement group result
PlacementGroupResult = Tuple[Any, List[int], List[int]]  # (pg, bundle_indices, gpu_ids)


class InProcessRewardExecutor(BaseRewardExecutor):
    """Thin executor wrapper around one in-process reward scorer."""

    def __init__(
        self,
        scorer: BaseRewardScorer,
        *,
        weight: float,
    ) -> None:
        super().__init__(
            model_name=scorer.get_model_name(),
            weight=weight,
            batch_size=scorer.batch_size,
            timeout=scorer.timeout,
        )
        self.scorer = scorer

    @property
    def preferred_input_kind(self) -> str:
        return self.scorer.preferred_input_kind

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        return self.scorer.compute_rewards(request)

    def is_available(self) -> bool:
        return self.scorer.is_available()

    def offload(self) -> None:
        self.scorer.offload()

    def onload(self) -> None:
        self.scorer.onload()

    def dispose(self) -> None:
        self.scorer.dispose()


class RewardService:
    """Reward service that owns per-component executors for one runtime host."""

    def _bind_reward_schema(
        self,
        reward_schema: RewardSchema,
        *,
        reward_pg_result: Optional[PlacementGroupResult],
        owner_name: str,
    ) -> None:
        if not isinstance(reward_schema, RewardSchema):
            raise TypeError(
                f"{owner_name} requires RewardSchema, "
                f"got: {type(reward_schema).__name__}"
            )
        self.reward_schema = reward_schema
        self.reward_definition = reward_schema.to_definition()
        self.reward_provider = reward_schema.to_provider_config()
        self.execution_plan = reward_schema.to_execution_plan()
        self.reward_pg = reward_pg_result
        self.executors = []
        self.aggregation = self.reward_definition.component_aggregation

    def __init__(
        self,
        reward_schema: RewardSchema,
        reward_pg_result: Optional[PlacementGroupResult] = None,
    ) -> None:
        self._bind_reward_schema(
            reward_schema,
            reward_pg_result=reward_pg_result,
            owner_name="RewardService",
        )

        self._init_executors()

        logger.info(
            "RewardService initialized with %d executor(s), aggregation=%s",
            len(self.executors),
            self.aggregation,
        )

    def _init_executors(self) -> None:
        """Initialize executors based on execution plan."""
        if self.execution_plan.uses_http_backend:
            self._init_http_executors()
        elif self._should_use_ray_executors():
            self._init_ray_executors()
        else:
            self._init_local_executors()

    def _should_use_ray_executors(self) -> bool:
        has_gpu_config = (
            self.execution_plan.uses_ray_backend
            and (
                self.execution_plan.dedicated_num_gpus > 0
                or self.execution_plan.dedicated_num_nodes > 0
            )
        )
        has_pg = self.reward_pg is not None
        return has_gpu_config and has_pg

    def _init_http_executors(self) -> None:
        """Initialize HTTP reward executors."""
        from .http import HTTPRewardExecutor

        urls = list(self.execution_plan.reward_service_urls or ())
        if not urls and self.execution_plan.reward_service_url:
            urls = [self.execution_plan.reward_service_url]
        reward_weights = self.reward_definition.reward_weights

        for i, url in enumerate(urls):
            if url is None:
                continue

            weight = 1.0
            if reward_weights and i < len(reward_weights):
                weight = reward_weights[i]

            executor = HTTPRewardExecutor(
                base_url=url,
                model_name=f"http_{i}",
                weight=weight,
                timeout=self.reward_provider.timeout,
                batch_size=self.reward_provider.batch_size,
            )
            self.executors.append(executor)
            logger.info("Added HTTPRewardExecutor: %s", url)

    def _init_ray_executors(self) -> None:
        """Initialize Ray reward executors for GPU-isolated rewards."""
        from .ray_executor import RayRewardExecutor

        pg, bundle_indices, gpu_ids = self.reward_pg
        gpus_per_actor = self.execution_plan.dedicated_gpus_per_actor
        reward_models = self.reward_definition.reward_models
        reward_weights = self.reward_definition.reward_weights

        if reward_models:
            weights = reward_weights or [1.0] * len(reward_models)

            for i, model in enumerate(reward_models):
                weight = weights[i] if i < len(weights) else 1.0

                actor_start = i * gpus_per_actor
                actor_end = actor_start + gpus_per_actor

                if actor_end > len(bundle_indices):
                    logger.warning(
                        "Not enough GPUs for model %s. Required %s, available %s",
                        model,
                        gpus_per_actor,
                        len(bundle_indices) - actor_start,
                    )
                    break

                executor = RayRewardExecutor(
                    model_name=model,
                    pg=pg,
                    bundle_indices=bundle_indices[actor_start:actor_end],
                    gpu_ids=gpu_ids[actor_start:actor_end],
                    reward_path=self.reward_provider.reward_path,
                    model_path=self.reward_provider.reward_model_saved_path,
                    num_actors=1,
                    gpus_per_actor=gpus_per_actor,
                    batch_size=self.reward_provider.batch_size,
                    timeout=self.reward_provider.timeout,
                    parallel_mode=False,
                    weight=weight,
                )
                self.executors.append(executor)
                logger.info("Added RayRewardExecutor: %s (weight=%s)", model, weight)

        else:
            num_actors = len(gpu_ids) // gpus_per_actor

            executor = RayRewardExecutor(
                model_name=self.reward_definition.default_model_name,
                pg=pg,
                bundle_indices=bundle_indices,
                gpu_ids=gpu_ids,
                reward_path=self.reward_provider.reward_path,
                model_path=self.reward_provider.reward_model_saved_path,
                num_actors=num_actors,
                gpus_per_actor=gpus_per_actor,
                batch_size=self.reward_provider.batch_size,
                timeout=self.reward_provider.timeout,
                parallel_mode=True,
                weight=1.0,
            )
            self.executors.append(executor)
            logger.info(
                "Added RayRewardExecutor: %s (%s parallel actors)",
                self.reward_definition.default_model_name,
                num_actors,
            )

    def _init_local_executors(self) -> None:
        """Initialize in-process executors backed by local scorers."""

        local_device_pref = str(
            getattr(self.execution_plan, "local_device", "cpu") or "cpu"
        ).strip().lower()
        if local_device_pref == "cpu":
            device = "cpu"
        elif local_device_pref == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif local_device_pref == "cuda":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                logger.warning(
                    "local_reward_device=cuda requested but CUDA is not available. "
                    "Falling back to CPU."
                )
                device = "cpu"
        else:
            logger.warning(
                "Unknown local_reward_device=%s. Falling back to CPU.",
                local_device_pref,
            )
            device = "cpu"

        if device == "cuda":
            logger.warning(
                "Local reward scorer is running on CUDA in-process. "
                "This can contend with rollout/training GPUs. Prefer dedicated reward "
                "actors (reward_dedicated_*) or HTTP reward service for isolation when "
                "you need strict resource isolation."
            )

        reward_path = getattr(self.reward_provider, "reward_path", None)

        def _create_executor(model_name: str, weight: float) -> BaseRewardExecutor:
            if reward_path:
                scorer_cls = load_function(reward_path)
            else:
                scorer_cls = resolve_builtin_reward_scorer_class(model_name)

            if not isinstance(scorer_cls, type) or not issubclass(scorer_cls, BaseRewardScorer):
                logger.warning(
                    "Local reward scorer %s does not inherit BaseRewardScorer; "
                    "treating it as a scorer via duck typing.",
                    reward_path or scorer_cls,
                )

            ctor_params = inspect.signature(scorer_cls.__init__).parameters
            init_kwargs: Dict[str, Any] = {
                "batch_size": self.reward_provider.batch_size,
                "timeout": self.reward_provider.timeout,
                "device": device,
            }
            if "model_name" in ctor_params:
                init_kwargs["model_name"] = model_name
            elif "frame_reward_model" in ctor_params:
                init_kwargs["frame_reward_model"] = model_name
            if "weight" in ctor_params:
                init_kwargs["weight"] = weight
            scorer = scorer_cls(**init_kwargs)
            return InProcessRewardExecutor(
                scorer=scorer,
                weight=weight,
            )

        reward_models = self.reward_definition.reward_models
        reward_weights = self.reward_definition.reward_weights

        if reward_models:
            weights = reward_weights or [1.0] * len(reward_models)

            for i, model in enumerate(reward_models):
                weight = weights[i] if i < len(weights) else 1.0
                executor = _create_executor(model_name=model, weight=weight)
                self.executors.append(executor)
                logger.info(
                    "Added in-process reward executor: %s via %s (weight=%s)",
                    model,
                    type(executor.scorer).__name__,
                    weight,
                )

        else:
            executor = _create_executor(
                model_name=self.reward_definition.default_model_name,
                weight=1.0,
            )
            self.executors.append(executor)
            logger.info(
                "Added in-process reward executor: %s via %s",
                self.reward_definition.default_model_name,
                type(executor.scorer).__name__,
            )

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        """Compute rewards using configured executors."""
        if not self.executors:
            return RewardResponse(
                rewards=[0.0] * request.batch_size,
                successes=[False] * request.batch_size,
                errors=["No executors configured"] * request.batch_size,
                compute_time=0.0,
            )

        start_time = time.time()

        if len(self.executors) == 1:
            return self.executors[0].compute_rewards(request)

        responses = []
        for executor in self.executors:
            try:
                resp = executor.compute_rewards(request)
                responses.append((resp, executor))
            except Exception as e:
                logger.error("Executor %s failed: %s", executor.get_model_name(), e)
                error_resp = RewardResponse(
                    rewards=[0.0] * request.batch_size,
                    successes=[False] * request.batch_size,
                    errors=[str(e)] * request.batch_size,
                    compute_time=0.0,
                )
                responses.append((error_resp, executor))

        return self._aggregate_responses(responses, time.time() - start_time)

    def _aggregate_responses(
        self,
        responses: List[Tuple[RewardResponse, BaseRewardExecutor]],
        total_time: float,
    ) -> RewardResponse:
        """Aggregate responses from multiple executors."""
        if not responses:
            return RewardResponse(
                rewards=[],
                successes=[],
                errors=[],
                compute_time=total_time,
            )

        batch_size = responses[0][0].batch_size

        if self.aggregation == "weighted_sum":
            return self._aggregate_weighted_sum(responses, batch_size, total_time)
        if self.aggregation == "mean":
            return self._aggregate_mean(responses, batch_size, total_time)
        if self.aggregation == "min":
            return self._aggregate_min(responses, batch_size, total_time)
        if self.aggregation == "max":
            return self._aggregate_max(responses, batch_size, total_time)
        if self.aggregation == "concat":
            return self._aggregate_concat(responses, batch_size, total_time)

        logger.warning("Unknown aggregation '%s', using weighted_sum", self.aggregation)
        return self._aggregate_weighted_sum(responses, batch_size, total_time)

    def _aggregate_weighted_sum(
        self,
        responses: List[Tuple[RewardResponse, BaseRewardExecutor]],
        batch_size: int,
        total_time: float,
    ) -> RewardResponse:
        total = torch.zeros(batch_size)
        total_weight = 0.0
        reward_components = {}

        for resp, executor in responses:
            weight = executor.get_weight()
            rewards_tensor = torch.tensor(resp.rewards)
            total += rewards_tensor * weight
            total_weight += weight
            reward_components[executor.get_model_name()] = resp.rewards

        final_rewards = (total / total_weight).tolist() if total_weight > 0 else total.tolist()

        all_successes = [True] * batch_size
        all_errors = [None] * batch_size
        for resp, _ in responses:
            for i, (success, error) in enumerate(zip(resp.successes, resp.errors)):
                if not success:
                    all_successes[i] = False
                    all_errors[i] = error

        return RewardResponse(
            rewards=final_rewards,
            reward_components=reward_components,
            successes=all_successes,
            errors=all_errors,
            compute_time=total_time,
        )

    def _aggregate_mean(
        self,
        responses: List[Tuple[RewardResponse, BaseRewardExecutor]],
        batch_size: int,
        total_time: float,
    ) -> RewardResponse:
        total = torch.zeros(batch_size)
        reward_components = {}

        for resp, executor in responses:
            rewards_tensor = torch.tensor(resp.rewards)
            total += rewards_tensor
            reward_components[executor.get_model_name()] = resp.rewards

        final_rewards = (total / len(responses)).tolist()

        all_successes = [True] * batch_size
        all_errors = [None] * batch_size
        for resp, _ in responses:
            for i, (success, error) in enumerate(zip(resp.successes, resp.errors)):
                if not success:
                    all_successes[i] = False
                    all_errors[i] = error

        return RewardResponse(
            rewards=final_rewards,
            reward_components=reward_components,
            successes=all_successes,
            errors=all_errors,
            compute_time=total_time,
        )

    def _aggregate_min(
        self,
        responses: List[Tuple[RewardResponse, BaseRewardExecutor]],
        batch_size: int,
        total_time: float,
    ) -> RewardResponse:
        all_rewards = torch.stack(
            [torch.tensor(resp.rewards) for resp, _ in responses]
        )
        final_rewards = all_rewards.min(dim=0)[0].tolist()
        reward_components = {
            executor.get_model_name(): resp.rewards
            for resp, executor in responses
        }

        all_successes = [True] * batch_size
        all_errors = [None] * batch_size
        for resp, _ in responses:
            for i, (success, error) in enumerate(zip(resp.successes, resp.errors)):
                if not success:
                    all_successes[i] = False
                    all_errors[i] = error

        return RewardResponse(
            rewards=final_rewards,
            reward_components=reward_components,
            successes=all_successes,
            errors=all_errors,
            compute_time=total_time,
        )

    def _aggregate_max(
        self,
        responses: List[Tuple[RewardResponse, BaseRewardExecutor]],
        batch_size: int,
        total_time: float,
    ) -> RewardResponse:
        all_rewards = torch.stack(
            [torch.tensor(resp.rewards) for resp, _ in responses]
        )
        final_rewards = all_rewards.max(dim=0)[0].tolist()
        reward_components = {
            executor.get_model_name(): resp.rewards
            for resp, executor in responses
        }

        all_successes = [True] * batch_size
        all_errors = [None] * batch_size
        for resp, _ in responses:
            for i, (success, error) in enumerate(zip(resp.successes, resp.errors)):
                if not success:
                    all_successes[i] = False
                    all_errors[i] = error

        return RewardResponse(
            rewards=final_rewards,
            reward_components=reward_components,
            successes=all_successes,
            errors=all_errors,
            compute_time=total_time,
        )

    def _aggregate_concat(
        self,
        responses: List[Tuple[RewardResponse, BaseRewardExecutor]],
        batch_size: int,
        total_time: float,
    ) -> RewardResponse:
        reward_components = {
            executor.get_model_name(): resp.rewards
            for resp, executor in responses
        }
        final_rewards = responses[0][0].rewards

        all_successes = [True] * batch_size
        all_errors = [None] * batch_size
        for resp, _ in responses:
            for i, (success, error) in enumerate(zip(resp.successes, resp.errors)):
                if not success:
                    all_successes[i] = False
                    all_errors[i] = error

        return RewardResponse(
            rewards=final_rewards,
            reward_components=reward_components,
            successes=all_successes,
            errors=all_errors,
            compute_time=total_time,
        )

    @property
    def preferred_input_kind(self) -> str:
        """Return the media kind expected by the configured executor set."""
        kinds = {
            str(getattr(executor, "preferred_input_kind", "image") or "image").strip().lower()
            for executor in self.executors
        }
        kinds.discard("")
        if not kinds:
            return "image"
        if len(kinds) > 1:
            raise ValueError(
                "Mixed reward input kinds in one service are not supported. "
                f"Configured kinds={sorted(kinds)}."
            )
        return next(iter(kinds))

    def is_available(self) -> bool:
        """Check if at least one executor is available."""
        return any(executor.is_available() for executor in self.executors)

    def offload(self) -> None:
        for executor in self.executors:
            executor.offload()
        logger.debug("RewardService offloaded %d executor(s)", len(self.executors))

    def onload(self) -> None:
        for executor in self.executors:
            executor.onload()
        logger.debug("RewardService onloaded %d executor(s)", len(self.executors))

    def dispose(self) -> None:
        for executor in self.executors:
            executor.dispose()
        self.executors = []
        logger.info("RewardService disposed")


class LocalRewardExecutor(RewardService):
    """Lightweight same-process reward service for rollout/training actors."""

    def __init__(
        self,
        reward_schema: RewardSchema,
        *,
        device_override: Optional[str] = None,
    ) -> None:
        if device_override is not None:
            reward_schema = replace(
                reward_schema,
                local_reward_device=str(device_override),
            )
        self._bind_reward_schema(
            reward_schema,
            reward_pg_result=None,
            owner_name="LocalRewardExecutor",
        )
        self._init_local_executors()
        logger.info(
            "LocalRewardExecutor initialized with %d executor(s), aggregation=%s, device=%s",
            len(self.executors),
            self.aggregation,
            self.execution_plan.local_device,
        )


__all__ = [
    "InProcessRewardExecutor",
    "RewardService",
    "LocalRewardExecutor",
]
