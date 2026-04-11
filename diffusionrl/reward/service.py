"""Reward executors and aggregation helpers."""

from __future__ import annotations

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

    def __init__(
        self,
        reward_schema: RewardSchema,
    ) -> None:
        if not isinstance(reward_schema, RewardSchema):
            raise TypeError(
                f"RewardService requires RewardSchema, "
                f"got: {type(reward_schema).__name__}"
            )
        self.reward_schema = reward_schema
        self.reward_definition = reward_schema.to_definition()
        self.reward_provider = reward_schema.to_provider_config()
        self.execution_plan = reward_schema.to_execution_plan()
        self.executors = []
        self.reward_aggregation_method = (
            self.reward_definition.reward_aggregation_method
        )

        self._init_executors()

        logger.info(
            "RewardService initialized with %d executor(s), aggregation=%s",
            len(self.executors),
            self.reward_aggregation_method,
        )

    def _init_executors(self) -> None:
        """Initialize executors based on execution plan."""
        if self.execution_plan.uses_http_backend:
            self._init_http_executors()
        else:
            self._init_local_executors()

    def _init_http_executors(self) -> None:
        """Initialize HTTP reward executors."""
        from .http import HTTPRewardExecutor

        urls = list(self.execution_plan.reward_service_urls or ())
        component_names = self.reward_definition.component_names
        component_weights = self.reward_definition.component_weights_list

        for i, url in enumerate(urls):
            if url is None:
                continue

            weight = 1.0
            if component_weights and i < len(component_weights):
                weight = component_weights[i]

            name = component_names[i] if component_names and i < len(component_names) else f"http_{i}"

            executor = HTTPRewardExecutor(
                base_url=url,
                model_name=name,
                weight=weight,
                timeout=self.reward_provider.timeout,
                batch_size=self.reward_provider.batch_size,
            )
            self.executors.append(executor)
            logger.info("Added HTTPRewardExecutor: %s (name=%s)", url, name)

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
                "This can contend with sampling GPUs. Prefer HTTP reward service "
                "for isolation when you need strict resource isolation."
            )

        reward_dotpath = getattr(self.reward_provider, "reward_dotpath", None)
        reward_model_ckpt_path = getattr(
            self.reward_provider,
            "reward_model_ckpt_path",
            None,
        )

        def _create_executor(model_name: str, weight: float) -> BaseRewardExecutor:
            if reward_dotpath:
                scorer_cls = load_function(reward_dotpath)
            else:
                scorer_cls = resolve_builtin_reward_scorer_class(model_name)

            if not isinstance(scorer_cls, type) or not issubclass(scorer_cls, BaseRewardScorer):
                logger.warning(
                    "Local reward scorer %s does not inherit BaseRewardScorer; "
                    "treating it as a scorer via duck typing.",
                    reward_dotpath or scorer_cls,
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
            if reward_model_ckpt_path:
                if "reward_model_ckpt_path" in ctor_params:
                    init_kwargs["reward_model_ckpt_path"] = reward_model_ckpt_path
                elif "checkpoint_path" in ctor_params:
                    init_kwargs["checkpoint_path"] = reward_model_ckpt_path
            if "weight" in ctor_params:
                init_kwargs["weight"] = weight
            scorer = scorer_cls(**init_kwargs)
            return InProcessRewardExecutor(
                scorer=scorer,
                weight=weight,
            )

        component_names = self.reward_definition.component_names
        component_weights = self.reward_definition.component_weights_list

        if component_names:
            weights = component_weights or [1.0] * len(component_names)

            for i, model in enumerate(component_names):
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

        if self.reward_aggregation_method == "weighted_sum":
            return self._aggregate_weighted_sum(responses, batch_size, total_time)
        if self.reward_aggregation_method == "mean":
            return self._aggregate_mean(responses, batch_size, total_time)
        if self.reward_aggregation_method == "min":
            return self._aggregate_min(responses, batch_size, total_time)
        if self.reward_aggregation_method == "max":
            return self._aggregate_max(responses, batch_size, total_time)
        if self.reward_aggregation_method == "concat":
            return self._aggregate_concat(responses, batch_size, total_time)

        logger.warning(
            "Unknown aggregation '%s', using weighted_sum",
            self.reward_aggregation_method,
        )
        return self._aggregate_weighted_sum(responses, batch_size, total_time)

    def _aggregate_weighted_sum(
        self,
        responses: List[Tuple[RewardResponse, BaseRewardExecutor]],
        batch_size: int,
        total_time: float,
    ) -> RewardResponse:
        total = torch.zeros(batch_size)
        total_weight = 0.0
        component_rewards = {}

        for resp, executor in responses:
            weight = executor.get_weight()
            rewards_tensor = torch.tensor(resp.rewards)
            total += rewards_tensor * weight
            total_weight += weight
            component_rewards[executor.get_model_name()] = resp.rewards

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
            component_rewards=component_rewards,
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
        component_rewards = {}

        for resp, executor in responses:
            rewards_tensor = torch.tensor(resp.rewards)
            total += rewards_tensor
            component_rewards[executor.get_model_name()] = resp.rewards

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
            component_rewards=component_rewards,
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
        component_rewards = {
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
            component_rewards=component_rewards,
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
        component_rewards = {
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
            component_rewards=component_rewards,
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
        component_rewards = {
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
            component_rewards=component_rewards,
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


__all__ = [
    "InProcessRewardExecutor",
    "RewardService",
]
