"""
RewardService - Unified abstraction for all reward computation.

Provides a single interface for:
- Local (CPU/GPU) rewards
- Remote (HTTP API) rewards
- Ray-based (GPU-isolated) rewards
- Multi-reward combinations with aggregation

"""

import inspect
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from .base import BaseRewardWorker, RewardRequest, RewardResponse
from diffusionrl.utils import load_function

logger = logging.getLogger(__name__)

# Type alias for placement group result
PlacementGroupResult = Tuple[Any, List[int], List[int]]  # (pg, bundle_indices, gpu_ids)


class RewardService:
    """
    Unified reward computation service.

    Automatically selects and configures appropriate workers based on
    configuration, and handles multi-reward aggregation.

    Backend selection priority:
    1. use_http_reward=True or reward_service_urls -> HTTPRewardWorker
    2. reward_dedicated_num_gpus > 0 -> RayRewardWorker (independent GPU)
    3. Otherwise -> LocalRewardWorker (CPU)

    Example usage:
        # Simple usage (auto-configured from args)
        service = RewardService(args, reward_pg_result=pgs.get("reward"))
        response = service.compute_rewards(request)

        # Multi-reward with weights
        args.reward_models = ["pickscore", "hpsv2"]
        args.reward_weights = [0.3, 0.7]
        service = RewardService(args, reward_pg_result=pgs.get("reward"))
        response = service.compute_rewards(request)  # Returns weighted average
    """

    def __init__(
        self,
        args,
        reward_pg_result: Optional[PlacementGroupResult] = None,
    ):
        """
        Initialize RewardService.

        Args:
            args: GRPOArguments instance with reward configuration
            reward_pg_result: Optional placement group result for GPU rewards
                             Tuple of (placement_group, bundle_indices, gpu_ids)
        """
        self.args = args
        self.reward_pg = reward_pg_result

        # Workers list (supports multiple for multi-reward)
        self.workers: List[BaseRewardWorker] = []

        # Aggregation configuration
        self.aggregation = getattr(args, "reward_aggregation", "weighted_sum")

        # Initialize workers based on configuration
        self._init_workers()

        logger.info(
            f"RewardService initialized with {len(self.workers)} worker(s), "
            f"aggregation={self.aggregation}"
        )

    def _init_workers(self) -> None:
        """Initialize workers based on args configuration."""
        # Parse multi-reward configuration
        reward_models = getattr(self.args, "reward_models", None)
        reward_weights = getattr(self.args, "reward_weights", None)
        reward_service_urls = getattr(self.args, "reward_service_urls", None)

        # Priority 1: Remote HTTP rewards
        if self.args.use_http_reward or reward_service_urls:
            self._init_http_workers(reward_service_urls)

        # Priority 2: GPU-isolated rewards (Ray workers)
        elif self._should_use_ray_workers():
            self._init_ray_workers(reward_models, reward_weights)

        # Priority 3: Local (CPU or same-process GPU)
        else:
            self._init_local_workers(reward_models, reward_weights)

    def _should_use_ray_workers(self) -> bool:
        """Check if Ray workers should be used."""
        # Use Ray workers if dedicated reward GPU pool is configured and we have a placement group
        reward_dedicated_num_gpus = getattr(self.args, "reward_dedicated_num_gpus", 0)
        reward_dedicated_num_nodes = getattr(self.args, "reward_dedicated_num_nodes", 0)

        has_gpu_config = reward_dedicated_num_gpus > 0 or reward_dedicated_num_nodes > 0
        has_pg = self.reward_pg is not None

        return has_gpu_config and has_pg

    def _init_http_workers(
        self,
        urls: Optional[List[str]] = None,
    ) -> None:
        """Initialize HTTP reward workers."""
        from .http import HTTPRewardWorker

        urls = urls or [self.args.reward_service_url]
        reward_weights = getattr(self.args, "reward_weights", None)

        for i, url in enumerate(urls):
            if url is None:
                continue

            weight = 1.0
            if reward_weights and i < len(reward_weights):
                weight = reward_weights[i]

            worker = HTTPRewardWorker(
                base_url=url,
                model_name=f"http_{i}",
                weight=weight,
                timeout=self.args.reward_timeout,
                batch_size=self.args.reward_batch_size,
            )
            self.workers.append(worker)
            logger.info(f"Added HTTPRewardWorker: {url}")

    def _init_ray_workers(
        self,
        reward_models: Optional[List[str]] = None,
        reward_weights: Optional[List[float]] = None,
    ) -> None:
        """Initialize Ray-based workers for GPU-isolated rewards."""
        from .ray_worker import RayRewardWorker

        pg, bundle_indices, gpu_ids = self.reward_pg
        gpus_per_actor = getattr(self.args, "reward_dedicated_gpus_per_actor", 1)

        if reward_models:
            # Multi-reward: each model gets its own actor(s)
            weights = reward_weights or [1.0] * len(reward_models)

            for i, model in enumerate(reward_models):
                weight = weights[i] if i < len(weights) else 1.0

                # Allocate GPUs for this model
                actor_start = i * gpus_per_actor
                actor_end = actor_start + gpus_per_actor

                if actor_end > len(bundle_indices):
                    logger.warning(
                        f"Not enough GPUs for model {model}. "
                        f"Required {gpus_per_actor}, available "
                        f"{len(bundle_indices) - actor_start}"
                    )
                    break

                worker = RayRewardWorker(
                    model_name=model,
                    pg=pg,
                    bundle_indices=bundle_indices[actor_start:actor_end],
                    gpu_ids=gpu_ids[actor_start:actor_end],
                    reward_path=self.args.reward_path,
                    model_path=getattr(self.args, "reward_model_path", None),
                    num_actors=1,
                    gpus_per_actor=gpus_per_actor,
                    batch_size=self.args.reward_batch_size,
                    timeout=self.args.reward_timeout,
                    parallel_mode=False,
                    weight=weight,
                )
                self.workers.append(worker)
                logger.info(f"Added RayRewardWorker: {model} (weight={weight})")

        else:
            # Single reward model: multiple actors process batch in parallel
            num_actors = len(gpu_ids) // gpus_per_actor

            worker = RayRewardWorker(
                model_name=self.args.reward_model_name,
                pg=pg,
                bundle_indices=bundle_indices,
                gpu_ids=gpu_ids,
                reward_path=self.args.reward_path,
                model_path=getattr(self.args, "reward_model_path", None),
                num_actors=num_actors,
                gpus_per_actor=gpus_per_actor,
                batch_size=self.args.reward_batch_size,
                timeout=self.args.reward_timeout,
                parallel_mode=True,  # Distribute batch across actors
                weight=1.0,
            )
            self.workers.append(worker)
            logger.info(
                f"Added RayRewardWorker: {self.args.reward_model_name} "
                f"({num_actors} parallel actors)"
            )

    def _init_local_workers(
        self,
        reward_models: Optional[List[str]] = None,
        reward_weights: Optional[List[float]] = None,
    ) -> None:
        """Initialize local workers (CPU or same-process GPU)."""
        from .local import LocalRewardWorker

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        reward_path = getattr(
            self.args,
            "reward_path",
            "diffusionrl.workers.reward.local.LocalRewardWorker",
        )
        try:
            worker_cls = load_function(reward_path) if reward_path else LocalRewardWorker
        except Exception as e:
            logger.warning(
                "Failed to load reward_path=%s (%s). Falling back to LocalRewardWorker.",
                reward_path,
                e,
            )
            worker_cls = LocalRewardWorker

        ctor_params = inspect.signature(worker_cls.__init__).parameters

        def _create_worker(model_name: str, weight: float) -> BaseRewardWorker:
            init_kwargs: Dict[str, Any] = {
                "weight": weight,
                "batch_size": self.args.reward_batch_size,
                "timeout": self.args.reward_timeout,
                "device": device,
            }
            if "model_name" in ctor_params:
                init_kwargs["model_name"] = model_name
            elif "frame_reward_model" in ctor_params:
                init_kwargs["frame_reward_model"] = model_name
            return worker_cls(**init_kwargs)

        if reward_models:
            # Multi-reward: create worker for each model
            weights = reward_weights or [1.0] * len(reward_models)

            for i, model in enumerate(reward_models):
                weight = weights[i] if i < len(weights) else 1.0

                worker = _create_worker(model_name=model, weight=weight)
                self.workers.append(worker)
                logger.info(
                    "Added local reward worker: %s via %s (weight=%s)",
                    model,
                    worker_cls.__name__,
                    weight,
                )

        else:
            # Single reward model
            worker = _create_worker(model_name=self.args.reward_model_name, weight=1.0)
            self.workers.append(worker)
            logger.info(
                "Added local reward worker: %s via %s",
                self.args.reward_model_name,
                worker_cls.__name__,
            )

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        """
        Compute rewards using configured workers.

        For multi-worker setups, rewards are aggregated according to
        the configured aggregation strategy.

        Args:
            request: RewardRequest with images/videos and prompts

        Returns:
            RewardResponse with computed (and possibly aggregated) rewards
        """
        if not self.workers:
            return RewardResponse(
                rewards=[0.0] * request.batch_size,
                successes=[False] * request.batch_size,
                errors=["No workers configured"] * request.batch_size,
                compute_time=0.0,
            )

        start_time = time.time()

        # Single worker: direct computation
        if len(self.workers) == 1:
            response = self.workers[0].compute_rewards(request)
            return response

        # Multiple workers: compute and aggregate
        responses = []
        for worker in self.workers:
            try:
                resp = worker.compute_rewards(request)
                responses.append((resp, worker))
            except Exception as e:
                logger.error(f"Worker {worker.get_model_name()} failed: {e}")
                # Create error response for failed worker
                error_resp = RewardResponse(
                    rewards=[0.0] * request.batch_size,
                    successes=[False] * request.batch_size,
                    errors=[str(e)] * request.batch_size,
                    compute_time=0.0,
                )
                responses.append((error_resp, worker))

        return self._aggregate_responses(responses, time.time() - start_time)

    def _aggregate_responses(
        self,
        responses: List[Tuple[RewardResponse, BaseRewardWorker]],
        total_time: float,
    ) -> RewardResponse:
        """
        Aggregate responses from multiple workers.

        Args:
            responses: List of (response, worker) tuples
            total_time: Total computation time

        Returns:
            Aggregated RewardResponse
        """
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
        elif self.aggregation == "mean":
            return self._aggregate_mean(responses, batch_size, total_time)
        elif self.aggregation == "min":
            return self._aggregate_min(responses, batch_size, total_time)
        elif self.aggregation == "max":
            return self._aggregate_max(responses, batch_size, total_time)
        elif self.aggregation == "concat":
            return self._aggregate_concat(responses, batch_size, total_time)
        else:
            logger.warning(f"Unknown aggregation '{self.aggregation}', using weighted_sum")
            return self._aggregate_weighted_sum(responses, batch_size, total_time)

    def _aggregate_weighted_sum(
        self,
        responses: List[Tuple[RewardResponse, BaseRewardWorker]],
        batch_size: int,
        total_time: float,
    ) -> RewardResponse:
        """Compute weighted sum of rewards."""
        total = torch.zeros(batch_size)
        total_weight = 0.0
        reward_components = {}

        for resp, worker in responses:
            weight = worker.get_weight()
            rewards_tensor = torch.tensor(resp.rewards)
            total += rewards_tensor * weight
            total_weight += weight

            # Store component
            model_name = worker.get_model_name()
            reward_components[model_name] = resp.rewards

        # Normalize by total weight
        if total_weight > 0:
            final_rewards = (total / total_weight).tolist()
        else:
            final_rewards = total.tolist()

        # Merge successes (all must succeed)
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
        responses: List[Tuple[RewardResponse, BaseRewardWorker]],
        batch_size: int,
        total_time: float,
    ) -> RewardResponse:
        """Compute mean of rewards (ignores weights)."""
        total = torch.zeros(batch_size)
        reward_components = {}

        for resp, worker in responses:
            rewards_tensor = torch.tensor(resp.rewards)
            total += rewards_tensor
            reward_components[worker.get_model_name()] = resp.rewards

        final_rewards = (total / len(responses)).tolist()

        # Merge successes
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
        responses: List[Tuple[RewardResponse, BaseRewardWorker]],
        batch_size: int,
        total_time: float,
    ) -> RewardResponse:
        """Take minimum reward across workers."""
        all_rewards = torch.stack(
            [torch.tensor(resp.rewards) for resp, _ in responses]
        )
        final_rewards = all_rewards.min(dim=0)[0].tolist()

        reward_components = {
            worker.get_model_name(): resp.rewards
            for resp, worker in responses
        }

        # Merge successes
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
        responses: List[Tuple[RewardResponse, BaseRewardWorker]],
        batch_size: int,
        total_time: float,
    ) -> RewardResponse:
        """Take maximum reward across workers."""
        all_rewards = torch.stack(
            [torch.tensor(resp.rewards) for resp, _ in responses]
        )
        final_rewards = all_rewards.max(dim=0)[0].tolist()

        reward_components = {
            worker.get_model_name(): resp.rewards
            for resp, worker in responses
        }

        # Merge successes
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
        responses: List[Tuple[RewardResponse, BaseRewardWorker]],
        batch_size: int,
        total_time: float,
    ) -> RewardResponse:
        """
        Return all rewards without aggregation.

        Uses first worker's rewards as the primary, stores all in components.
        """
        reward_components = {
            worker.get_model_name(): resp.rewards
            for resp, worker in responses
        }

        # Use first worker's rewards as primary
        final_rewards = responses[0][0].rewards

        # Merge successes
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

    def is_available(self) -> bool:
        """Check if at least one worker is available."""
        return any(worker.is_available() for worker in self.workers)

    def offload(self) -> None:
        """Offload all workers."""
        for worker in self.workers:
            worker.offload()
        logger.debug(f"RewardService offloaded {len(self.workers)} worker(s)")

    def onload(self) -> None:
        """Onload all workers."""
        for worker in self.workers:
            worker.onload()
        logger.debug(f"RewardService onloaded {len(self.workers)} worker(s)")

    def dispose(self) -> None:
        """Clean up all workers."""
        for worker in self.workers:
            worker.dispose()
        self.workers = []
        logger.info("RewardService disposed")

    def get_worker_info(self) -> List[Dict[str, Any]]:
        """Get information about configured workers."""
        return [
            {
                "type": type(worker).__name__,
                "model_name": worker.get_model_name(),
                "weight": worker.get_weight(),
                "available": worker.is_available(),
            }
            for worker in self.workers
        ]
