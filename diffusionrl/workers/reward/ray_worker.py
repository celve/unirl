"""
Ray-based reward worker for GPU-isolated reward computation.

Uses Ray actors and placement groups to isolate GPU memory
from inference and training actors.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .base import BaseRewardWorker, RewardRequest, RewardResponse
from diffusionrl.utils import load_function

logger = logging.getLogger(__name__)


@ray.remote
class _RewardActor:
    """
    Internal Ray actor for isolated GPU reward computation.

    Runs on a dedicated GPU (or set of GPUs for large models)
    managed by a placement group.

    This actor:
    - Loads the reward model on its dedicated GPU(s)
    - Computes rewards for incoming requests
    - Supports offload/onload for memory management
    """

    def __init__(
        self,
        reward_path: str,
        model_name: str,
        device_id: int = 0,
        gpus_per_actor: int = 1,
        model_path: Optional[str] = None,
        batch_size: int = 8,
        timeout: float = 60.0,
        **kwargs,
    ):
        """
        Initialize reward actor.

        Args:
            reward_path: Python path to reward worker class
            model_name: Name of the reward model
            device_id: GPU device ID to use
            gpus_per_actor: Number of GPUs for this actor (for large models)
            model_path: Optional path to model weights
            batch_size: Maximum batch size for processing
            timeout: Timeout for reward computation
            **kwargs: Additional arguments for reward worker
        """
        import torch

        self.model_name = model_name
        self.device_id = device_id
        self.gpus_per_actor = gpus_per_actor

        # Set CUDA visible devices for single-GPU actors
        if gpus_per_actor == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
            logger.info(f"_RewardActor using GPU {device_id}")

        # Load the reward worker
        worker_cls = load_function(reward_path)

        init_kwargs = {
            "model_name": model_name,
            "batch_size": batch_size,
            "timeout": timeout,
            "device": "cuda",  # Use CUDA since we've set CUDA_VISIBLE_DEVICES
        }

        if model_path:
            init_kwargs["model_path"] = model_path

        init_kwargs.update(kwargs)

        self.worker = worker_cls(**init_kwargs)
        self._is_available = True
        logger.info(f"_RewardActor initialized: {model_name} on GPU {device_id}")

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        """
        Compute rewards for the given request.

        Args:
            request: RewardRequest with images/videos and prompts

        Returns:
            RewardResponse with computed rewards
        """
        return self.worker.compute_rewards(request)

    def offload(self) -> None:
        """Offload model from GPU to CPU."""
        if hasattr(self.worker, "offload"):
            self.worker.offload()
            logger.debug(f"_RewardActor offloaded: {self.model_name}")

    def onload(self) -> None:
        """Load model back to GPU."""
        if hasattr(self.worker, "onload"):
            self.worker.onload()
            logger.debug(f"_RewardActor onloaded: {self.model_name}")

    def is_available(self) -> bool:
        """Check if the actor is available."""
        return self._is_available and self.worker.is_available()

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model_name


class RayRewardWorker(BaseRewardWorker):
    """
    Ray-based reward worker for GPU-isolated reward computation.

    Uses Ray actors and placement groups to isolate GPU memory
    from inference and training actors.

    **Only used for independent GPU mode** (reward_dedicated_num_gpus > 0).
    Colocate mode uses InferenceActor built-in reward instead.

    Supports two modes:
    1. parallel_mode=False: Single actor (or multi-GPU actor) handles all requests
    2. parallel_mode=True: Multiple actors process batch chunks in parallel

    Example usage:
        # Single actor mode
        worker = RayRewardWorker(
            model_name="hpsv2",
            pg=placement_group,
            bundle_indices=[0],
            gpu_ids=[0],
            reward_path="diffusionrl.workers.reward.local.LocalRewardWorker",
        )

        # Parallel mode (distribute batch across actors)
        worker = RayRewardWorker(
            model_name="hpsv2",
            pg=placement_group,
            bundle_indices=[0, 1, 2, 3],
            gpu_ids=[0, 1, 2, 3],
            reward_path="diffusionrl.workers.reward.local.LocalRewardWorker",
            num_actors=4,
            parallel_mode=True,
        )
    """

    def __init__(
        self,
        model_name: str,
        pg,  # PlacementGroup
        bundle_indices: List[int],
        gpu_ids: List[int],
        reward_path: str = "diffusionrl.workers.reward.local.LocalRewardWorker",
        model_path: Optional[str] = None,
        num_actors: int = 1,
        gpus_per_actor: int = 1,
        batch_size: int = 8,
        timeout: float = 60.0,
        parallel_mode: bool = False,
        weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize Ray reward worker.

        Args:
            model_name: Name of the reward model
            pg: Ray placement group for resource allocation
            bundle_indices: Placement group bundle indices for each actor
            gpu_ids: GPU IDs for each actor
            reward_path: Python path to reward worker class
            model_path: Optional path to model weights
            num_actors: Number of actors to create
            gpus_per_actor: Number of GPUs per actor (for large models)
            batch_size: Maximum batch size per actor
            timeout: Timeout for reward computation
            parallel_mode: If True, distribute batch across actors
            weight: Weight for multi-reward aggregation
            **kwargs: Additional arguments for reward actors
        """
        super().__init__(
            model_name=model_name,
            weight=weight,
            batch_size=batch_size,
            timeout=timeout,
        )

        self.pg = pg
        self.bundle_indices = bundle_indices
        self.gpu_ids = gpu_ids
        self.reward_path = reward_path
        self.model_path = model_path
        self.num_actors = num_actors
        self.gpus_per_actor = gpus_per_actor
        self.parallel_mode = parallel_mode
        self.extra_kwargs = kwargs

        self.actors: List[ray.actor.ActorHandle] = []
        self._is_initialized = False

        # Create actors
        self._create_actors()

    def _create_actors(self) -> None:
        """Create reward actors on the placement group."""
        try:
            for i in range(self.num_actors):
                bundle_idx = self.bundle_indices[i * self.gpus_per_actor]
                gpu_id = self.gpu_ids[i * self.gpus_per_actor]

                actor_options = {
                    "num_gpus": self.gpus_per_actor,
                    "scheduling_strategy": PlacementGroupSchedulingStrategy(
                        placement_group=self.pg,
                        placement_group_bundle_index=bundle_idx,
                    ),
                }

                actor = _RewardActor.options(**actor_options).remote(
                    reward_path=self.reward_path,
                    model_name=self.model_name,
                    device_id=gpu_id,
                    gpus_per_actor=self.gpus_per_actor,
                    model_path=self.model_path,
                    batch_size=self.batch_size,
                    timeout=self.timeout,
                    **self.extra_kwargs,
                )
                self.actors.append(actor)

            self._is_initialized = True
            logger.info(
                f"RayRewardWorker created {self.num_actors} actors "
                f"(parallel_mode={self.parallel_mode})"
            )

        except Exception as e:
            logger.error(f"Failed to create RayRewardWorker actors: {e}")
            self._is_initialized = False
            raise

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        """
        Compute rewards using Ray actors.

        If parallel_mode is True and multiple actors exist, the batch
        is split and processed in parallel.

        Args:
            request: RewardRequest with images/videos and prompts

        Returns:
            RewardResponse with computed rewards
        """
        if not self._is_initialized or not self.actors:
            return RewardResponse(
                rewards=[0.0] * request.batch_size,
                successes=[False] * request.batch_size,
                errors=["Worker not initialized"] * request.batch_size,
                compute_time=0.0,
            )

        start_time = time.time()

        if self.parallel_mode and len(self.actors) > 1:
            response = self._compute_parallel(request)
        else:
            # Single actor mode
            response = ray.get(self.actors[0].compute_rewards.remote(request))

        # Update compute time to include Ray overhead
        response.compute_time = time.time() - start_time
        return response

    def _compute_parallel(self, request: RewardRequest) -> RewardResponse:
        """
        Distribute batch across multiple actors and compute in parallel.

        Args:
            request: RewardRequest with images/videos and prompts

        Returns:
            Merged RewardResponse
        """
        batch_size = request.batch_size
        num_actors = len(self.actors)
        chunk_size = (batch_size + num_actors - 1) // num_actors

        # Split request into chunks
        futures = []
        for i, actor in enumerate(self.actors):
            start = i * chunk_size
            end = min(start + chunk_size, batch_size)
            if start >= batch_size:
                break

            # Create chunk request
            chunk_request = RewardRequest(
                images=request.images[start:end] if request.images else None,
                videos=request.videos[start:end] if request.videos else None,
                prompts=request.prompts[start:end],
                metadata=request.metadata[start:end] if request.metadata else None,
                reward_types=request.reward_types,
                return_components=request.return_components,
            )
            futures.append(actor.compute_rewards.remote(chunk_request))

        # Gather results
        responses = ray.get(futures)

        # Merge responses
        return self._merge_responses(responses)

    def _merge_responses(self, responses: List[RewardResponse]) -> RewardResponse:
        """Merge multiple RewardResponses into one."""
        all_rewards = []
        all_successes = []
        all_errors = []
        max_compute_time = 0.0
        merged_components: Dict[str, List[float]] = {}

        for resp in responses:
            all_rewards.extend(resp.rewards)
            all_successes.extend(resp.successes)
            all_errors.extend(resp.errors)
            max_compute_time = max(max_compute_time, resp.compute_time)

            # Merge reward components
            for key, values in resp.reward_components.items():
                if key not in merged_components:
                    merged_components[key] = []
                merged_components[key].extend(values)

        return RewardResponse(
            rewards=all_rewards,
            reward_components=merged_components,
            successes=all_successes,
            errors=all_errors,
            compute_time=max_compute_time,
        )

    async def compute_rewards_async(self, request: RewardRequest) -> RewardResponse:
        """
        Compute rewards asynchronously.

        For Ray actors, this is essentially the same as sync since
        Ray handles the async communication.

        Args:
            request: RewardRequest with images/videos and prompts

        Returns:
            RewardResponse with computed rewards
        """
        return self.compute_rewards(request)

    def is_available(self) -> bool:
        """Check if all actors are available."""
        if not self._is_initialized or not self.actors:
            return False

        try:
            availabilities = ray.get(
                [actor.is_available.remote() for actor in self.actors]
            )
            return all(availabilities)
        except Exception:
            return False

    def offload(self) -> None:
        """Offload all actors' models from GPU."""
        if self.actors:
            ray.get([actor.offload.remote() for actor in self.actors])
            logger.debug(f"RayRewardWorker offloaded {len(self.actors)} actors")

    def onload(self) -> None:
        """Load all actors' models back to GPU."""
        if self.actors:
            ray.get([actor.onload.remote() for actor in self.actors])
            logger.debug(f"RayRewardWorker onloaded {len(self.actors)} actors")

    def dispose(self) -> None:
        """Kill all actors and clean up."""
        for actor in self.actors:
            try:
                ray.kill(actor)
            except Exception as e:
                logger.warning(f"Failed to kill actor: {e}")

        self.actors = []
        self._is_initialized = False
        logger.info("RayRewardWorker disposed")
