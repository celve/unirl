"""Ray reward executor backend for GPU-isolated reward computation."""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import ray

from diffusionrl.ray.group_base import PlacementGroupActorPool
from .base import BaseRewardExecutor, BaseRewardScorer, RewardRequest, RewardResponse
from .scorers.registry import (
    resolve_builtin_reward_scorer_class,
    resolve_builtin_reward_scorer_path,
)
from diffusionrl.utils import load_function

logger = logging.getLogger(__name__)


def _resolve_scorer_path_and_class(
    *,
    reward_dotpath: Optional[str],
    model_name: str,
) -> tuple[str, type]:
    """Resolve scorer path/class from explicit path or built-in model name."""
    if reward_dotpath:
        scorer_path = str(reward_dotpath)
        scorer_cls = load_function(scorer_path)
    else:
        scorer_path = resolve_builtin_reward_scorer_path(model_name)
        scorer_cls = resolve_builtin_reward_scorer_class(model_name)
    return scorer_path, scorer_cls


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
        reward_dotpath: str,
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
            reward_dotpath: Python path to reward scorer class
            model_name: Name of the reward model
            device_id: GPU device ID to use
            gpus_per_actor: Number of GPUs for this actor (for large models)
            model_path: Optional path to model weights
            batch_size: Maximum batch size for processing
            timeout: Timeout for reward computation
            **kwargs: Additional arguments for reward scorer
        """
        import torch

        self.model_name = model_name
        self.device_id = device_id
        self.gpus_per_actor = gpus_per_actor

        # Set CUDA visible devices for single-GPU actors
        if gpus_per_actor == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
            logger.info(f"_RewardActor using GPU {device_id}")

        # Load the reward scorer.
        scorer_cls = load_function(reward_dotpath)

        init_kwargs = {
            "model_name": model_name,
            "batch_size": batch_size,
            "timeout": timeout,
            "device": "cuda",  # Use CUDA since we've set CUDA_VISIBLE_DEVICES
        }

        if model_path:
            init_kwargs["model_path"] = model_path

        init_kwargs.update(kwargs)

        self.scorer = scorer_cls(**init_kwargs)
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
        return self.scorer.compute_rewards(request)

    def offload(self) -> None:
        """Offload model from GPU to CPU."""
        if hasattr(self.scorer, "offload"):
            self.scorer.offload()
            logger.debug(f"_RewardActor offloaded: {self.model_name}")

    def onload(self) -> None:
        """Load model back to GPU."""
        if hasattr(self.scorer, "onload"):
            self.scorer.onload()
            logger.debug(f"_RewardActor onloaded: {self.model_name}")

    def is_available(self) -> bool:
        """Check if the actor is available."""
        return self._is_available and self.scorer.is_available()

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model_name


class RayRewardExecutor(BaseRewardExecutor):
    """
    Ray-based reward executor for GPU-isolated reward computation.

    Uses Ray actors and placement groups to isolate GPU memory
    from rollout and training actors.

    **Only used for independent GPU mode** (reward_dedicated_num_gpus > 0).
    Colocate mode uses RolloutActor built-in reward instead.

    Supports two modes:
    1. parallel_mode=False: Single actor (or multi-GPU actor) handles all requests
    2. parallel_mode=True: Multiple actors process batch chunks in parallel

    Example usage:
        # Single actor mode
        executor = RayRewardExecutor(
            model_name="hpsv2",
            pg=placement_group,
            bundle_indices=[0],
            gpu_ids=[0],
        )

        # Parallel mode (distribute batch across actors)
        executor = RayRewardExecutor(
            model_name="hpsv2",
            pg=placement_group,
            bundle_indices=[0, 1, 2, 3],
            gpu_ids=[0, 1, 2, 3],
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
        reward_dotpath: Optional[str] = None,
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
        Initialize Ray reward executor.

        Args:
            model_name: Name of the reward model
            pg: Ray placement group for resource allocation
            bundle_indices: Placement group bundle indices for each actor
            gpu_ids: GPU IDs for each actor
            reward_dotpath: Optional Python path to a custom reward scorer class.
                When omitted, built-in scorers are resolved from model_name.
            model_path: Optional path to model weights
            num_actors: Number of reward worker actors to create
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
        self.reward_dotpath = reward_dotpath
        self.scorer_path, scorer_cls = _resolve_scorer_path_and_class(
            reward_dotpath=reward_dotpath,
            model_name=model_name,
        )
        self.model_path = model_path
        self.num_actors = num_actors
        self.gpus_per_actor = gpus_per_actor
        self.parallel_mode = parallel_mode
        self.extra_kwargs = kwargs
        self.input_kind = self._resolve_input_kind(
            scorer_cls=scorer_cls,
            scorer_path=self.scorer_path,
        )

        self._actor_pool: Optional[PlacementGroupActorPool] = None
        self._is_initialized = False

        # Create actor pool
        self._create_actor_pool()

    @staticmethod
    def _resolve_input_kind(*, scorer_cls: type, scorer_path: str) -> str:
        if not isinstance(scorer_cls, type) or not issubclass(scorer_cls, BaseRewardScorer):
            logger.warning(
                "Ray reward scorer %s does not inherit BaseRewardScorer; "
                "treating it as a scorer via duck typing.",
                scorer_path,
            )
        input_kind = str(getattr(scorer_cls, "input_kind", "image") or "image").strip().lower()
        if input_kind not in {"image", "video"}:
            raise ValueError(
                "Reward scorer class must declare input_kind as 'image' or 'video'. "
                f"Got {input_kind!r} for reward_dotpath={scorer_path!r}."
            )
        return input_kind

    def _actor_handles(self) -> List[ray.actor.ActorHandle]:
        if self._actor_pool is None:
            return []
        return self._actor_pool.get_actors()

    def _create_actor_pool(self) -> None:
        """Create reward actors on the placement group via a generic actor pool."""
        try:
            per_actor_kwargs: List[Dict[str, Any]] = []
            for i in range(self.num_actors):
                per_actor_kwargs.append(
                    {
                        "device_id": self.gpu_ids[i * self.gpus_per_actor],
                        "gpus_per_actor": self.gpus_per_actor,
                    }
                )

            self._actor_pool = PlacementGroupActorPool(
                actor_class=_RewardActor,
                num_actors=self.num_actors,
                pg=self.pg,
                bundle_indices=self.bundle_indices,
                gpu_ids=self.gpu_ids,
                num_gpus_per_actor=float(self.gpus_per_actor),
                num_gpus_per_engine=self.gpus_per_actor,
                per_actor_kwargs=per_actor_kwargs,
                reward_dotpath=self.scorer_path,
                model_name=self.model_name,
                model_path=self.model_path,
                batch_size=self.batch_size,
                timeout=self.timeout,
                **self.extra_kwargs,
            )

            self._is_initialized = True
            logger.info(
                f"RayRewardExecutor created {self.num_actors} actors "
                f"(parallel_mode={self.parallel_mode})"
            )

        except Exception as e:
            logger.error(f"Failed to create RayRewardExecutor actors: {e}")
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
        actor_handles = self._actor_handles()
        if not self._is_initialized or not actor_handles:
            return RewardResponse(
                rewards=[0.0] * request.batch_size,
                successes=[False] * request.batch_size,
                errors=["Reward actor pool not initialized"] * request.batch_size,
                compute_time=0.0,
            )

        start_time = time.time()

        if self.parallel_mode and len(actor_handles) > 1:
            response = self._compute_parallel(request)
        else:
            # Single actor mode
            response = self._actor_pool.call_rank0("compute_rewards", request)

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
        actor_handles = self._actor_handles()
        num_actors = len(actor_handles)
        chunk_size = (batch_size + num_actors - 1) // num_actors

        # Split request into chunks
        shards: List[Optional[RewardRequest]] = [None] * num_actors
        for i in range(num_actors):
            start = i * chunk_size
            end = min(start + chunk_size, batch_size)
            if start >= batch_size:
                break

            # Create chunk request
            shards[i] = RewardRequest(
                images=request.images[start:end] if request.images else None,
                videos=request.videos[start:end] if request.videos else None,
                prompts=request.prompts[start:end],
                prompt_ids=request.prompt_ids[start:end] if request.prompt_ids else None,
                sample_ids=request.sample_ids[start:end] if request.sample_ids else None,
                group_ids=request.group_ids[start:end] if request.group_ids else None,
                metadata=request.metadata[start:end] if request.metadata else None,
                reward_types=request.reward_types,
                return_components=request.return_components,
            )

        # Gather results
        responses = self._actor_pool.scatter_gather("compute_rewards", shards)

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
            for key, values in resp.component_rewards.items():
                if key not in merged_components:
                    merged_components[key] = []
                merged_components[key].extend(values)

        return RewardResponse(
            rewards=all_rewards,
            component_rewards=merged_components,
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
        if not self._is_initialized or self._actor_pool is None:
            return False

        try:
            availabilities = self._actor_pool.call_all("is_available")
            return all(availabilities)
        except Exception:
            return False

    def offload(self) -> None:
        """Offload all actors' models from GPU."""
        actor_handles = self._actor_handles()
        if actor_handles and self._actor_pool is not None:
            self._actor_pool.call_all("offload")
            logger.debug(f"RayRewardExecutor offloaded {len(actor_handles)} actors")

    def onload(self) -> None:
        """Load all actors' models back to GPU."""
        actor_handles = self._actor_handles()
        if actor_handles and self._actor_pool is not None:
            self._actor_pool.call_all("onload")
            logger.debug(f"RayRewardExecutor onloaded {len(actor_handles)} actors")

    def dispose(self) -> None:
        """Kill all actors and clean up."""
        if self._actor_pool is not None:
            self._actor_pool.dispose()
            self._actor_pool = None
        self._is_initialized = False
        logger.info("RayRewardExecutor disposed")


__all__ = [
    "RayRewardExecutor",
]
