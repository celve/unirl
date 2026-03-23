"""diffusionrl Rollout Manager - rollout-side producer facade."""
from dataclasses import replace
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import ray
import torch

from diffusionrl.config.build_domain_args import (
    build_sampling_config,
)
from diffusionrl.config.resolution import resolve_prompts_per_rollout, resolve_sampling_requirements
from diffusionrl.orchestration import EvalRunner
from diffusionrl.orchestration.request_builder import (
    RolloutRequestBuilder,
    SampledRequestResult,
)
from diffusionrl.orchestration.rollout_workflow import (
    RolloutWorkflow,
    distributed_sample,
)
from diffusionrl.reward.factory import create_manager_reward_executor
from diffusionrl.reward.schema import RewardSchema
from diffusionrl.types.buffer_contracts import RolloutPayload
from diffusionrl.types.sampling import RolloutRequest
from diffusionrl.types.training_batch import TrainingBatch
from diffusionrl.utils import load_function

logger = logging.getLogger(__name__)


@ray.remote
class RolloutManager:
    """
    Rollout-side local runtime owner and public producer facade.

    This actor owns rollout-local services and state:
    - dynamic loading of algorithm, data source, and reward runtime
    - sampling actor-group attachment
    - rollout/eval public entrypoints exposed to the driver
    - rollout-side counters and last-produced metadata

    The rollout business chain itself lives in ``RolloutWorkflow``. This
    manager wires actor-local services into that workflow through a small
    number of explicit seams, then finalizes rollout-local bookkeeping.
    """

    def __init__(  # [PUBLIC-API → create_rollout_manager()] 构造，仅存配置，不初始化组件
        self,
        args,
        reward_pg_result: Optional[Tuple] = None,
        *,
        algorithm_config: Dict[str, Any],
    ):
        """
        Initialize RolloutManager.

        Args:
            args: TrainingArguments instance
            reward_pg_result: Optional placement group result for reward (pg, bundle_indices, gpu_ids)
        """
        self.args = args
        self.reward_pg_result = reward_pg_result
        self._is_initialized = False

        # Components (loaded in init())
        self.algorithm = None
        self.reward_service = None
        self.data_source = None

        self._reward_schema = RewardSchema.from_args(args)
        self.eval_runner = None
        self.rollout_workflow = None

        self.sampling_group = None

        # Stats
        self._total_samples_generated = 0
        self._sampling_requirements = None
        self._last_rollout_metadata: Dict[str, Any] = {}
        self._sampling_config = build_sampling_config(args)
        self._request_builder = RolloutRequestBuilder.from_args(
            args,
            sampling_defaults=self._sampling_config,
        )
        if not isinstance(algorithm_config, dict):
            raise ValueError("RolloutManager requires a non-empty algorithm_config dict from the driver.")
        self._algorithm_config = dict(algorithm_config)
        self._prompt_batch_size = int(resolve_prompts_per_rollout(self.args))

    def init(self) -> None:  # [PUBLIC-API → create_rollout_manager()] 初始化全部子组件
        """
        Initialize all components via dynamic loading.

        Components are loaded using load_function with paths from args.
        """
        logger.info("Initializing RolloutManager...")

        # 1. Load algorithm
        algorithm_path = str(self._algorithm_config["algorithm_path"])
        algorithm_cls = load_function(algorithm_path)
        if not hasattr(algorithm_cls, "from_config"):
            raise TypeError(
                f"Algorithm {algorithm_path} must implement classmethod from_config(config)."
            )
        self.algorithm = algorithm_cls.from_config(self._algorithm_config)
        self._sampling_requirements = resolve_sampling_requirements(self.args, algorithm=self.algorithm)
        logger.info(
            f"Algorithm loaded: {algorithm_path} "
            f"(clip_max={self.args.algorithm.adv_clip_abs}, "
            f"sde_ratio={dict(self._algorithm_config.get('sde_schedule_config') or {}).get('sde_ratio', 'N/A')})"
        )
        requests_per_rollout = self._request_builder.estimate_request_batches(
            prompt_count=self._prompt_batch_size,
            samples_per_prompt=int(self.algorithm.samples_per_prompt),
        )
        if requests_per_rollout > 1:
            logger.info(
                "Training-actor direct sampling sub-batching enabled: "
                "max_samples_per_request=%s sampling_requests_per_rollout=%s "
                "(prompts_per_rollout=%s generated_samples_per_rollout=%s)",
                getattr(self._request_builder, "max_samples_per_request", None),
                requests_per_rollout,
                self._prompt_batch_size,
                self._generated_samples_per_rollout(),
            )
        logger.info(
            "Resolved sampling contract: requires_trajectory=%s requires_log_prob=%s "
            "requires_embeddings=%s extras=%s",
            bool(self._sampling_requirements.requires_trajectory),
            bool(self._sampling_requirements.requires_log_prob),
            bool(self._sampling_requirements.requires_embeddings),
            dict(getattr(self._sampling_requirements, "extras", {}) or {}),
        )

        # 2. Initialize reward runtime
        self._init_reward_service()
        if self.reward_service is None:
            if self._reward_schema.uses_sampling_actor_execution:
                logger.info("Manager reward runtime skipped: sampling-actor-local reward execution is active.")
            else:
                raise RuntimeError("Manager reward runtime initialization failed.")
        else:
            logger.info(f"Manager reward runtime loaded with {len(self.reward_service.executors)} executor(s)")

        # 3. Load data source if available
        try:
            data_source_cls = load_function(self.args.data_source_path)
            self.data_source = data_source_cls(self.args)
            logger.info(f"Data source loaded: {self.args.data_source_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load data source: {e}") from e

        reward_scoring_mode = (
            "precomputed"
            if self._reward_schema.uses_sampling_actor_execution
            else "service"
        )
        self.eval_runner = EvalRunner(
            args=self.args,
            sampling_config=self._sampling_config,
            data_source=self.data_source,
            reward_scoring_mode=reward_scoring_mode,
            reward_service=self.reward_service,
            algorithm=self.algorithm,
            default_prompt_batch_fn=lambda: self._prepare_batch(data_source=self.data_source),
        )
        self.rollout_workflow = RolloutWorkflow(
            args=self.args,
            algorithm=self.algorithm,
            reward_scoring_mode=reward_scoring_mode,
            reward_service=self.reward_service,
            request_builder=self._request_builder,
            reward_component_weights=self._reward_schema.component_weights(),
            load_prompt_batch_fn=lambda: self._prepare_batch(data_source=self.data_source),
        )

        self._is_initialized = True
        logger.info("RolloutManager initialized")

    def _init_reward_service(self) -> None:  # [INTERNAL → init()]
        """Initialize reward service (single reward boundary)."""
        self.reward_service = create_manager_reward_executor(
            self._reward_schema,
            reward_pg_result=self.reward_pg_result,
        )

    def attach_sampling_group(self, actor_group) -> None:  # [PUBLIC-API → train.py]
        """Attach the sampling actor group used for rollout/eval generation."""
        self.sampling_group = actor_group
        logger.info("Sampling actor group attached")

    def build_training_batch(  # [PUBLIC-API → internal orchestration, debug] rollout batch entrypoint
        self,
        rollout_id: int,
        *,
        sde_indices: Optional[Set[int]] = None,
        collect_media_preview: bool = False,
        media_max_items: int = 8,
        debug_trace: Optional[Dict[str, Any]] = None,
    ) -> TrainingBatch:
        """Produce one typed TrainingBatch from the rollout-side producer path.

        Reading order:
        - RolloutManager owns actor-local wiring and public entrypoints.
        - RolloutWorkflow owns the readable sample -> reward -> advantage ->
          assemble business chain.
        - _execute_sampling_request is the actor-local sampling seam injected
          into the workflow.
        - this method finalizes rollout-local metadata and counters.
        """
        if not self._is_initialized:
            raise RuntimeError("RolloutManager not initialized. Call init() first.")
        if self.rollout_workflow is None:
            raise RuntimeError("Rollout workflow is not initialized. Call init() first.")

        self._last_rollout_metadata = {}
        logger.info(f"Starting generation for rollout {rollout_id}")

        # 2. Get algorithm requirements to determine pipeline
        requirements = self._sampling_requirements
        if requirements is None:
            requirements = resolve_sampling_requirements(self.args, algorithm=self.algorithm)
            self._sampling_requirements = requirements
        actor_group = self.sampling_group
        if actor_group is None:
            raise RuntimeError("No sampling group attached. Call attach_sampling_group() first.")

        train_data, metadata = self.rollout_workflow.build_training_batch(
            rollout_id=rollout_id,
            sde_indices=sde_indices,
            requirements=requirements,
            execute_sampling_request=lambda request, **kwargs: self._execute_sampling_request(
                request,
                actor_group=actor_group,
                **kwargs,
            ),
            collect_media_preview=collect_media_preview,
            media_max_items=media_max_items,
            debug_trace=debug_trace,
        )

        if not hasattr(train_data, "slice"):
            raise TypeError(
                "Rollout pipeline must return a TrainingBatch-like object exposing slice(). "
                f"Got type={type(train_data)}"
            )

        if metadata is not None:
            self._last_rollout_metadata = dict(metadata)
        self._finalize_rollout_state(train_data=train_data)
        if debug_trace is not None:
            debug_trace["training_batch"] = train_data

        return train_data

    def produce_training_payload(  # [PUBLIC-API → train.py, train_async.py] 核心产出方法
        self,
        rollout_id: int,
        *,
        sde_indices: Optional[Set[int]] = None,
        collect_media_preview: bool = False,
        media_max_items: int = 8,
    ) -> RolloutPayload:
        """Produce one buffer-ready rollout payload without pushing it.

        This is the buffer-centric public seam used by train.py/train_async:
        rollout producer builds a typed training batch plus metadata, and the
        caller decides when/how to push into the rollout buffer.
        """
        train_data = self.build_training_batch(
            rollout_id,
            sde_indices=sde_indices,
            collect_media_preview=collect_media_preview,
            media_max_items=media_max_items,
        )
        metadata = dict(self._last_rollout_metadata) if self._last_rollout_metadata else {}
        return RolloutPayload(
            rollout_id=int(rollout_id),
            training_batch=train_data,
            metadata=metadata,
        )

    def _finalize_rollout_state(self, *, train_data: TrainingBatch) -> None:  # [INTERNAL → build_training_batch()]
        """Update rollout-side counters after one payload is produced."""
        if hasattr(train_data, "rewards") and train_data.rewards is not None:
            sample_count = int(train_data.rewards.shape[0])
        elif hasattr(train_data, "batch_size"):
            sample_count = int(train_data.batch_size)
        else:
            raise ValueError(
                "TrainingBatch must expose rewards or batch_size so rollout statistics "
                "can track the generated sample count."
            )
        self._total_samples_generated += int(sample_count)

    def build_training_debug_payload(
        self,
        rollout_id: int,
        *,
        sde_indices: Optional[Set[int]] = None,
    ) -> Dict[str, Any]:  # [PUBLIC-API → train.py] debug 模式
        """Public rollout-side debug entrypoint built on the main rollout path."""
        if not self._is_initialized:
            raise RuntimeError("RolloutManager not initialized. Call init() first.")

        debug_trace: Dict[str, Any] = {}
        self.build_training_batch(
            rollout_id,
            sde_indices=sde_indices,
            collect_media_preview=False,
            media_max_items=0,
            debug_trace=debug_trace,
        )
        return debug_trace

    def _generated_samples_per_rollout(self) -> int:  # [HELPER] 多处内部使用
        samples_per_prompt = int(
            getattr(self.algorithm, "samples_per_prompt", getattr(self.args.algorithm, "samples_per_prompt", 1))
        )
        return max(1, self._prompt_batch_size * max(1, samples_per_prompt))

    # --- Pipeline executor methods ---

    def _prepare_batch(self, *, data_source: Any) -> Dict[str, Any]:  # [INTERNAL → build_training_batch(), eval_runner]
        """Fetch one prompt batch from data source."""
        if data_source is None:
            raise RuntimeError("RolloutManager requires an initialized data source.")
        batch_size = self._prompt_batch_size
        samples = data_source.get_samples(batch_size)
        if isinstance(samples, dict):
            return samples
        raise TypeError(
            "DataSource.get_samples() must return Dict[str, Any] with at least 'prompts'. "
            f"Got {type(samples).__name__}."
        )

    def _execute_sampling_request(  # [INTERNAL → rollout_workflow] 构造完整 RolloutRequest 并调 distributed_sample
        self,
        request: RolloutRequest,
        *,
        actor_group: Any,
        sde_indices: Optional[Set[int]],
        requirements: Optional[Any] = None,
        sampling_overrides: Optional[Dict[str, Any]] = None,
    ) -> SampledRequestResult:
        """Run distributed sampling for one typed rollout request."""
        prompts = request.prompts or []
        if not isinstance(prompts, list) or len(prompts) == 0:
            raise ValueError(
                "Rollout sampling requires non-empty text prompts in request.prompts. "
                "Prompt-embedding-only batches are no longer supported."
            )

        overrides = dict(sampling_overrides or {})
        debug_output_dir = getattr(self.args.debug, "debug_output_dir", None)
        if debug_output_dir:
            # Keep consistency-debug wiring explicit at request construction time so
            # sampler dumps follow the same args->request path as normal rollout knobs.
            overrides.setdefault("debug_output_dir", str(debug_output_dir))
        samples_per_prompt = int(request.samples_per_prompt)
        init_same_noise = bool(request.init_same_noise)
        resolved_num_inference_steps = overrides.pop("num_inference_steps", request.num_inference_steps)
        resolved_guidance_scale = overrides.pop("guidance_scale", request.guidance_scale)
        resolved_height = overrides.pop("height", request.height)
        resolved_width = overrides.pop("width", request.width)
        resolved_num_frames = overrides.pop("num_frames", request.num_frames)
        if resolved_num_inference_steps is None:
            raise ValueError("RolloutRequest.num_inference_steps must be resolved before sampling.")
        if resolved_guidance_scale is None:
            raise ValueError("RolloutRequest.guidance_scale must be resolved before sampling.")
        if resolved_height is None or resolved_width is None or resolved_num_frames is None:
            raise ValueError(
                "RolloutRequest geometry must be resolved before sampling "
                f"(height={resolved_height}, width={resolved_width}, num_frames={resolved_num_frames})."
            )
        num_inference_steps = int(resolved_num_inference_steps)
        guidance_scale = float(resolved_guidance_scale)
        height = int(resolved_height)
        width = int(resolved_width)
        num_frames = int(resolved_num_frames)
        requires_trajectory = True
        requires_log_prob = True
        if requirements is not None:
            requires_trajectory = bool(getattr(requirements, "requires_trajectory", True))
            requires_log_prob = bool(getattr(requirements, "requires_log_prob", True))

        typed_request = replace(
            request,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_frames=num_frames,
            sde_indices=sde_indices,
            decode_for_reward=True,
            keep_reward_media_for_manager=bool(overrides.pop("_keep_reward_media_for_manager", False)),
            init_same_noise=init_same_noise,
            samples_per_prompt=samples_per_prompt,
            return_trajectories=requires_trajectory,
            return_log_probs=requires_log_prob,
            kwargs=overrides,
        )
        sampler_outputs = distributed_sample(
            actor_group=actor_group,
            request=typed_request,
        )
        return SampledRequestResult(
            sampler_outputs=sampler_outputs,
        )

    def eval(self, rollout_id: int) -> Dict[str, Any]:  # [PUBLIC-API → train.py, train_async.py]
        """
        Run evaluation generation.

        Args:
            rollout_id: Current rollout iteration

        Returns:
            Dictionary of evaluation metrics and samples
        """
        logger.info(f"Running evaluation for rollout {rollout_id}")

        actor_group = self.sampling_group
        if actor_group is None:
            raise RuntimeError("No sampling group attached. Call attach_sampling_group() first.")
        if self.eval_runner is None:
            raise RuntimeError("EvalRunner not initialized. Call init() first.")
        return self.eval_runner.evaluate(
            rollout_id=rollout_id,
            actor_group=actor_group,
        )

    def get_dataset_step_info(self) -> Dict[str, Any]:  # [PUBLIC-API → create_rollout_manager()]
        """Compute rollout-step progress information for the current dataset."""
        prompts_per_rollout = self._prompt_batch_size
        drop_last = bool(getattr(self.data_source, "drop_last", False))
        info: Dict[str, Any] = {
            "num_samples": 0,
            "prompts_per_rollout": prompts_per_rollout,
            "estimated_steps_per_dataset_pass": 0,
            "steps_before_reset": 0,
            "remainder_samples": 0,
            "drop_last": drop_last,
            "exact_dataset_pass_per_cycle": False,
        }

        if self.data_source is None or not hasattr(self.data_source, "num_samples"):
            return info

        num_samples = int(self.data_source.num_samples)
        info["num_samples"] = num_samples
        if num_samples <= 0:
            return info

        estimated_steps = (num_samples + prompts_per_rollout - 1) // prompts_per_rollout
        remainder = num_samples % prompts_per_rollout
        if drop_last:
            steps_before_reset = num_samples // prompts_per_rollout
        else:
            steps_before_reset = estimated_steps

        info.update(
            {
                "estimated_steps_per_dataset_pass": int(estimated_steps),
                "steps_before_reset": int(steps_before_reset),
                "remainder_samples": int(remainder),
                "exact_dataset_pass_per_cycle": bool(
                    remainder == 0 and steps_before_reset == estimated_steps
                ),
            }
        )
        return info

    def get_stats(self) -> Dict[str, Any]:  # [PUBLIC-API → train.py] 统计信息
        """Get rollout statistics."""
        return {
            "total_samples_generated": self._total_samples_generated,
        }

    def dispose(self) -> None:  # [PUBLIC-API → train.py, train_async.py] 清理
        """Clean up resources."""
        if self.reward_service is not None:
            self.reward_service.dispose()
            self.reward_service = None
        self.eval_runner = None
        self.rollout_workflow = None
        self.data_source = None
        self.algorithm = None
        self.sampling_group = None
        self._is_initialized = False
        logger.info("RolloutManager disposed")


def create_rollout_manager(  # [PUBLIC-API → train.py] 工厂：创建 + init + 返回 handle
    args,
    reward_pg_result: Optional[Tuple] = None,
    *,
    algorithm_config: Dict[str, Any],
) -> Tuple[ray.ObjectRef, Dict[str, Any]]:
    """
    Factory function to create RolloutManager.

    Args:
        args: TrainingArguments instance
        reward_pg_result: Placement group result for reward actors

    Returns:
        Tuple of (RolloutManager actor handle, dataset step info)
    """
    if not isinstance(algorithm_config, dict):
        raise ValueError("create_rollout_manager requires algorithm_config to be built by the driver.")

    rollout_manager = RolloutManager.options(
        num_cpus=1,
        num_gpus=0,
    ).remote(args, reward_pg_result, algorithm_config=algorithm_config)

    # Initialize
    ray.get(rollout_manager.init.remote())

    dataset_step_info = ray.get(rollout_manager.get_dataset_step_info.remote())

    return rollout_manager, dataset_step_info
