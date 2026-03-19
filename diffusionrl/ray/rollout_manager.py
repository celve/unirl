"""diffusionrl Rollout Manager - thin rollout producer facade."""
from dataclasses import replace
from functools import partial
import logging
import time as _time
from typing import Any, Dict, List, Optional, Set, Tuple

import ray
import torch

from diffusionrl.config.build_domain_args import (
    RewardSchema,
    build_algorithm_config,
    build_sampling_config,
)
from diffusionrl.config.resolution import resolve_prompts_per_rollout
from diffusionrl.reward.factory import create_manager_reward_executor
from diffusionrl.reward.pipeline import score_from_rollout_outputs as _score_reward_stage
from diffusionrl.runtime.contracts import resolve_sampling_requirements
from diffusionrl.runtime.eval import EvalRunner
from diffusionrl.runtime.pipeline.rollout_pipeline import compute_advantages as _compute_advantages_stage
from diffusionrl.runtime.pipeline.rollout_pipeline import (
    distributed_sample,
)
from diffusionrl.runtime.rollout.request_builder import (
    RolloutRequestBuilder,
    SampledRequestResult,
)
from diffusionrl.types.sampling import RolloutRequest
from diffusionrl.types.training_batch import TrainingBatch
from diffusionrl.utils import load_function

logger = logging.getLogger(__name__)


@ray.remote
class RolloutManager:
    """
    Rollout Manager - Coordinates the data generation pipeline.

    This manager handles:
    - Loading and managing algorithm, sampler, reward, and model components
    - Orchestrating the sampling -> reward -> advantage computation pipeline
    - Converting data to training format
    - Evaluation generation

    Key design principle:
    - All major components are still dynamically loaded through config paths.
    - RolloutManager is the rollout-side producer facade, not a secondary plugin workflow host.
    """

    def __init__(  # [PUBLIC-API → create_rollout_manager()] 构造，仅存配置，不初始化组件
        self,
        args,
        reward_pg_result: Optional[Tuple] = None,
        algorithm_config: Optional[Dict[str, Any]] = None,
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
        self._algorithm_config = dict(algorithm_config) if algorithm_config is not None else build_algorithm_config(args)
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

        self.eval_runner = EvalRunner(
            args=self.args,
            sampling_config=self._sampling_config,
            data_source=self.data_source,
            reward_service=self.reward_service,
            algorithm=self.algorithm,
            default_prompt_batch_fn=lambda: self._prepare_batch(data_source=self.data_source),
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
        """Produce one typed TrainingBatch from the rollout-side producer path."""
        if not self._is_initialized:
            raise RuntimeError("RolloutManager not initialized. Call init() first.")

        self._last_rollout_metadata = {}
        logger.info(f"Starting generation for rollout {rollout_id}")

        # 1. Get batch from data source (prompt-only external input contract)
        batch = self._prepare_batch(data_source=self.data_source)
        prompts = batch.get("prompts", [])

        # 2. Get algorithm requirements to determine pipeline
        requirements = self._sampling_requirements
        if requirements is None:
            requirements = resolve_sampling_requirements(self.args, algorithm=self.algorithm)
            self._sampling_requirements = requirements
        actor_group = self.sampling_group
        if actor_group is None:
            raise RuntimeError("No sampling group attached. Call attach_sampling_group() first.")

        train_data = self._generate_training_data(
            batch=batch,
            rollout_id=rollout_id,
            sde_indices=sde_indices,
            requirements=requirements,
            actor_group=actor_group,
            collect_media_preview=collect_media_preview,
            media_max_items=media_max_items,
            debug_trace=debug_trace,
        )

        if not hasattr(train_data, "slice"):
            raise TypeError(
                "Rollout pipeline must return a TrainingBatch-like object exposing slice(). "
                f"Got type={type(train_data)}"
            )

        self._finalize_rollout_state(train_data=train_data, prompts=prompts)
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
    ) -> Dict[str, Any]:
        """Produce one buffer-ready rollout payload without pushing it.

        This is the buffer-centric public seam used by train.py/train_async:
        rollout producer builds a typed training batch plus metadata, and the
        caller decides when/how to push into the rollout buffer.
        """
        payload_t0 = _time.perf_counter()
        train_data = self.build_training_batch(
            rollout_id,
            sde_indices=sde_indices,
            collect_media_preview=collect_media_preview,
            media_max_items=media_max_items,
        )
        payload_t1 = _time.perf_counter()
        logger.debug(
            "[TIMING] produce_training_payload rollout=%s: build_batch=%.2fs",
            rollout_id,
            payload_t1 - payload_t0,
        )
        metadata = dict(self._last_rollout_metadata) if self._last_rollout_metadata else None
        return {
            "rollout_id": int(rollout_id),
            "training_batch": train_data,
            "metadata": metadata,
        }

    def _finalize_rollout_state(self, *, train_data: TrainingBatch, prompts: List[str]) -> None:  # [INTERNAL → build_training_batch()]
        """Update rollout-side counters after one payload is produced."""
        del prompts
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

    def _build_reward_prompts(  # [HELPER → _generate_training_data()]
        self,
        *,
        prompts: List[str],
        sample_count: int,
    ) -> List[str]:
        candidate = list(prompts)
        if not candidate:
            return []
        expanded: List[str] = []
        while len(expanded) < sample_count:
            expanded.extend(candidate)
        return expanded[:sample_count]

    def _build_wandb_media_preview(  # [HELPER → _generate_training_data()]
        self,
        *,
        sampler_outputs: List[Any],
        reward_prompts: List[str],
        rewards: torch.Tensor,
        max_items: int,
    ) -> Optional[Dict[str, Any]]:
        limit = max(1, int(max_items))
        rewards_flat: List[float] = []
        if torch.is_tensor(rewards) and rewards.numel() > 0:
            rewards_flat = [float(v) for v in rewards.detach().cpu().reshape(-1).tolist()]

        images: List[Any] = []
        prompts: List[str] = []
        reward_values: List[float] = []
        global_sample_idx = 0

        for output in sampler_outputs:
            batch_size = int(getattr(output, "batch_size", 0) or 0)
            decoded_images = list(getattr(output, "decoded_images", None) or [])
            for image_idx, image in enumerate(decoded_images):
                if len(images) >= limit:
                    break
                if not hasattr(image, "save"):
                    continue
                sample_idx = global_sample_idx + image_idx
                images.append(image)
                prompt = reward_prompts[sample_idx] if sample_idx < len(reward_prompts) else ""
                prompts.append(str(prompt))
                reward_val = rewards_flat[sample_idx] if sample_idx < len(rewards_flat) else 0.0
                reward_values.append(float(reward_val))
            if len(images) >= limit:
                break
            global_sample_idx += batch_size

        if not images:
            return None

        return {
            "images": images,
            "prompts": prompts,
            "rewards": reward_values,
        }

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

    def _generate_training_data(  # [INTERNAL → build_training_batch()] 核心流水线: sample→reward→advantage→assemble
        self,
        batch: Dict[str, Any],
        rollout_id: int,
        sde_indices: Optional[Set[int]],
        requirements: Any,
        actor_group: Any,
        *,
        collect_media_preview: bool = False,
        media_max_items: int = 8,
        debug_trace: Optional[Dict[str, Any]] = None,
    ) -> TrainingBatch:
        """Unified training data generation for all algorithm types.

        Algorithm-specific behavior (forward vs trajectory, replay/alignment policy)
        is delegated to algorithm strategy hooks.

        Args:
            batch: Dict containing prompts (and optionally embeddings/metadata)
            rollout_id: Current rollout iteration number
            sde_indices: Explicit rollout SDE timestep indices from the control plane
            requirements: Algorithm sampling requirements
            actor_group: Actor group for distributed sampling
        """
        prompts = batch.get("prompts", []) or []
        if sde_indices is not None:
            logger.debug("Received explicit rollout SDE indices: %s", sorted(int(i) for i in sde_indices)[:5])

        sampling_overrides: Dict[str, Any] = {
            "_keep_reward_media_for_manager": bool(collect_media_preview),
        }

        validation_config = self.algorithm.get_sampler_validation_config(args=self.args)
        if not isinstance(validation_config, dict):
            validation_config = {}
        request_batches = self._request_builder.build_request_batches(
            batch=batch,
            samples_per_prompt=int(self.algorithm.samples_per_prompt),
        )
        request_num_inference_steps = int(request_batches[0][1].num_inference_steps)

        sample_t0 = _time.perf_counter()
        sampled_rollout = self._request_builder.execute_request_batches(
            request_batches=request_batches,
            rollout_id=rollout_id,
            sample_request=partial(
                self._sample,
                actor_group=actor_group,
                sde_indices=sde_indices,
                requirements=requirements,
                sampling_overrides=sampling_overrides,
            ),
            validate_sampler_outputs=partial(
                self._validate_sampler_outputs,
                requirements=requirements,
                allow_replay=bool(validation_config.get("allow_replay", False)),
                assert_step_alignment=bool(validation_config.get("assert_step_alignment", True)),
                mode_label=str(validation_config.get("mode_label", "trajectory")),
            ),
        )
        sample_t1 = _time.perf_counter()
        sampler_outputs = sampled_rollout.sampler_outputs
        train_prompts = sampled_rollout.train_prompts
        train_prompt_ids = sampled_rollout.train_prompt_ids
        sample_ids = sampled_rollout.sample_ids
        group_ids = sampled_rollout.group_ids
        prompt_metadata = sampled_rollout.prompt_metadata

        # Reward + advantage
        reward_t0 = _time.perf_counter()
        rewards, reward_components = self._compute_rewards_only(
            reward_service=self.reward_service,
            sampler_outputs=sampler_outputs,
            prompts=train_prompts if train_prompts else prompts,
            prompt_ids=train_prompt_ids,
            sample_ids=sample_ids,
            group_ids=group_ids,
            prompt_metadata=prompt_metadata,
        )
        advantages = _compute_advantages_stage(
            algorithm=self.algorithm,
            rewards=rewards,
            group_ids=group_ids,
            reward_components=reward_components,
            reward_component_weights=self._reward_schema.component_weights(),
        )
        reward_t1 = _time.perf_counter()

        if collect_media_preview:
            reward_prompts = self._build_reward_prompts(
                prompts=train_prompts if train_prompts else prompts,
                sample_count=int(rewards.shape[0]),
            )
            media_preview = self._build_wandb_media_preview(
                sampler_outputs=sampler_outputs,
                reward_prompts=reward_prompts,
                rewards=rewards,
                max_items=media_max_items,
            )
            if media_preview is not None:
                self._last_rollout_metadata["wandb_media_preview"] = media_preview

        # Assemble
        assemble_t0 = _time.perf_counter()
        assembled_batch = self.algorithm.assemble_training_batch(
            num_inference_steps=request_num_inference_steps,
            sampler_outputs=sampler_outputs,
            rewards=rewards,
            advantages=advantages,
            prompts=train_prompts if train_prompts else prompts,
            sde_indices=sde_indices,
        )
        training_batch = self._attach_batch_identities(
            batch=assembled_batch,
            prompt_ids=train_prompt_ids,
            sample_ids=sample_ids,
            group_ids=group_ids,
        )
        assemble_t1 = _time.perf_counter()
        logger.debug(
            "[TIMING] _generate_training_data rollout=%s: sample=%.2fs reward_advantage=%.2fs assemble=%.2fs total=%.2fs",
            rollout_id,
            sample_t1 - sample_t0,
            reward_t1 - reward_t0,
            assemble_t1 - assemble_t0,
            assemble_t1 - sample_t0,
        )
        if debug_trace is not None:
            reward_prompts = self._build_reward_prompts(
                prompts=train_prompts if train_prompts else prompts,
                sample_count=int(rewards.shape[0]),
            )
            debug_trace.update(
                {
                    "rollout_id": int(rollout_id),
                    "debug_mode": str(getattr(self.args.debug, "debug_mode", "none")),
                    "prompts": list(prompts),
                    "train_prompts": list(train_prompts if train_prompts else prompts),
                    "prompt_ids": list(train_prompt_ids or []),
                    "sample_ids": list(sample_ids or []),
                    "group_ids": list(group_ids or []),
                    "reward_prompts": reward_prompts,
                    "sde_indices": sorted(int(v) for v in (sde_indices or [])),
                    "sampler_outputs": sampler_outputs,
                    "rewards": rewards,
                    "advantages": advantages,
                    "reward_components": reward_components,
                }
            )
        return training_batch

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

    def _sample(  # [INTERNAL → _generate_training_data()] 构造完整 RolloutRequest 并调 distributed_sample
        self,
        *,
        actor_group: Any,
        request: RolloutRequest,
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

    # TODO(refactor): Move sampler-output contract validation into
    # runtime/rollout/sampler_output_contract.py next to the fallback helpers
    # above. This is post-sampling contract handling, not panel orchestration.
    # RolloutManager should pass requirements/validation config in and let the
    # runtime helper raise contract errors or return validated outputs.
    def _validate_sampler_outputs(  # [INTERNAL → _generate_training_data()] 校验 RolloutOutput 合约
        self,
        *,
        sampler_outputs: List[Any],
        requirements: Any,
        allow_replay: bool,
        assert_step_alignment: bool,
        mode_label: str,
    ) -> None:
        """Validate sampler outputs against algorithm requirements."""
        replay_notice_emitted = False
        for idx, out in enumerate(sampler_outputs):
            try:
                meta = getattr(out, "metadata", {}) or {}
                generator_type = meta.get("generator_type") if isinstance(meta, dict) else None
                allow_missing_log_probs = bool(allow_replay)
                if allow_missing_log_probs and not replay_notice_emitted:
                    logger.warning(
                        "Replay path enabled: allowing missing rollout log_probs; "
                        "training actors will replay old log_probs before backward."
                    )
                    replay_notice_emitted = True

                out.validate_contract(
                    requires_log_probs=bool(requirements.requires_log_prob) and not allow_missing_log_probs,
                    requires_trajectory=bool(requirements.requires_trajectory),
                    requires_embeddings=bool(getattr(requirements, "requires_embeddings", True)),
                )

                if assert_step_alignment:
                    resolved_steps = out.resolved_step_indices
                    if int(resolved_steps.shape[0]) != int(out.timesteps.shape[0]):
                        raise ValueError(
                            f"step/timestep length mismatch: step_indices={resolved_steps.shape[0]}, "
                            f"timesteps={out.timesteps.shape[0]}"
                        )
            except Exception as e:
                meta = getattr(out, "metadata", {}) or {}
                generator_type = meta.get("generator_type") if isinstance(meta, dict) else None
                capabilities = meta.get("engine_capabilities") if isinstance(meta, dict) else None
                traj_shape = tuple(out.trajectories.shape) if getattr(out, "trajectories", None) is not None else None
                latents_shape = tuple(out.latents.shape) if getattr(out, "latents", None) is not None else None
                steps_shape = tuple(out.resolved_step_indices.shape) if hasattr(out, "resolved_step_indices") else None
                hint = ""
                if generator_type == "sglang":
                    hint = (
                        f" {generator_type} currently may omit rollout log_probs; "
                        "enable replay_log_probs and ensure prompt text inputs are present."
                    )
                raise RuntimeError(
                    f"Sampler output contract validation failed in {mode_label} path at index={idx}: {e}.{hint} "
                    f"capabilities={capabilities}, latents_shape={latents_shape}, "
                    f"trajectories_shape={traj_shape}, step_indices_shape={steps_shape}"
                ) from e

    def _compute_rewards_only(  # [INTERNAL → _generate_training_data()] 封装 reward.pipeline.score_from_rollout_outputs
        self,
        *,
        reward_service: Any,
        sampler_outputs: List[Any],
        prompts: List[str],
        prompt_ids: Optional[List[str]] = None,
        sample_ids: Optional[List[str]] = None,
        group_ids: Optional[List[str]] = None,
        prompt_metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
        samples_per_prompt_override: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """Compute rewards for one sampled rollout batch."""
        samples_per_prompt = int(
            samples_per_prompt_override
            if samples_per_prompt_override is not None
            else getattr(self.algorithm, "samples_per_prompt", getattr(self.args.algorithm, "samples_per_prompt", 1))
        )

        return _score_reward_stage(
            reward_service=reward_service,
            samples_per_prompt=samples_per_prompt,
            sampler_outputs=sampler_outputs,
            prompts=prompts,
            prompt_ids=prompt_ids,
            sample_ids=sample_ids,
            group_ids=group_ids,
            prompt_metadata=prompt_metadata,
        )

    def _attach_batch_identities(  # [INTERNAL → _generate_training_data()] 给 batch 挂 prompt_ids/sample_ids/group_ids
        self,
        *,
        batch: TrainingBatch,
        prompt_ids: Optional[List[str]] = None,
        sample_ids: Optional[List[str]] = None,
        group_ids: Optional[List[str]] = None,
    ) -> TrainingBatch:
        """Attach explicit per-sample identity fields to a training batch."""
        batch_size = int(getattr(batch, "batch_size", 0))
        if batch_size <= 0:
            return batch

        resolved_prompt_ids = prompt_ids
        if resolved_prompt_ids is None:
            resolved_prompt_ids = getattr(batch, "prompt_ids", None)
        if resolved_prompt_ids is None or len(resolved_prompt_ids) != batch_size:
            raise ValueError(
                "Training batch identity attachment requires explicit sample-aligned prompt_ids. "
                f"Got batch_size={batch_size}, prompt_ids_len="
                f"{len(resolved_prompt_ids) if resolved_prompt_ids is not None else None}."
            )
        batch.prompt_ids = list(resolved_prompt_ids)

        resolved_sample_ids = sample_ids
        if resolved_sample_ids is None:
            resolved_sample_ids = getattr(batch, "sample_ids", None)
        if resolved_sample_ids is None or len(resolved_sample_ids) != batch_size:
            raise ValueError(
                "Training batch identity attachment requires explicit sample_ids aligned to the reward batch. "
                f"Got batch_size={batch_size}, sample_ids_len="
                f"{len(resolved_sample_ids) if resolved_sample_ids is not None else None}."
            )
        batch.sample_ids = list(resolved_sample_ids)

        resolved_group_ids = group_ids
        if resolved_group_ids is None:
            resolved_group_ids = getattr(batch, "group_ids", None)
        if resolved_group_ids is None or len(resolved_group_ids) != batch_size:
            raise ValueError(
                "Training batch identity attachment requires explicit group_ids aligned to the reward batch. "
                f"Got batch_size={batch_size}, group_ids_len="
                f"{len(resolved_group_ids) if resolved_group_ids is not None else None}."
            )
        batch.group_ids = list(resolved_group_ids)

        return batch

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
        self.sampling_group = None
        logger.info("RolloutManager disposed")


def create_rollout_manager(  # [PUBLIC-API → train.py] 工厂：创建 + init + 返回 handle
    args,
    reward_pg_result: Optional[Tuple] = None,
    algorithm_config: Optional[Dict[str, Any]] = None,
) -> Tuple[ray.ObjectRef, Dict[str, Any]]:
    """
    Factory function to create RolloutManager.

    Args:
        args: TrainingArguments instance
        reward_pg_result: Placement group result for reward actors

    Returns:
        Tuple of (RolloutManager actor handle, dataset step info)
    """
    rollout_manager = RolloutManager.options(
        num_cpus=1,
        num_gpus=0,
    ).remote(args, reward_pg_result, algorithm_config)

    # Initialize
    ray.get(rollout_manager.init.remote())

    dataset_step_info = ray.get(rollout_manager.get_dataset_step_info.remote())

    return rollout_manager, dataset_step_info
