"""diffusionrl train actor for the new ``RolloutResp`` / Policy-stack path.

Sibling of :class:`diffusionrl.ray.train_actor.TrainActor`. Drives the new
training stack end-to-end: a :class:`Policy` composed via
:func:`compose_policy` over a :class:`Stage` from a ``models_new`` pipeline,
trained by :class:`StageTrainStack` against a ``RolloutResp``.

Key differences vs the legacy actor:

- No legacy ``TrainBackend`` / ``FSDPBackend``. The Policy stack
  (``LoRAPolicy`` → ``FSDPPolicy`` → ``EMAPolicy``) owns FSDP wrap, LoRA
  injection and EMA shadow. The actor reads ``cfg.training.policies``
  (an ordered list of policy configs) and ``cfg.training.policy_source``
  (the slot name on the pipeline whose Stage anchors the stack).
- Slot-keyed algorithms. ``cfg.algorithms.<slot>`` (plural) replaces the
  single legacy ``cfg.algorithm``; each entry materializes a
  :class:`StageAlgorithm`. The actor injects ``stage=`` (and optionally
  ``params=``) at instantiate time so the algorithm preset can stay a
  static ``_target_`` declaration.
- ``RolloutResp`` (not legacy ``TrainingBatch``) as the train-time data
  type; sliced via :meth:`Batched.slice` and dispatched per slot.

Dual-mode sampling
------------------
The actor supports BOTH separate sampling and direct sampling, gated on
:func:`diffusionrl.config.validation.is_direct_sampling`:

- **Separate sampling** (default; ``cfg.rollout.engine: vllm_omni``
  or sglang): rollout runs on a sibling :class:`NewRolloutActor`;
  ``RolloutResp`` arrives via Ray. The mixin's host-contract attributes
  on this actor stay ``None``; calling ``run_rollout_pipeline`` / etc.
  would raise.
- **Direct sampling** (``cfg.rollout.engine: trainside``): the
  FSDP-wrapped Policy IS the sampler. The actor populates
  ``self.engine = TrainsideRolloutEngine(pipeline=..., policy=...)``
  plus the rest of the :class:`NewRolloutPipelineMixin` host-contract
  (``self._rollout_plan`` / ``self.algorithm`` / ``self._reward_pipeline``
  / ``self.generate(req)``) so :class:`NewRolloutActorGroup.from_train_group`
  can adopt this actor's handle as a rollout actor with no proxy / no
  weight sync.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union

import hydra.utils
import ray
import torch
from omegaconf import DictConfig

from diffusionrl.ray.actor_config import ConfigActor
from diffusionrl.ray.distributed import DistributedMixin
from diffusionrl.ray.mixins import TrainingWeightSyncMixin
from diffusionrl.ray.mixins.new_rollout_pipeline import NewRolloutPipelineMixin
from diffusionrl.reward.pipeline import RewardPipeline
from diffusionrl.rollout.engine import chunked_engine_generate_req
from diffusionrl.training_new import StageMiniBatchResult, StageTrainStack
from diffusionrl.training_new.ema_policy import EMAPolicy
from diffusionrl.training_new.fsdp_policy import FSDPPolicy
from diffusionrl.training_new.policy import Policy, compose_policy, walk_source_chain
from diffusionrl.transfer.buffer import Buffer, BufferHandle
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp
from diffusionrl.utils import clear_memory as _clear_gpu_memory

logger = logging.getLogger(__name__)


def _iter_optimizer_param_states(optimizer: Any):
    """Yield per-parameter optimizer state dicts (AdamW-style)."""
    yield from optimizer.state.values()


def _detect_lora_on_model(model: Any) -> bool:
    """Return True iff ``model`` has at least one PEFT adapter registered.

    Walks the same wrap layers that
    ``training_new/fsdp_policy.py:_extract_peft_lora_state`` checks: the
    PEFT adapter dict can live directly on ``model`` (post-injection,
    pre-FSDP wrap) or under ``model.module`` (FSDP wrap) or
    ``model.base_model`` (PEFT's own wrapper). Returning True implies the
    trainer side must sync LoRA tensors (not just base weights) to the
    rollout engine on every ``sync_weights_to_rollout`` call.
    """
    candidates = []
    seen = set()
    cur = model
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        candidates.append(cur)
        cur = getattr(cur, "module", None) or getattr(cur, "base_model", None)
    for c in candidates:
        pc = getattr(c, "peft_config", None)
        if isinstance(pc, dict) and len(pc) > 0:
            return True
    return False


def _materialize_policy_config(node: Any) -> Any:
    """Materialize a ``cfg.training.policies`` entry into a typed config dataclass.

    ``materialize(node)`` returns a typed dataclass when ``node`` carries a
    structured schema (e.g. via Hydra defaults composition), but yaml-inline
    list elements arrive as plain ``dict``s. ``compose_policy`` and the
    Policy runtime classes (``FSDPPolicy``, ``LoRAPolicy``, ``EMAPolicy``)
    expect a dataclass instance whose dotted attributes they read directly
    (``self.config.cpu_offload`` etc.).

    For the dict case, look up the runtime class via ``_target_`` and
    instantiate the paired ``<RuntimeCls>Config`` dataclass living in the
    same module. Convention is enforced by ``@register_config`` callers
    today; raise a clear error if no paired Config class exists so future
    Policies that diverge from the convention surface here instead of
    inside ``compose_policy``.
    """
    import sys

    import hydra.utils

    from diffusionrl.config.instantiate import materialize as _materialize

    obj = _materialize(node)
    if not isinstance(obj, dict):
        return obj
    target_path = obj.get("_target_")
    if not target_path:
        raise ValueError(
            "_materialize_policy_config: policy node has no _target_; "
            "use a @register_config preset (group='training_new/policy')."
        )
    runtime_cls = hydra.utils.get_method(target_path)
    cfg_cls_name = runtime_cls.__name__ + "Config"
    cfg_module = sys.modules[runtime_cls.__module__]
    cfg_cls = getattr(cfg_module, cfg_cls_name, None)
    if cfg_cls is None:
        raise ValueError(
            f"_materialize_policy_config: no paired Config class "
            f"{cfg_cls_name!r} in module {runtime_cls.__module__!r} for "
            f"runtime target {target_path!r}. Pass an instantiated Config "
            "dataclass directly, or attach the structured schema via "
            "Hydra defaults composition."
        )
    return cfg_cls(**{k: v for k, v in obj.items() if k != "_target_"})


@ray.remote(num_gpus=1)
class NewTrainActor(
    ConfigActor,
    TrainingWeightSyncMixin,
    DistributedMixin,
    NewRolloutPipelineMixin,
    Buffer,
):
    """Train actor for the new pipeline + Policy-stack path.

    MRO: ``ConfigActor`` installs the cfg, ``TrainingWeightSyncMixin``
    contributes the rollout-side weight push, ``DistributedMixin`` owns
    rank/world/master state and torch.distributed init,
    :class:`NewRolloutPipelineMixin` contributes the
    ``generate_buffered`` / ``attach_reward`` / ``run_rollout_pipeline``
    surface used by :class:`NewRolloutActorGroup.from_train_group` in
    direct-sampling mode, and ``Buffer`` provides the local Batched store.

    The mixin's host-contract attributes (``self.engine`` /
    ``self._rollout_plan`` / ``self.algorithm`` / ``self._reward_pipeline``)
    are populated in ``__init__`` only when
    :func:`is_direct_sampling` returns True. In separate-sampling mode
    they stay ``None`` and calling ``run_rollout_pipeline`` raises.
    """

    def __init__(
        self,
        *,
        cfg: DictConfig,
        world_size: int,
        rank: int,
        master_addr: Optional[str],
        master_port: Optional[int],
        seed: int = 42,
    ) -> None:
        from diffusionrl.config.instantiate import build, materialize
        from diffusionrl.utils import set_seed

        set_seed(int(seed))

        super().__init__(
            cfg=cfg,
            world_size=world_size,
            rank=rank,
            master_addr=master_addr,
            master_port=master_port,
        )
        self._init_weight_sync_state()

        # ------------------------------------------------------------------
        # Topology + distributed init
        # ------------------------------------------------------------------
        self._device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
        torch.cuda.set_device(self._device)
        self._init_distributed()

        # ``_use_lora`` is set later (after ``compose_policy`` injects the
        # LoRA adapter) via runtime detection on ``self.model.peft_config``.
        # See the ``_detect_lora_on_model`` call below — relying on the old
        # ``cfg.model.use_lora`` knob is unsafe on the Policy-stack path
        # because LoRA lives under ``cfg.training.policies``, not under
        # ``cfg.model``.
        self._use_lora = False

        # ------------------------------------------------------------------
        # Build the pipeline. ``cfg.model._target_`` resolves to a pipeline
        # factory (e.g. ``HunyuanImage3Pipeline.from_meta_config``). Bundle
        # weight materialization is deferred to a separate collective call.
        # ------------------------------------------------------------------
        self.pipeline = build(cfg.model, strategy=build(cfg.sampling.sde_strategy))
        self.bundle = self.pipeline.bundle

        # ------------------------------------------------------------------
        # Compose the Policy stack. Configs are listed inside-out: the first
        # config becomes the innermost policy (closest to the Stage), the
        # last becomes the outermost handle held on ``self.policy``.
        # Both peft injection and ``fully_shard`` mutate the underlying
        # nn.Module in place — so ``stage.replay`` keeps working through the
        # chain while ``policy.parameters()`` exposes the FSDP-aware
        # optimizer surface.
        # ------------------------------------------------------------------
        source_slot = str(cfg.training.policy_source)
        source_stage = getattr(self.pipeline, source_slot)
        policy_configs = [_materialize_policy_config(node) for node in cfg.training.policies]
        self.policy: Policy = compose_policy(source_stage, policy_configs)

        # Mixin contract: ``self.model`` must point at the trainable module
        # for LoRA-tensor extraction. ``policy.model`` is the in-place
        # mutated module shared by every layer in the stack.
        self.model = self.policy.model

        # ------------------------------------------------------------------
        # Materialize bundle weights. Single-call DCP broadcast from rank 0;
        # every rank participates. ``with_aux`` controls which aux models
        # (e.g. ``vae``) get materialized — train actors typically leave aux
        # on meta and let the rollout side hold the real VAE.
        # ------------------------------------------------------------------
        # HI3 bundle uses meta-init + DCP gather inside ``materialize``;
        # SD3 bundle already eager-loads to GPU in ``from_config`` and
        # exposes no ``materialize`` method. Skip the call in that case.
        materialize_cfg = cfg.training.get("materialize")
        with_aux: tuple = ()
        if materialize_cfg is not None:
            with_aux = tuple(materialize_cfg.get("with_aux", []) or ())
        # Eager-loading bundles (e.g. SD3Bundle) build weights directly in
        # ``from_config`` and don't expose ``materialize`` — for those,
        # there's nothing to do here. Meta-init bundles (e.g.
        # HunyuanImage3Bundle) build on meta-device and materialize via
        # DCP, so we call through. ``with_aux`` is honored only by the
        # meta-init path; an eager bundle has already loaded everything
        # it's going to load.
        bundle_materialize = getattr(self.bundle, "materialize", None)
        if callable(bundle_materialize):
            bundle_materialize(device=self._device, with_aux=with_aux)
        elif with_aux:
            logger.info(
                "Rank %s: bundle %s loads eagerly; ignoring cfg.training.materialize.with_aux=%s",
                self.rank,
                type(self.bundle).__name__,
                with_aux,
            )

        # Walk the chain inward letting each policy run its own
        # post-materialize state init (LoRA reset, EMA snapshot, …).
        self.policy.post_materialize_init()

        # Runtime LoRA detection. The new Policy-stack path injects LoRA
        # via ``LoRAPolicy`` inside ``cfg.training.policies`` — by this
        # point ``compose_policy`` + ``post_materialize_init`` have
        # mutated ``self.model`` in place with the PEFT adapter and the
        # ``peft_config`` dict is the authoritative signal. Reading
        # ``cfg.model.use_lora`` was wrong here (LoRA isn't under
        # ``cfg.model`` anymore) and silently disabled the LoRA-sync
        # path on the vllm-omni rollout engine. Mirrors the detection
        # used in ``training_new/fsdp_policy.py:_extract_peft_lora_state``.
        self._use_lora = _detect_lora_on_model(self.model)
        if self._use_lora:
            logger.info(
                "Rank %s: LoRA detected on self.model (peft_config keys=%s)",
                self.rank,
                list(getattr(self.model, "peft_config", None) or {}),
            )

        # ------------------------------------------------------------------
        # Build per-slot algorithms. Each preset carries ``_target_``;
        # ``stage`` and ``params`` are injected from the pipeline so the
        # cfg can stay a static declaration.
        #
        # The slot key serves two roles: (a) ``resp.rollout_traces.get(slot)``
        # in :meth:`StageTrainStack.train_microbatch`, and (b) attribute
        # lookup on the pipeline for the Stage and per-request params.
        # When (a) and (b) name the same thing (HI3 convention) the
        # defaults suffice. When they diverge (SD3 emits
        # ``rollout_traces["image"]`` but exposes ``pipe.diffusion``), the
        # ``stage_attr`` knob overrides the pipeline-attribute lookup;
        # ``params_attr`` plays the symmetric role for per-slot params.
        # If neither the pipeline nor the cfg supplies ``params``, leave
        # the kwarg unset and let the algorithm receive its default (or
        # use an inline ``params: {_target_: ...}`` block on alg_node,
        # which Hydra will instantiate during the per-slot
        # ``hydra.utils.instantiate`` call).
        # ------------------------------------------------------------------
        from omegaconf import OmegaConf as _OmegaConf

        self.algorithms: Dict[str, Any] = {}
        _CONTROL_KEYS = ("stage_attr", "params_attr", "conditions_cls")
        for slot, alg_node in cfg.algorithms.items():
            # Slot names match ``RolloutResp.rollout_traces`` keys (e.g. "image"),
            # NOT pipeline attribute names. ``stage_attr`` lets the cfg
            # point at the pipeline attribute explicitly; default to slot
            # for the case where they coincide.
            stage_attr = alg_node.get("stage_attr", slot)
            stage = getattr(self.pipeline, stage_attr)
            params_attr = alg_node.get("params_attr", f"{slot}_params")
            params_obj = getattr(self.pipeline, params_attr, None)
            # conditions_cls is a class reference, not an instance — Hydra
            # has no first-class syntax for this in a kwarg position, so we
            # accept a dotted path string and resolve it via
            # ``hydra.utils.get_class``. Matches the pattern used in
            # scripts/smoke_new_train_actor_sd3.py
            # (conditions_cls=SD3Conditions).
            conditions_cls_path = alg_node.get("conditions_cls")
            conditions_cls = hydra.utils.get_class(str(conditions_cls_path)) if conditions_cls_path else None
            # Strip control keys so Hydra doesn't forward them to
            # ``DiffusionGRPO.__init__`` (which would raise TypeError on
            # unknown kwargs).
            clean_node = _OmegaConf.create({k: v for k, v in alg_node.items() if k not in _CONTROL_KEYS})
            inject_kwargs: Dict[str, Any] = {"stage": stage}
            # Only inject ``params`` when the pipeline supplies it; otherwise
            # let alg_node.params (if present, e.g. an inline
            # ``params: {_target_: SD3DiffusionParams, ...}``) drive
            # construction via Hydra recursion.
            if params_obj is not None:
                inject_kwargs["params"] = params_obj
            if conditions_cls is not None:
                inject_kwargs["conditions_cls"] = conditions_cls
            # Forward-process algorithms (DiffusionNFT) need a handle to
            # the NFTLoRAPolicy sitting in the policy chain to switch
            # between "default" and "old" adapters in their loss math.
            # Walk-from-stage doesn't reach policies (the chain points
            # stage ← policies, not the other way), so we inject the
            # policy reference at construction time.
            target = str(alg_node.get("_target_") or "")
            if target.endswith(".DiffusionNFT"):
                from diffusionrl.training_new.nft_lora_policy import NFTLoRAPolicy

                nft_lora = next(
                    (p for p in walk_source_chain(self.policy) if isinstance(p, NFTLoRAPolicy)),
                    None,
                )
                if nft_lora is None:
                    raise RuntimeError(
                        f"NewTrainActor: algorithms.{slot}._target_={target!r} "
                        f"requires an NFTLoRAPolicy in the policy chain. "
                        f"Configure training.policies with NFTLoRAPolicy as "
                        f"the innermost adapter-management layer (replacing "
                        f"the plain LoRAPolicy used by GRPO recipes)."
                    )
                inject_kwargs["nft_lora_policy"] = nft_lora
            self.algorithms[slot] = hydra.utils.instantiate(
                clean_node,
                _convert_="object",
                **inject_kwargs,
            )

        # ------------------------------------------------------------------
        # Optimizer + scheduler over the Policy's exposed parameters.
        # ``policy.parameters()`` already filters to requires_grad=True
        # under a LoRA stack (LoRAPolicy override); the AdamW factory
        # filters again defensively.
        # ------------------------------------------------------------------
        from diffusionrl.training_new.factories import build_lr_scheduler, build_optimizer

        optimizer_config = materialize(cfg.training.optimizer)
        scheduler_config = materialize(cfg.training.lr_scheduler)
        self.optimizer = build_optimizer(
            optimizer_config,
            params=list(self.policy.parameters()),
            backend=None,
            actor=self,
        )
        self.lr_scheduler = build_lr_scheduler(
            scheduler_config,
            optimizer=self.optimizer,
            backend=None,
            actor=self,
        )

        self.train_stack = StageTrainStack(
            policy=self.policy,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            algorithms=self.algorithms,
            cfg=cfg,
        )

        # Eval-EMA swap state (set by apply_eval_ema, cleared by restore).
        self._eval_ema_active: bool = False

        # ------------------------------------------------------------------
        # NewRolloutPipelineMixin host-contract.
        #
        # In direct-sampling mode (cfg.rollout.engine: trainside) the
        # FSDP-wrapped Policy itself serves rollouts via the in-process
        # TrainsideRolloutEngine; NewRolloutActorGroup.from_train_group
        # adopts this actor handle as a rollout actor with no proxy and
        # no weight sync. In separate-sampling mode the host-contract
        # attrs stay None and ``self.generate(req)`` raises.
        # ------------------------------------------------------------------
        from diffusionrl.config.validation import is_direct_sampling

        self.engine = None
        self._rollout_plan = None
        self._reward_config = None
        self._reward_pipeline: Optional[RewardPipeline] = None
        self.algorithm = None

        if is_direct_sampling(cfg):
            self.engine = build(
                cfg.rollout.engine,
                pipeline=self.pipeline,
                policy=self.policy,
            )
            self._rollout_plan = materialize(cfg.rollout.plan)
            self._reward_config = cfg.reward
            self.algorithm = build(cfg.algorithm)
            logger.info(
                "Rank %s: direct-sampling engine installed (%s); rollout_plan.forward_batch_size=%s",
                self.rank,
                type(self.engine).__name__,
                getattr(self._rollout_plan, "forward_batch_size", None),
            )

        logger.info(
            "Rank %s: NewTrainActor initialized (slots=%s, policy_chain=%s, direct_sampling=%s)",
            self.rank,
            list(self.algorithms.keys()),
            " → ".join(type(p).__name__ for p in walk_source_chain(self.policy)),
            self.engine is not None,
        )

    # ------------------------------------------------------------------
    # Distributed env (one rank per actor — same shape as legacy)
    # ------------------------------------------------------------------

    def _setup_distributed_env(self) -> None:
        if self.master_addr is None or self.master_port is None:
            raise ValueError("master_addr and master_port must be set")

        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_devices:
            local_rank = 0
        else:
            device_count = torch.cuda.device_count()
            local_rank = self.rank % device_count if device_count > 0 else 0

        self._write_distributed_env(
            master_addr=self.master_addr,
            master_port=self.master_port,
            world_size=self.world_size,
            rank=self.rank,
            local_rank=local_rank,
        )
        logger.info(
            f"Distributed env setup: rank={self.rank}, world_size={self.world_size}, "
            f"master={self.master_addr}:{self.master_port}"
        )

    # ------------------------------------------------------------------
    # NewRolloutPipelineMixin host-contract methods (direct-sampling only)
    # ------------------------------------------------------------------

    def generate(self, req: RolloutReq) -> RolloutResp:
        """Generate one rollout via the in-process Trainside engine.

        Mirror of :meth:`NewRolloutActor.generate` for the direct-sampling
        path: chunks ``req`` at ``forward_batch_size`` and concatenates
        per-chunk responses. Raises if direct sampling was not enabled at
        construction (``cfg.rollout.engine`` did not target
        :class:`TrainsideRolloutEngine`).
        """
        if self.engine is None:
            raise RuntimeError(
                "NewTrainActor.generate: cfg.rollout.engine must target "
                "TrainsideRolloutEngine for the train actor to serve rollouts. "
                "In separate-sampling mode, route generation through "
                "NewRolloutActor instead."
            )
        if int(req.batch_size) == 0:
            raise ValueError("NewTrainActor.generate requires non-empty req (batch_size>0).")
        return chunked_engine_generate_req(
            self.engine,
            req,
            chunk_size=self._rollout_plan.forward_batch_size,
        )

    def _ensure_reward_pipeline(self) -> RewardPipeline:
        """Lazy build of the reward pipeline from ``cfg.reward``.

        Cfg-selectable backend: ``cfg.reward`` may target a local
        :class:`RewardPipeline` (default) or a remote
        :class:`RewardServiceExecutor`. No branching here — the existing
        reward registry handles backend selection.
        """
        if self._reward_pipeline is None:
            if self._reward_config is None:
                raise RuntimeError(
                    "NewTrainActor._ensure_reward_pipeline: cfg.reward was not "
                    "captured at construction (direct sampling not enabled)."
                )
            self._reward_pipeline = RewardPipeline.from_configs(self._reward_config)
        return self._reward_pipeline

    # ------------------------------------------------------------------
    # Train (RolloutResp in, StageMiniBatchResult out)
    # ------------------------------------------------------------------

    def train(
        self,
        rollout_step: int,
        resp_or_handle: Union[ray.ObjectRef, RolloutResp],
    ) -> StageMiniBatchResult:
        """Execute one training step on a materialized RolloutResp."""
        if isinstance(resp_or_handle, ray.ObjectRef):
            resp: RolloutResp = ray.get(resp_or_handle)
        else:
            resp = resp_or_handle
        return self._train_resp(rollout_step, resp)

    def train_from_buffer(
        self,
        rollout_step: int,
        handle: BufferHandle,
    ) -> StageMiniBatchResult:
        """Pop a RolloutResp from a remote buffer and train on it."""
        resp: RolloutResp = ray.get(handle.actor_handle.pop_buffer.remote(handle))
        return self._train_resp(rollout_step, resp)

    def _train_resp(self, rollout_step: int, resp: RolloutResp) -> StageMiniBatchResult:
        resp = resp.to_device(self._device)
        num_rollouts = int(self._cfg.run.num_rollouts)
        progress = max(0.0, min(1.0, float(rollout_step) / max(1, num_rollouts)))
        result = self.train_stack.train_minibatch(resp, training_progress=progress)
        # Per-rollout-boundary Policy hook. Fires once per ``train()`` RPC
        # call. Production NFT runs with ``num_updates_per_batch=1`` so
        # this coincides with the rollout boundary; recipes that bump
        # that knob multiply the hook frequency by the same factor, so
        # rollout-end-keyed Policies (NFTLoRAPolicy with
        # ema_update_timing="rollout_end") must keep num_updates_per_batch=1.
        self.train_stack.on_rollout_end()
        return result

    # ------------------------------------------------------------------
    # Eval-EMA swap (RPC-style; uses the explicit EMAPolicy lifecycle
    # methods rather than the contextmanager so the swap can span
    # ``actor.apply_eval_ema.remote()`` ⟶ ``actor.restore_from_eval.remote()``).
    # ------------------------------------------------------------------

    def apply_eval_ema(self) -> None:
        ema = next(
            (p for p in walk_source_chain(self.policy) if isinstance(p, EMAPolicy)),
            None,
        )
        if ema is None:
            return
        ema.apply_ema_to_model()
        self._eval_ema_active = True

    def restore_from_eval(self) -> None:
        if not self._eval_ema_active:
            return
        ema = next(
            (p for p in walk_source_chain(self.policy) if isinstance(p, EMAPolicy)),
            None,
        )
        if ema is not None:
            ema.restore_from_ema()
        self._eval_ema_active = False

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def load_checkpoint(self, path: str) -> None:
        checkpoint_path = os.path.join(path, "checkpoint.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.lr_scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "ema_state_dict" in checkpoint:
            ema = next(
                (p for p in walk_source_chain(self.policy) if isinstance(p, EMAPolicy)),
                None,
            )
            if ema is not None:
                ema.load_ema_state_dict(checkpoint["ema_state_dict"])

    def save_model(self, path: str) -> None:
        """Collective save. Every rank must call (DCP gather happens inside
        ``policy.state_dict()``); only rank 0 writes the file.
        """
        policy_sd = self.policy.state_dict()
        if self.rank != 0:
            return
        os.makedirs(path, exist_ok=True)
        checkpoint: Dict[str, Any] = {
            "policy_state_dict": policy_sd,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.lr_scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()
        ema = next(
            (p for p in walk_source_chain(self.policy) if isinstance(p, EMAPolicy)),
            None,
        )
        if ema is not None:
            checkpoint["ema_state_dict"] = ema.ema_state_dict()
        torch.save(checkpoint, os.path.join(path, "checkpoint.pt"))

    # ------------------------------------------------------------------
    # Memory lifecycle
    # ------------------------------------------------------------------

    def offload(self) -> None:
        fsdp = next(
            (p for p in walk_source_chain(self.policy) if isinstance(p, FSDPPolicy)),
            None,
        )
        if fsdp is not None:
            fsdp.offload()
        for state in _iter_optimizer_param_states(self.optimizer):
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        _clear_gpu_memory()

    def onload(self) -> None:
        fsdp = next(
            (p for p in walk_source_chain(self.policy) if isinstance(p, FSDPPolicy)),
            None,
        )
        if fsdp is not None:
            fsdp.onload()
        for state in _iter_optimizer_param_states(self.optimizer):
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self._device)

    # ------------------------------------------------------------------
    # Smoke-only helper for weight-sync e2e tests.
    #
    # Forces every trainable parameter into a fresh, deterministic random
    # state so the post-sync rollout-side checksums MUST differ from
    # pre-sync ones — otherwise a no-op weight-sync handler would silently
    # pass the smoke. Not on the production path; do not call from train.py.
    # ------------------------------------------------------------------

    def compute_local_param_checksums(
        self,
        *,
        names: List[str],
        prefix: str = "",
    ) -> Dict[str, str]:
        """Hash this rank's view of ``raw_state_dict(self.model)`` for ``names``.

        Each name in ``names`` should be the **prefixed** (rollout-side)
        name — e.g. ``transformer.context_embedder.bias``. The trainer
        model holds it as bare ``context_embedder.bias``; this method
        strips ``prefix`` to match.

        Because ``raw_state_dict``'s ``_to_full_tensor`` materializes
        DTensor shards via ``redistribute(Replicate)`` (an all-gather
        collective), every DP rank must call this method
        simultaneously. After the gather every rank holds the same full
        tensor, so the returned ``{name: hex}`` dicts agree across DP
        ranks for the TP-flat names the smoke probes.

        Returns the same SHA-256 prefix ``fingerprint_tensor`` computes
        on the rollout side, so the smoke can assert byte equality.
        """
        from diffusionrl.rollout.engine.vllm_omni.weight_sync.checksum import (
            fingerprint_tensor,
        )
        from diffusionrl.utils.peft_merge import raw_state_dict

        target = set(names)
        out: Dict[str, str] = {}
        for raw_name, param in raw_state_dict(self.model):
            prefixed = prefix + raw_name
            if prefixed in target:
                out[prefixed] = fingerprint_tensor(param)
        return out

    def randomize_weights_for_smoke(self, seed: int = 0) -> None:
        """Replace every trainable param's data with ``torch.randn_like`` values.

        Seeded per-rank so each FSDP shard ends up with reproducible content
        without all ranks landing on the same tensor (which would mask
        broadcast-vs-no-op bugs). The mutation runs through ``policy.parameters()``
        so it respects whatever the Policy stack exposes (LoRA filters to
        adapters; bare FSDP exposes all weights).
        """
        gen = torch.Generator(device=self._device)
        gen.manual_seed(int(seed) + int(self.rank))
        with torch.no_grad():
            for p in self.policy.parameters():
                if not p.requires_grad:
                    continue
                local = p.data
                from torch.distributed.tensor import DTensor

                if isinstance(local, DTensor):
                    shard = local.to_local()
                    shard.copy_(torch.randn(shard.shape, dtype=shard.dtype, device=shard.device, generator=gen))
                else:
                    local.copy_(torch.randn(local.shape, dtype=local.dtype, device=local.device, generator=gen))
        logger.info("Rank %s: randomize_weights_for_smoke complete (seed=%d)", self.rank, seed)


__all__ = ["NewTrainActor"]
