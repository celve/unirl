"""AgenticRolloutEngine — multi-turn (agentic) rollout over a rank-0 coordinator (LIN-522).

See ``docs/async-rollout-service-design.md``. The engine is a ``BaseRolloutEngine``
on a DP-replicated slab. Each worker builds its **own local** inner single-turn
engine + environment (the ``ComposedRolloutEngine`` build-inner pattern) and runs
**one persistent drain loop** that pulls single-trajectory tasks from a central
queue and runs each as a multi-turn agent loop on the inner engine's event loop.

Two roles, one class:

- **Coordinator = rank 0** (``set_batch``, ``BROADCAST + RANK_ZERO``): rank 0 receives
  the whole batch and fans it into ``n × P`` single-trajectory tasks on a FIFO queue,
  then serves ``next_task`` pulls (raw ``Worker.call``) from the draining heads.
- **Driver = each DP head** (``run_drain`` / ``_drain`` / ``_run_one`` / ``_pull``,
  ``DP_SCATTER_HEAD + DP_HEAD``): the dispatch layer routes ``run_drain`` to the DP-head
  ranks (``tp_rank==0``) — with ``tp=1`` that is every worker (flat DP), with a
  grouped-TP rollout one head per TP group (participant ranks stay idle; their inner
  runtime participates in the group's own collective). The persistent drain loop is the
  always-on driver of the inner engine's loop, so concurrent trajectories continuous-
  batch through the inner backend's shared semaphore, and control RPCs (``abort`` /
  ``pause``) reliably ride the driven loop. ``_collect_dp_merge`` flattens the per-head
  ``List[Sample]`` into one list — the trainer calls ``set_batch`` then ``run_drain``.

Weight sync is unchanged (design §8): the drain is a full barrier (returns only when
every trajectory finished), and ``run_drain`` holds the inner backend's loop-lock for
the batch — so the trainer syncs weights into the inner engines *between* rollout
steps, against a quiesced rollout. Delegated verbs below forward to the inner engine.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import ray
import torch

from unirl.config.require import require
from unirl.distributed.group.dispatch import Dispatch, Execute, distributed
from unirl.rollout.engine.agentic.config import AgenticRolloutEngineConfig
from unirl.rollout.engine.base import BaseRolloutEngine
from unirl.types.sample import Sample, _part_with_field
from unirl.types.sampling import total_samples_per_prompt

logger = logging.getLogger(__name__)


class AgenticRolloutEngine(BaseRolloutEngine):
    """Multi-turn rollout engine: rank-0 coordinator over per-worker pull-drain loops."""

    _component_name = "agentic"

    # ------------------------------------------------------------------
    # Construction (mirrors ComposedRolloutEngine: build the inner engine + env)
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: AgenticRolloutEngineConfig,
        *,
        device: Optional[torch.device] = None,
        strategy: Any = None,
        rank: Optional[int] = None,
        model_config: Optional[Any] = None,
    ) -> None:
        require(
            isinstance(config, AgenticRolloutEngineConfig),
            f"AgenticRolloutEngine requires AgenticRolloutEngineConfig; got {type(config).__name__}",
        )
        self.cfg = config
        self.rank = rank

        deps = dict(device=device, rank=rank, model_config=model_config)
        # Each worker builds its OWN local inner engine + environment.
        self._env = config.env  # an Environment (built per worker via its _target_); must be re-entrant
        require(self._env is not None, "AgenticRolloutEngine requires an env (config.env)")
        # Single source of truth for tool schemas (LIN-519): advertise the env's
        # tools to the model through the inner engine's chat template, so a recipe
        # never restates the schema JSON (which could silently drift from the env's
        # actual tools). Mutates the inner CONFIG before it is built; an explicit
        # ``inner.chat_template_kwargs.tools`` in the recipe still wins.
        self._maybe_inject_tool_schemas(config.inner, self._env)
        self._inner: BaseRolloutEngine = config.inner.make_engine(strategy=strategy, **deps)

        self._sp = config.episode_sampling  # per-turn sampling params; carries n via samples_per_prompt
        self._n = total_samples_per_prompt(self._sp)  # GRPO group size
        self._max_turns = int(config.max_turns)
        # Per-worker trajectory cap = the pull gate (distinct from the inner backend's
        # per-request semaphore; see design §5). A trajectory holds a cap slot across
        # its whole life, incl. tool-wait between turns — so siblings keep the GPU busy.
        self._cap = asyncio.Semaphore(int(config.per_worker_concurrency))

        # Adopt the inner engine's loop + drive it through the inner's lock-guarded
        # _run_coro, so the drain loop and the inner's weight-sync verbs serialize on
        # ONE lock — the quiesce boundary (design §6/§8).
        self._init_async_loop(self._inner._loop)

        # Coordinator state: rank 0 owns the FIFO queue (filled by set_batch, drained
        # by the DP-head workers via next_task). No cached worker list — the dispatch
        # layer routes run_drain to the heads, and the coordinator handle is passed in.
        self._queue: Deque[Sample] = deque()
        self._qlock = threading.Lock()

    def _run_coro(self, coro: Any) -> Any:
        # Drive on the inner engine's loop under the inner's lock (shared quiesce).
        return self._inner._run_coro(coro)

    @staticmethod
    def _maybe_inject_tool_schemas(inner_cfg: Any, env: Any) -> None:
        """Copy the env's tool JSON-schemas into the inner engine's chat-template
        kwargs so the model is told about the tools without the recipe restating
        them. No-op when the env exposes no ``tool_schemas`` or the inner config
        has no ``chat_template_kwargs``; an explicit ``tools`` entry is preserved."""
        get_schemas = getattr(env, "tool_schemas", None)
        if not callable(get_schemas) or not hasattr(inner_cfg, "chat_template_kwargs"):
            return
        ctk = dict(inner_cfg.chat_template_kwargs or {})
        ctk.setdefault("tools", get_schemas())
        inner_cfg.chat_template_kwargs = ctk

    # ------------------------------------------------------------------
    # Coordinator (rank 0) — the NCCLWeightSync pattern
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.RANK_ZERO)
    def set_batch(self, request: Sample) -> None:
        """Rank 0 fills the trajectory queue for one rollout batch.

        Fans ``request`` into ``n × P`` single-trajectory tasks (the ``n`` GRPO
        siblings of a prompt share its slash-free root id — design §3.1) onto rank 0's
        FIFO. The DP-head workers then pull from rank 0 via :meth:`next_task` while
        draining (:meth:`run_drain`). Paired at the trainer:
        ``handle.set_batch(batch); trajs = handle.run_drain(handle.workers[0], role)``.
        """
        # n sibling tasks per prompt; siblings share the prompt's root id (group-by-root downstream).
        tasks = [prompt for prompt in request.split() for _ in range(self._n)]
        with self._qlock:
            self._queue = deque(tasks)

    def next_task(self, worker_rank: int) -> Optional[Sample]:
        """Hand out the next trajectory task, or ``None`` when the queue is drained.

        Reached by raw ``Worker.call`` from pulling workers (un-decorated). FIFO for
        v1; this is the seam where sticky-prefix / staleness-aware routing later
        lives (design §11). ``worker_rank`` is an (unused) hint for that future.
        """
        del worker_rank
        with self._qlock:
            return self._queue.popleft() if self._queue else None

    # ------------------------------------------------------------------
    # Per-worker execution — the persistent drain loop (the loop driver)
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.DP_SCATTER_HEAD, execute_mode=Execute.DP_HEAD)
    def run_drain(self, coordinator: Any, role_name: str) -> List[Sample]:
        """Drive one DP-head's drain loop for the whole batch; return its trajectories.

        Dispatched to DP heads only (``Execute.DP_HEAD``): with ``tp=1`` that is every
        worker (the flat-DP path, unchanged); with a grouped-TP rollout it is one head
        per TP group — participant ranks stay idle while their inner runtime
        participates in the group's own collective. ``coordinator`` (rank 0's Worker
        handle = the queue owner) + ``role_name`` are broadcast to the heads by
        ``DP_SCATTER_HEAD``; each head pulls tasks from ``coordinator`` via
        :meth:`next_task`. ``_collect_dp_merge`` flattens the per-head ``List[Sample]``
        into one flat list, so the trainer needs no ``[0]`` unwrap.

        Sync at the RPC boundary; internally one ``run_until_complete`` (via the
        inner's ``_run_coro``) is the single driver of the inner engine's loop — so
        concurrent trajectories are coroutines on one loop (continuous batching) and
        hold the inner backend's loop-lock for the batch (the quiesce boundary).
        """
        return self._run_coro(self._drain(coordinator, role_name))

    async def _drain(self, coordinator: Any, role_name: str) -> List[Sample]:
        out: List[Sample] = []
        inflight: set = set()

        def _done(fut: "asyncio.Future") -> None:
            self._cap.release()
            inflight.discard(fut)
            out.append(fut.result())  # _run_one never raises (failure-isolated)

        while True:
            await self._cap.acquire()  # block until a trajectory slot frees
            task = await self._pull(coordinator, role_name)
            if task is None:  # sentinel: queue drained
                self._cap.release()
                break
            fut = asyncio.ensure_future(self._run_one(task))
            inflight.add(fut)
            fut.add_done_callback(_done)

        if inflight:
            await asyncio.gather(*list(inflight))  # let the last in-flight trajectories finish
        return out

    async def _run_one(self, task: Sample) -> Sample:
        """One trajectory's agent loop. Failure-isolated: never raises into the drain."""
        sample = task
        env_reward: Optional[float] = None
        try:
            sample = self._env.reset(task)  # [input(1)], root id = prompt id
            for _ in range(self._max_turns):
                sample = await self._inner.agenerate(sample.fork(1, sampling_params=self._sp))  # +[gen(1)]
                observation, done, info = await self._env.astep(sample)  # async tool boundary (§7)
                # Env-sourced reward (LIN-519): interactive envs (ALFWorld, …) return a
                # per-trajectory return in ``info["reward"]`` (last value = the episode
                # return); tool-only envs (calculator/search) omit it — a no-op here.
                if isinstance(info, dict) and info.get("reward") is not None:
                    env_reward = float(info["reward"])
                if done:
                    break
                if observation is not None:
                    sample = sample.observe(observation)  # +[obs(1)]
            return self._attach_env_reward(sample, env_reward)
        except Exception as exc:  # noqa: BLE001 — isolate: one bad trajectory must not sink the drain
            logger.warning("AgenticRolloutEngine: trajectory failed, returning partial: %s", exc, exc_info=True)
            return self._attach_env_reward(sample, env_reward)

    @staticmethod
    def _attach_env_reward(sample: Sample, reward: Optional[float]) -> Sample:
        """Attach an env-sourced trajectory return to the LAST generated Part so the
        trainer (:class:`~unirl.trainer.agentic_env.AgenticEnvTrainer`) can read it
        directly — env tasks bypass ``RewardService``. No-op when the env supplied no
        reward, so tool-only envs (calculator/search) are byte-identical."""
        if reward is None:
            return sample
        gens = sample.gen_parts()
        if not gens:
            return sample
        last = gens[-1]
        rewarded = _part_with_field(
            last, "rewards", torch.full((int(last.batch_size),), float(reward), dtype=torch.float32)
        )
        return sample.with_parts([rewarded if p is last else p for p in sample.parts])

    async def _pull(self, coordinator: Any, role_name: str) -> Optional[Sample]:
        """Pull the next task from the coordinator. Bridges the Ray RPC onto the
        backend loop via ``run_in_executor`` (a Ray ObjectRef isn't awaitable on
        that loop) — the same trick :meth:`Environment.astep` uses for slow tools."""
        loop = asyncio.get_running_loop()
        ref = coordinator.call.remote(role_name, "next_task", (self.rank or 0,), {})
        return await loop.run_in_executor(None, ray.get, ref)

    # ------------------------------------------------------------------
    # Lifecycle + control plane — delegated to the inner engine
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def shutdown(self) -> None:
        self._inner.shutdown()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sleep(self) -> None:
        self._inner.sleep()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def wake_up(self) -> None:
        self._inner.wake_up()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def onload_weights(self, *, track_prefix: str = "") -> None:
        self._inner.onload_weights(track_prefix=track_prefix)

    @property
    def is_offloaded(self) -> bool:
        return self._inner.is_offloaded

    def health_check(self) -> bool:
        return self._inner.health_check()

    def get_memory_info(self) -> Dict[str, float]:
        return self._inner.get_memory_info()

    def abort(self, ids: Optional[List[str]] = None) -> List[Sample]:
        return self._inner.abort(ids)

    def pause(self) -> None:
        self._inner.pause()

    def resume(self) -> None:
        self._inner.resume()

    # ------------------------------------------------------------------
    # Weight sync — delegated to the inner engine (single inner, no prefix routing).
    # Reached via raw Worker.call by the weight-sync driver (e.g. NCCLWeightSync).
    # ------------------------------------------------------------------

    def init_weights_update_group(self, **kwargs: Any) -> None:
        self._inner.init_weights_update_group(**kwargs)

    def update_weights_from_distributed(self, **kwargs: Any) -> None:
        self._inner.update_weights_from_distributed(**kwargs)

    def destroy_weights_update_group(self, **kwargs: Any) -> None:
        self._inner.destroy_weights_update_group(**kwargs)

    def update_weights_from_ipc(self, **kwargs: Any) -> None:
        self._inner.update_weights_from_ipc(**kwargs)

    def update_weights_from_tensor(self, **kwargs: Any) -> None:
        self._inner.update_weights_from_tensor(**kwargs)

    def set_lora_from_tensors(self, adapter_name: str, lora_tensors: Dict[str, Any], **kwargs: Any) -> None:
        self._inner.set_lora_from_tensors(adapter_name, lora_tensors, **kwargs)


__all__ = ["AgenticRolloutEngine"]
