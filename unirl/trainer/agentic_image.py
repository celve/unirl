"""AgenticImageTrainer — joint RL over the agentic image rollout engine (LIN-577).

The trainer half of the in-loop agentic image stack: a multi-turn agent renders
images mid-trajectory (:class:`~unirl.rollout.engine.agentic.image_engine.
AgenticImageRolloutEngine` with ``in_loop_images``), and ONE terminal image reward
trains **both** models — the agent LLM that decided what to draw and critique, and
the diffusion renderer that drew it.

This is the merge the LIN-577 plan called for, and it is exactly the two existing
halves glued at the advantage:

- :class:`~unirl.trainer.pe.PETrainer` supplies the *placement*: two ``TrainStack``
  siblings (``ar`` / ``diffusion``), per-track weight sync keyed by ``track_prefix``
  — the same ``{ar, diffusion}`` keys the rollout engine demuxes on — checkpointing
  of both sides, and the ``freeze_llm`` toggle. Inherited wholesale.
- :class:`~unirl.trainer.agentic.AgenticTrainer` supplies the *shape*: the rollout
  returns ``List[Sample]`` trajectories of variable depth, a scalar reward per
  trajectory, a group-relative advantage over the ``n`` siblings of a prompt, and
  that advantage broadcast to every generated turn before ONE padded step.

**What is new here.** PE's trajectory is a fixed ``[input, ar, diffusion]`` chain,
so it locates each track with a single ``gen_part_index``. An in-loop agentic
trajectory is ``[input, ar, img, ar, img, …]`` — *several* Parts per track, a
different count per trajectory. So the routing is a partition, not an index: every
gen Part is bucketed by its ``sampling_params`` type, each bucket concatenated into
one padded training Part, and each handed to its own stack. Depth heterogeneity is
absorbed by the same concat-and-pad the agentic trainer already uses for ragged
turn counts.

**Credit assignment.** The terminal image is the outcome, so every action that led
to it — each agent turn AND each render — carries the trajectory's advantage. This
is PE's ``propagate_rewards`` generalized from one hop to a variable trajectory:
there, the image reward flows to the single ``ar`` Part; here it flows to all of
them. A NaN reward (crashed trajectory) is excluded from the group's mean/std and
given zero advantage, so an infrastructure fault neither rewards nor penalizes.

Deferred, and worth knowing before reading the metrics: only the TERMINAL image is
scored, so intermediate renders are trained on the trajectory's outcome rather than
their own quality. Scoring every render would give the diffusion track a denser,
better-attributed signal — the natural next step once this loop is shown to move.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import torch

from unirl.algorithms.normalizers import build_group_index_map
from unirl.distributed.tensor import hydrate
from unirl.train.stack import TrainStepResult
from unirl.trainer.base import prepare_input_sample
from unirl.trainer.pe import TRACK_NAMES, PETrainer
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Part, Sample, _part_with_field
from unirl.types.sampling import DiffusionSamplingParams

logger = logging.getLogger(__name__)


class AgenticImageTrainer(PETrainer):
    """Multi-turn agentic RL training an agent LLM and a diffusion renderer jointly."""

    def __init__(self, *, stop: Optional[List[str]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Per-turn stop: a tool-call turn ends at ``</tool_call>`` and yields to the
        # environment; a final-answer turn runs to EOS. Rides the request root's
        # control bag, which ``resolve_sampling`` reads as ``control["ar"]``.
        self._stop = list(stop) if stop else ["</tool_call>"]
        # Wire the rank-0 coordinator (the agentic engine's ``set_workers`` contract);
        # ``.workers`` / ``.role_name`` are Handle attributes.
        self.rollout.set_workers(self.rollout.workers, self.rollout.role_name)

    # ------------------------------------------------------------------
    # Request — the agentic engine fans its own siblings
    # ------------------------------------------------------------------

    def _build_request_sample(
        self,
        inputs: Sample,
        rollout_id: int,
        *,
        sampling: Optional[Dict[str, Any]] = None,
    ) -> Sample:
        """The ``P`` prompts as an input-only Sample — NO ``fork``.

        Unlike PE's composed engine, the agentic engine fans the ``n`` GRPO siblings
        itself from ``episode_sampling``, so forking here would multiply them.
        """
        del sampling  # the engine's episode_sampling owns per-turn params and n
        return prepare_input_sample(
            inputs,
            rollout_id,
            allowed_primitives={"text", "image"},
            caller="AgenticImageTrainer._build_request_sample",
            root_control={"ar": {"stop": list(self._stop)}},
        )

    # ------------------------------------------------------------------
    # One rollout -> reward -> advantage -> two stacks
    # ------------------------------------------------------------------

    def train_step(
        self,
        sample: Sample,
        *,
        training_progress: float = 0.0,
        sync_weights: bool = False,
        rollout_id: int = 0,
    ) -> Tuple[Dict[str, TrainStepResult], float]:
        """One ``rollout → score terminal image → advantage → two steps`` pass."""
        t0 = time.perf_counter()

        # 1) Rollout. On-policy: push freshly-trained weights before generating.
        self.rollout.wake_up()
        if sync_weights:
            if self.diffusion_sync is not None:
                self.diffusion_sync.sync()
            if self.ar_sync is not None:
                self.ar_sync.sync()
        do_offload = self._enable_fsdp_offload and not self._rollout_is_trainside
        if do_offload:
            self.diffusion.backend.offload()
            if self.ar.backend is not None:
                self.ar.backend.offload()
        trajs: List[Sample] = self.rollout.generate(sample)[0]  # BROADCAST+RANK_ZERO -> [List[Sample]]
        self.rollout.sleep()
        if do_offload:
            self.diffusion.backend.onload()
            if self.ar.backend is not None:
                self.ar.backend.onload()

        # 2) Score each trajectory's TERMINAL image.
        rewards, group_ids, scored_idx = self._score_terminal_images(trajs, rollout_id)
        if rewards.numel() == 0:
            # Nothing rendered this round — almost always the agent not emitting a
            # usable draw call (bad system instruction, or it answered in text).
            # Skip the step loudly instead of crashing downstream on an empty batch:
            # a run that silently trains on nothing is worse than one that says so.
            turns = [len(tr.gen_parts()) for tr in trajs]
            logger.warning(
                "AgenticImageTrainer rollout %d: 0/%d trajectories rendered an image "
                "(turns mean=%.2f max=%d) — skipping the step. Check the agent's "
                "system_instruction and that the draw tool name matches draw_tool_name.",
                rollout_id,
                len(trajs),
                (sum(turns) / len(turns)) if turns else 0.0,
                max(turns, default=0),
            )
            return {}, 0.0
        finite = torch.isfinite(rewards)
        mean_reward = float(rewards[finite].mean().item()) if bool(finite.any()) else 0.0

        # 3) Group-relative advantage over the n siblings of each prompt.
        advantages = self._trajectory_advantages(rewards, group_ids)

        # 4) Partition every gen Part by track and stamp the trajectory's advantage.
        by_track = self._partition_by_track(trajs, advantages, scored_idx)

        results = self._train_tracks_and_log(
            by_track, trajs, rewards, advantages, rollout_id=rollout_id, training_progress=training_progress, t0=t0
        )
        return results, mean_reward

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _score_terminal_images(self, trajs: List[Sample], rollout_id: int) -> Tuple[torch.Tensor, List[str], List[int]]:
        """Score each trajectory's last rendered image; returns ``(rewards, group_ids, idx)``.

        ``idx`` are the indices of trajectories that actually rendered something —
        a trajectory that never called ``draw`` (or crashed before it) has nothing
        to score and is dropped from training rather than counted as a zero, which
        would be a policy signal it did not earn.

        A FRESH flat Sample is built for scoring: ``score_and_attach`` rejects a
        frontier that already carries rewards, and it reads its prompt context from
        the lineage — so the terminal images are re-rooted onto the ORIGINAL task
        prompt (not the agent's intermediate draw text), scoring image-vs-intent.
        """
        prompts: List[str] = []
        images: List[Any] = []
        group_ids: List[str] = []
        idx: List[int] = []
        for i, tr in enumerate(trajs):
            renders = [p for p in tr.gen_parts() if isinstance(p.sampling_params, DiffusionSamplingParams)]
            if not renders:
                continue
            terminal = renders[-1].primitives.get("image")
            if not isinstance(terminal, Images):
                continue
            # The engine returns worker-side TensorRefs; realize before indexing
            # (TensorRef supports slices only, so ``to_list()`` would raise).
            pixels = hydrate(terminal.pixels)
            if pixels is None or pixels.shape[0] == 0:
                continue
            root = tr.parts[0]
            task = root.primitives.get("text")
            prompts.append(task.texts[0] if isinstance(task, Texts) and task.texts else "")
            images.append(pixels[-1])  # M==1 in-loop; last row if a recipe widened it
            group_ids.append(root.sample_ids[0])  # siblings of one prompt share its root id
            idx.append(i)

        if not idx:
            logger.warning("AgenticImageTrainer rollout %d: no trajectory rendered an image.", rollout_id)
            return torch.zeros(0, dtype=torch.float32), [], []

        diff_sp = self.sampling_params.get("diffusion")
        root = Part.input(
            [f"score{rollout_id}:{k}" for k in range(len(idx))],
            primitives={"text": Texts(texts=prompts)},
        )
        shell = root.fork(1, sampling_params=diff_sp)
        scoring = Sample(parts=[root, shell]).with_filled_frontier(
            primitives={"image": Images(pixels=torch.stack(images))}
        )
        scored = self.reward.score_and_attach(scoring)
        raw = scored.parts[-1].rewards
        if raw is None:
            raise ValueError("AgenticImageTrainer: reward service returned no rewards for the terminal images")
        return hydrate(raw).to(torch.float32).reshape(-1), group_ids, idx

    # ------------------------------------------------------------------
    # Advantage
    # ------------------------------------------------------------------

    def _trajectory_advantages(self, rewards: torch.Tensor, group_ids: List[str]) -> torch.Tensor:
        """Group-relative GRPO advantage per trajectory (population std).

        NaN reward = crashed trajectory: excluded from its group's mean/std and given
        ZERO advantage, so an infrastructure fault is neutral rather than a strong
        negative the policy would learn from.
        """
        r = rewards.to(torch.float32)
        if r.numel() == 0:
            return r
        finite = torch.isfinite(r)
        adv = torch.zeros_like(r)
        for idxs in build_group_index_map(group_ids).values():
            sel = torch.tensor(idxs, dtype=torch.long)
            fin = finite[sel]
            if not bool(fin.any()):
                continue
            g = r[sel]
            gf = g[fin]
            centered = g - gf.mean()
            if gf.numel() > 1:
                centered = centered / (gf.std(unbiased=False) + 1e-8)
            adv[sel] = torch.where(fin, centered, torch.zeros_like(centered))
        return adv

    # ------------------------------------------------------------------
    # Routing — the PE index becomes a partition
    # ------------------------------------------------------------------

    def _partition_by_track(
        self, trajs: List[Sample], advantages: torch.Tensor, scored_idx: List[int]
    ) -> Dict[str, List[Part]]:
        """Bucket every gen Part of every scored trajectory by track.

        PE can index one Part per track because its chain is fixed; an in-loop
        trajectory holds several of each, so the Parts are partitioned by
        ``sampling_params`` type — diffusion renders to ``diffusion``, agent turns
        to ``ar`` — and every Part inherits its trajectory's scalar advantage.
        """
        out: Dict[str, List[Part]] = {name: [] for name in TRACK_NAMES}
        for slot, i in enumerate(scored_idx):
            adv_i = float(advantages[slot].item())
            for gp in trajs[i].gen_parts():
                track = "diffusion" if isinstance(gp.sampling_params, DiffusionSamplingParams) else "ar"
                gp = _part_with_field(gp, "advantages", torch.full((gp.batch_size,), adv_i, dtype=torch.float32))
                # Free decoded media/text before the train pass, and drop any
                # per-turn reward: it rides only some turns, so concatenating would
                # leave a short, misaligned rewards field that breaks the DP scatter.
                # The reward is already folded into ``advantages`` above.
                gp = _part_with_field(gp, "primitives", {})
                gp = _part_with_field(gp, "rewards", None)
                out[track].append(gp)
        return out

    def _pad_track(self, part: Part, stack: Any) -> Part:
        """Pad to a multiple of that track's DP size with zero-advantage rows.

        Replicates the shortest row so the ragged Σ-turns batch satisfies
        ``pytree_chunk``'s divisibility check; advantage 0 makes the padding a
        zero-gradient no-op for GRPO/DRPO/CPPO.
        """
        # Read dp off the Handle when it exposes a plain int; a Handle attribute can
        # be a proxy that coerces to a wrong/1 value, which silently under-pads and
        # then trips the stack's own divisibility check.
        raw = getattr(stack, "dp_size", None)
        dp = raw if isinstance(raw, int) and raw > 0 else int(getattr(self, "num_devices", 1) or 1)
        pad = (-int(part.batch_size)) % dp
        if pad == 0:
            return part
        lengths = part.segment.lengths if part.segment is not None else None
        src = int(torch.argmin(lengths).item()) if (lengths is not None and lengths.numel()) else 0
        block = part.select(torch.full((pad,), src, dtype=torch.long))
        block = _part_with_field(block, "advantages", torch.zeros(pad, dtype=torch.float32))
        return Part.concat([part, block])

    def _train_tracks_and_log(
        self,
        by_track: Dict[str, List[Part]],
        trajs: List[Sample],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        *,
        rollout_id: int,
        training_progress: float,
        t0: float,
    ) -> Dict[str, TrainStepResult]:
        """One padded step per trained track, then log."""
        results: Dict[str, TrainStepResult] = {}
        rows: Dict[str, int] = {}
        for name in self._train_tracks:
            parts = by_track.get(name) or []
            if not parts:
                logger.warning("AgenticImageTrainer rollout %d: track %r had no trainable Parts.", rollout_id, name)
                continue
            stack = getattr(self, name).stack
            train_part = self._pad_track(Part.concat(parts), stack)
            rows[name] = int(train_part.batch_size)
            results[name] = stack.train_track(train_part, training_progress=float(training_progress))

        renders = [
            len([p for p in tr.gen_parts() if isinstance(p.sampling_params, DiffusionSamplingParams)]) for tr in trajs
        ]
        turns = [len(tr.gen_parts()) for tr in trajs]
        logger.info(
            "rollout %d: %d trajectories, turns mean=%.2f max=%d, renders mean=%.2f hist=%s",
            rollout_id,
            len(trajs),
            (sum(turns) / len(turns)) if turns else 0.0,
            max(turns, default=0),
            (sum(renders) / len(renders)) if renders else 0.0,
            dict(sorted(Counter(renders).items())),
        )
        metrics: Dict[str, Any] = {
            "agent/mean_turns": (sum(turns) / len(turns)) if turns else 0.0,
            "agent/max_turns": max(turns, default=0),
            "agent/mean_renders": (sum(renders) / len(renders)) if renders else 0.0,
            "agent/no_render_trajectories": sum(1 for r in renders if r == 0),
            "agent/scored_trajectories": int(rewards.numel()),
        }
        metrics.update({f"agent/train_rows_{k}": v for k, v in rows.items()})

        log_sample = self._build_log_sample(trajs, rewards, advantages, rollout_id)
        self.wandb_logger.log_rollout_step(
            rollout_id,
            results,
            log_sample,
            step_time_s=time.perf_counter() - t0,
            extra_metrics=metrics,
        )
        return results

    def _build_log_sample(
        self, trajs: List[Sample], rewards: torch.Tensor, advantages: torch.Tensor, rollout_id: int
    ) -> Sample:
        """A flat one-row-per-scored-trajectory Sample carrying reward + advantage,
        for ``compute_rollout_sample_metrics``' reward/advantage distributions."""
        m = int(rewards.numel())
        diff_sp = self.sampling_params.get("diffusion")
        root = Part.input([f"log{rollout_id}:{i}" for i in range(m)], primitives={"text": Texts(texts=[""] * m)})
        log_sample = Sample(parts=[root, root.fork(1, sampling_params=diff_sp)])
        frontier = _part_with_field(log_sample.parts[-1], "rewards", rewards.clone())
        frontier = _part_with_field(frontier, "advantages", advantages.clone())
        return log_sample.replace_frontier(frontier)

    def evaluate(self, rollout_id: int) -> float:
        raise NotImplementedError(
            "AgenticImageTrainer.evaluate is not implemented: the agentic engine returns "
            "List[Sample], not a Sample. Set eval_interval=0 (agentic eval is a follow-up)."
        )


__all__ = ["AgenticImageTrainer"]
