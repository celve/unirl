"""AgenticImageRolloutEngine — agentic loop + terminal diffusion image gen (LIN-577).

A specialization of :class:`AgenticRolloutEngine`: after the multi-turn LLM agent
loop finishes a trajectory, its final answer conditions a diffusion image
generation, and the image is appended as a terminal generation Part on the SAME
lineage (connected lineage). This is the multi-turn analogue of
:class:`~unirl.rollout.engine.composed.engine.ComposedRolloutEngine`'s single-turn
prompt-enhancement flow.

Why a subclass and not a two-child composed engine: the agentic engine is a
**rank-0 coordinator** whose generation runs IN-PROCESS on every worker
(``run_drain -> _drain_worker -> _run_one``). So the diffusion step belongs
inside the per-worker drain — that worker diffuses its own finished
trajectories — not as a rank-0 second pass, which sidesteps the
rank-0-vs-``DP_SCATTER`` dispatch mismatch. Subclassing inherits the whole
coordinator/drain surface unchanged (``set_workers`` / ``submit`` / ``poll`` /
``abort`` / ``generate`` / ``drained`` / ``next_task`` / ``drain_*``); only
``run_drain`` (adds the terminal diffusion phase), weight sync (``track_prefix``
demux over ``{ar, diffusion}``) and lifecycle (cover both children) are
overridden.

Two shapes, selected by ``config.in_loop_images``:

**Terminal** (``False``, the v1 shape). The agent loop is all-text; after
``run_drain`` joins every drain thread, each finished trajectory's final answer
conditions ONE diffusion generation, appended as a terminal gen Part. Cheap,
because the whole worker's trajectories batch into a single diffusion call.

**In-loop** (``True``). The agent renders *mid*-trajectory by calling the ``draw``
tool: the image lands on the trajectory as a **trainable** diffusion gen Part
(carrying its ``LatentSegment``), the agent sees it on the next turn, and a later
draw edits it (ti2i). Three constraints make this work, each load-bearing:

1. *The image Part is tagged* ``role="tool"``. A gen Part's role resolves to
   ``"assistant"``, and ``build_vision_conversations`` fuses **consecutive
   same-role** turns into one message ordered ``image_blocks + text_blocks`` — so
   an untagged image would collapse draw/critique/redraw into a single assistant
   message with every image hoisted in front of every text, destroying the
   temporal structure the agent needs to refine. Role is orthogonal to ``is_gen``,
   so the tagged Part stays trainable.
2. *Requests are re-rooted, then written back.* The ti2i adapter demands exactly
   one text and one image turn, which a multi-turn trajectory never satisfies. The
   batcher re-roots onto a fresh ``{"text", "image"}`` input Part (one turn per
   modality via ``PRIMITIVE_MODALITY_ORDER``) and this engine writes the resulting
   block back onto the real lineage.
3. *Concurrent image turns are coalesced* (:class:`DiffusionTurnBatcher`). The
   diffusion engines serialize concurrent ``generate`` callers, unlike the AR
   inner whose backend batches them, so K drain threads would otherwise cost K
   sequential passes.

Still deferred (see the LIN-577 plan): the joint trainer, a runnable recipe, and
partial/async diffusion timing.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from unirl.config.require import require
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.rollout.engine.agentic.engine import AgenticRolloutEngine
from unirl.rollout.engine.agentic.image_batcher import DiffusionTurnBatcher
from unirl.rollout.engine.agentic.image_config import AgenticImageRolloutEngineConfig
from unirl.rollout.engine.base import BaseSingleTurnRolloutEngine
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Part, Primitive, Sample, _part_with_field
from unirl.types.sampling import total_samples_per_prompt

logger = logging.getLogger(__name__)

# Mirrors ``unirl.trainer.agentic._extract_answer``; kept local to avoid a
# rollout -> trainer import (the rollout layer must not depend on the trainer).
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _extract_answer(text: Optional[str]) -> str:
    """The last ``<answer>...</answer>`` span, else the whole (stripped) text."""
    if not text:
        return ""
    matches = list(_ANSWER_RE.finditer(text))
    return matches[-1].group(1).strip() if matches else text.strip()


#: Argument names a model reaches for when it has to guess the draw schema.
_PROMPT_ALIASES = ("prompt", "description", "text", "caption", "image_prompt", "query")


def _draw_prompt(args: Any) -> Optional[str]:
    """The visual description out of a draw call's arguments, or ``None``.

    Chat templates vary in whether they inject tool schemas at all — Qwen2.5-VL's
    ignores ``tools=`` outright — so a model told about ``draw`` only in prose will
    invent its own argument names (``{"subject": …, "action": …, "setting": …}``).
    Rejecting those would silently drop nearly every render, so try the usual
    aliases and otherwise stitch the string values together in order. Returns
    ``None`` only when there is no usable text at all, which the caller surfaces as
    an ordinary tool observation so the agent can correct itself.
    """
    if not isinstance(args, dict):
        return None
    for key in _PROMPT_ALIASES:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    parts = [v.strip() for v in args.values() if isinstance(v, str) and v.strip()]
    return ", ".join(parts) if parts else None


def _shutdown_quietly(engine: Any) -> None:
    shutdown = getattr(engine, "shutdown", None)
    if not callable(shutdown):
        return
    try:
        shutdown()
    except Exception as exc:  # noqa: BLE001 — cleanup after a construction failure
        logger.warning("AgenticImageRolloutEngine: child cleanup after ctor failure raised: %s", exc)


class AgenticImageRolloutEngine(AgenticRolloutEngine):
    """Agentic multi-turn LLM loop with diffusion image turns (terminal or in-loop)."""

    _component_name = "agentic_image"

    def __init__(
        self,
        config: AgenticImageRolloutEngineConfig,
        *,
        device: Optional[Any] = None,
        strategy: Any = None,
        rank: Optional[int] = None,
        model_config: Optional[Any] = None,
    ) -> None:
        require(
            isinstance(config, AgenticImageRolloutEngineConfig),
            f"AgenticImageRolloutEngine requires AgenticImageRolloutEngineConfig; got {type(config).__name__}",
        )
        require(config.diffusion is not None, "AgenticImageRolloutEngine requires a diffusion child (config.diffusion)")
        require(
            config.diffusion_sampling is not None,
            "AgenticImageRolloutEngine requires config.diffusion_sampling (DiffusionSamplingParams)",
        )

        # Builds the LLM inner (self._inner) + env + rank-0 coordinator state.
        super().__init__(config, device=device, strategy=strategy, rank=rank, model_config=model_config)

        # Build the co-located diffusion child (mirrors ComposedRolloutEngine). The
        # SDE ``strategy`` is a diffusion concept; the LLM inner (built by super)
        # ignores it.
        try:
            diffusion = config.diffusion.make_engine(
                strategy=strategy, device=device, rank=rank, model_config=model_config
            )
            require(
                isinstance(diffusion, BaseSingleTurnRolloutEngine),
                "AgenticImageRolloutEngine diffusion child must be a BaseSingleTurnRolloutEngine; "
                f"got {type(diffusion).__name__}",
            )
        except BaseException:
            _shutdown_quietly(self._inner)  # don't leak the already-built LLM inner
            raise

        self._diffusion = diffusion
        # Same keys the PE recipe's ``sync.{ar,diffusion}.track_prefix`` uses.
        self._child_by_name: Dict[str, BaseSingleTurnRolloutEngine] = {
            "ar": self._inner,
            "diffusion": self._diffusion,
        }
        self._diff_sp = config.diffusion_sampling
        self._diff_M = total_samples_per_prompt(self._diff_sp)
        require(
            self._diff_M >= 1,
            f"AgenticImageRolloutEngine: diffusion samples_per_prompt (M={self._diff_M}) must be >= 1",
        )
        self._answer_marker = config.answer_marker
        self._answer_max_chars = config.answer_max_chars
        # Colocated diffusion: wake/sleep it around the terminal phase (shared slab).
        self._colocate = bool(config.sleep_diffusion_on_start)

        # In-loop image turns (see the module docstring). Validated here rather than
        # discovered mid-drain, where a bad shape would surface as a mangled
        # conversation or a silently branched trajectory.
        self._in_loop = bool(config.in_loop_images)
        self._draw_tool = str(config.draw_tool_name)
        self._batcher: Optional[DiffusionTurnBatcher] = None
        if self._in_loop:
            require(
                self._diff_M == 1,
                "AgenticImageRolloutEngine: in_loop_images requires diffusion samples_per_prompt == 1 "
                f"(got M={self._diff_M}); forking M>1 mid-trajectory branches it and leaves later AR "
                "turns M-wide. Fan GRPO siblings with the agentic engine's n instead.",
            )
            require(
                not self._colocate,
                "AgenticImageRolloutEngine: in_loop_images requires sleep_diffusion_on_start=false. "
                "Per-phase wake/sleep cannot work when diffusion turns interleave with in-flight AR "
                "turns on other drain threads; keep both children resident and budget them with "
                "per-engine mem_fraction_static (see examples/pe/pe_sglang_full_wise.yaml).",
            )
            require(
                bool(self._draw_tool),
                "AgenticImageRolloutEngine: in_loop_images requires a non-empty draw_tool_name",
            )
            self._batcher = DiffusionTurnBatcher(
                self._diffusion,
                self._diff_sp,
                max_batch=self._concurrency,
                window_s=float(config.draw_batch_window_s),
            )

        if self._colocate:
            self._diffusion.sleep()

    # ------------------------------------------------------------------
    # Per-worker drain — LLM trajectories, then the terminal diffusion phase
    # ------------------------------------------------------------------

    def run_drain(self, coordinator: Any, role_name: str) -> None:
        """Run this worker's drain, rendering images in-loop or after the barrier.

        **Terminal mode.** ``super().run_drain`` joins every drain thread before
        returning (barrier), so the inner engine is idle and ``self._completed``
        holds this worker's finished trajectories. We then generate one image batch
        (M per trajectory) conditioned on each trajectory's final answer and append
        it as a terminal diffusion Part (connected lineage).

        **In-loop mode.** The images were already rendered *inside* the loop by
        :meth:`_observe`, so there is no terminal phase — only the batcher's
        lifetime to bracket. It is stopped in ``finally`` so a drain that raises
        cannot strand a caller parked on a batch, and so the next drive starts with
        a fresh collector.
        """
        if not self._in_loop:
            super().run_drain(coordinator, role_name)
            self._diffuse_completed()
            return
        self._batcher.start()
        try:
            super().run_drain(coordinator, role_name)
        finally:
            self._batcher.stop()

    # ------------------------------------------------------------------
    # In-loop image turns — the agent draws, sees, and redraws
    # ------------------------------------------------------------------

    def _observe(self, sample: Sample, observation: Primitive, info: Any) -> Sample:
        """Render a ``draw`` call as a diffusion turn; otherwise observe normally.

        The environment already decided the loop continues and surfaced the parsed
        call in ``info["tool_calls"]``; this only decides *which engine* renders the
        turn. A malformed draw (no usable prompt) deliberately falls through to the
        ordinary text observation, so the tool's own error message reaches the model
        and the agent can correct itself rather than the trajectory dying.
        """
        if not self._in_loop:
            return super()._observe(sample, observation, info)
        draw = self._draw_request(info)
        if draw is None:
            return super()._observe(sample, observation, info)
        return self._image_turn(sample, draw)

    def _draw_request(self, info: Any) -> Optional[Dict[str, Any]]:
        """The frontier row's draw call, or ``None`` if this turn is not a draw.

        One trajectory is one row in the drain (``_run_one`` forks width 1), so
        ``tool_calls`` carries exactly one entry to inspect.
        """
        calls = info.get("tool_calls") if isinstance(info, dict) else None
        if not calls:
            return None
        call = calls[0]
        if not isinstance(call, dict) or call.get("name") != self._draw_tool:
            return None
        args = call.get("arguments") or {}
        prompt = _draw_prompt(args)
        if prompt is None:
            logger.warning(
                "AgenticImageRolloutEngine: %r call without a usable prompt (args=%s); observing instead",
                self._draw_tool,
                sorted(args) if isinstance(args, dict) else type(args).__name__,
            )
            return None
        return {"prompt": prompt, "edit": args.get("edit")}

    def _image_turn(self, sample: Sample, draw: Dict[str, Any]) -> Sample:
        """Append one rendered image as a **trainable** diffusion gen Part.

        Edits the trajectory's most recent image (ti2i) when there is one and the
        call did not opt out; otherwise renders from scratch (t2i). The batcher
        re-roots the request — the ti2i adapter needs exactly 1 text + 1 image turn —
        and returns the filled block, which is written back here onto the real
        lineage so credit assignment still sees one connected trajectory.

        The Part is tagged ``role="tool"``: a gen Part would otherwise resolve to
        ``"assistant"`` and fuse with the surrounding agent text into one message
        (see the module docstring). Tagging keeps the turn structure intact without
        touching ``is_gen``, so the denoise trajectory stays a training target.
        """
        previous = self._last_image(sample)
        edit = draw["edit"]
        source = previous if (previous is not None and (edit is None or bool(edit))) else None
        block = self._batcher.generate(draw["prompt"], source)
        filled = sample.fork(1, sampling_params=self._diff_sp).with_filled_frontier(
            segment=block.segment,
            primitives=dict(block.primitives),
            primitive_metadata=dict(block.primitive_metadata),
            conditions=dict(block.conditions),
            media_preview=block.media_preview,
            weight_version=block.weight_version,
        )
        return filled.replace_frontier(_part_with_field(filled.parts[-1], "role", "tool"))

    @staticmethod
    def _last_image(sample: Sample) -> Optional[Images]:
        """The trajectory's most recent image, or ``None`` before the first draw."""
        for part in reversed(sample.parts):
            image = part.primitives.get("image")
            if isinstance(image, Images) and len(image) > 0:
                return image
        return None

    # ------------------------------------------------------------------
    # Terminal image turn — one diffusion gen after the loop finishes
    # ------------------------------------------------------------------

    def _diffuse_completed(self) -> None:
        """Diffuse this worker's terminal trajectories and append the image Parts.

        Snapshot + clear ``self._completed`` under the buffer lock (safe: super's
        drain joined all threads, so nothing is appending), run the slow diffusion
        OFF the lock, then re-buffer the augmented trajectories.
        """
        with self._buf_lock:
            terminal = self._completed
            self._completed = []
        if not terminal:
            return
        try:
            if self._colocate:
                self._inner.sleep()
                self._diffusion.wake_up()
            augmented = self._diffuse(terminal)
        finally:
            if self._colocate:
                self._diffusion.sleep()
                self._inner.wake_up()
        with self._buf_lock:
            self._completed = augmented + self._completed

    def _diffuse(self, terminal: List[Sample]) -> List[Sample]:
        """Batch-generate M images per trajectory (conditioned on its final answer)
        and append each as a terminal diffusion Part on the trajectory's lineage.

        Mirrors ``ComposedRolloutEngine._generate_core``'s re-root: the trajectory's
        terminal answer lives on a deeply-slashed gen id, so it is re-rooted onto a
        fresh slash-free ``pe{k}`` root before forking the diffusion shell. One
        batched diffusion call, then split back per trajectory.
        """
        answers = [self._extract_terminal(tr) for tr in terminal]
        K = len(terminal)
        pe_input = Part.input(
            sample_ids=[f"pe{k}" for k in range(K)],
            primitives={"text": Texts(texts=answers)},
        )
        diff_shell = pe_input.fork(self._diff_M, sampling_params=self._diff_sp)
        diff_out = self._diffusion.generate(Sample(parts=[pe_input, diff_shell]))
        require(
            len(diff_out.parts[-1].sample_ids) == K * self._diff_M,
            f"AgenticImageRolloutEngine: diffusion child returned {len(diff_out.parts[-1].sample_ids)} samples; "
            f"expected K*M={K}*{self._diff_M}={K * self._diff_M}",
        )
        # One subtree per pe{k} root (M diffusion rows each); row order aligns with
        # ``terminal`` because pe{k} was built from terminal[k].
        diff_samples = diff_out.split()
        require(
            len(diff_samples) == K,
            f"AgenticImageRolloutEngine: diffusion split yielded {len(diff_samples)} groups; expected K={K}",
        )
        augmented: List[Sample] = []
        for tr, diff_s in zip(terminal, diff_samples):
            block = diff_s.parts[-1]
            augmented.append(
                tr.fork(self._diff_M, sampling_params=self._diff_sp).with_filled_frontier(
                    segment=block.segment,
                    primitives=dict(block.primitives),
                    primitive_metadata=dict(block.primitive_metadata),
                    conditions=dict(block.conditions),
                    media_preview=block.media_preview,
                    weight_version=block.weight_version,
                )
            )
        return augmented

    def _extract_terminal(self, tr: Sample) -> str:
        """The trajectory's final answer → the diffusion condition text."""
        gens = tr.gen_parts()
        term = ""
        if gens:
            primitive = gens[-1].primitives.get("text")
            if isinstance(primitive, Texts) and primitive.texts:
                term = primitive.texts[0]
        if not self._answer_marker:
            return _extract_answer(term)
        # Lazy import: the shared PE marker extractor lives in the models.pe
        # package whose __init__ eagerly pulls the PE model stack (diffusers /
        # transformers). Import it only when a marker is actually configured, so
        # the engine module stays importable without that stack for the default
        # (<answer>) path.
        from unirl.models.pe.instruction import postprocess_pe_texts

        root_prompt = ""
        if tr.parts:
            rp = tr.parts[0].primitives.get("text")
            if isinstance(rp, Texts) and rp.texts:
                root_prompt = rp.texts[0]
        cleaned, _stats = postprocess_pe_texts(
            [term],
            user_prompts=[root_prompt],
            samples_per_prompt=1,
            marker=self._answer_marker,
            max_chars=self._answer_max_chars,
        )
        return cleaned[0]

    # ------------------------------------------------------------------
    # Lifecycle — cover BOTH children (base agentic covers only the inner LLM).
    # Re-apply @distributed on sleep/wake_up/onload_weights so the Handle binds
    # the subclass attribute (see BaseRolloutEngine.sleep docstring).
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def shutdown(self) -> None:
        # Stop the collector before the children: it may still be mid-``generate`` on
        # the diffusion child, and a drain that raised past run_drain's finally can
        # leave it running.
        if self._batcher is not None:
            try:
                self._batcher.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning("AgenticImageRolloutEngine: diffusion batcher stop raised: %s", exc)
        for name, child in self._child_by_name.items():
            try:
                child.shutdown()
            except Exception as exc:  # noqa: BLE001
                logger.warning("AgenticImageRolloutEngine: child %r shutdown raised: %s", name, exc)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sleep(self) -> None:
        for child in self._child_by_name.values():
            child.sleep()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def wake_up(self) -> None:
        for child in self._child_by_name.values():
            child.wake_up()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def onload_weights(self, *, track_prefix: str = "") -> None:
        for child in self._children_for_track_prefix(track_prefix):
            child.onload_weights()

    @property
    def is_offloaded(self) -> bool:
        return all(child.is_offloaded for child in self._child_by_name.values())

    def health_check(self) -> bool:
        return all(child.health_check() for child in self._child_by_name.values())

    def get_memory_info(self) -> Dict[str, float]:
        out: Dict[str, float] = {"allocated_gb": 0.0, "cached_gb": 0.0}
        for child in self._child_by_name.values():
            info = child.get_memory_info() or {}
            out["allocated_gb"] += float(info.get("allocated_gb", 0.0))
            out["cached_gb"] += float(info.get("cached_gb", 0.0))
        return out

    def pause(self) -> None:
        for child in self._child_by_name.values():
            child.pause()

    def resume(self) -> None:
        for child in self._child_by_name.values():
            child.resume()

    # ------------------------------------------------------------------
    # Weight sync — track_prefix demux over {ar: inner LLM, diffusion: child}.
    # Mirrors ComposedRolloutEngine; the base agentic engine forwards every verb
    # to the inner only, which would send a diffusion-track update to the LLM.
    # ------------------------------------------------------------------

    def _demux_by_prefix(self, keys: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Split a dict by child-name key prefix, stripping the prefix."""
        result: Dict[str, Dict[str, Any]] = {}
        for child_name in self._child_by_name:
            prefix = f"{child_name}."
            subset = {k[len(prefix) :]: v for k, v in keys.items() if k.startswith(prefix)}
            if subset:
                result[child_name] = subset
        return result

    def _children_for_track_prefix(self, track_prefix: str) -> List[BaseSingleTurnRolloutEngine]:
        if not track_prefix:
            return list(self._child_by_name.values())
        child = self._child_by_name.get(track_prefix)
        if child is None:
            raise ValueError(
                f"AgenticImageRolloutEngine: unknown track_prefix {track_prefix!r}; "
                f"expected one of {sorted(self._child_by_name)}."
            )
        return [child]

    def _require_track_prefix(self, method: str, track_prefix: str) -> None:
        if not track_prefix:
            raise ValueError(
                f"AgenticImageRolloutEngine.{method} requires track_prefix so the update can be routed "
                f"to one child; expected one of {sorted(self._child_by_name)}."
            )

    def init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        track_prefix: str = "",
    ) -> None:
        self._require_track_prefix("init_weights_update_group", track_prefix)
        for child in self._children_for_track_prefix(track_prefix):
            child.init_weights_update_group(
                master_address=master_address,
                master_port=master_port,
                rank_offset=rank_offset,
                world_size=world_size,
                group_name=group_name,
                backend=backend,
            )

    def update_weights_from_distributed(
        self,
        *,
        names: List[str],
        dtypes: List[str],
        shapes: List[List[int]],
        group_name: str,
        target_modules: Optional[List[str]] = None,
        flush_cache: bool = True,
        track_prefix: str = "",
    ) -> None:
        self._require_track_prefix("update_weights_from_distributed", track_prefix)
        for child in self._children_for_track_prefix(track_prefix):
            child.update_weights_from_distributed(
                names=names,
                dtypes=dtypes,
                shapes=shapes,
                group_name=group_name,
                target_modules=target_modules,
                flush_cache=flush_cache,
            )

    def destroy_weights_update_group(self, *, group_name: str, track_prefix: str = "") -> None:
        self._require_track_prefix("destroy_weights_update_group", track_prefix)
        for child in self._children_for_track_prefix(track_prefix):
            child.destroy_weights_update_group(group_name=group_name)

    def update_weights_from_ipc(
        self,
        *,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
        use_shm: bool = False,
        track_prefix: str = "",
    ) -> None:
        self._require_track_prefix("update_weights_from_ipc", track_prefix)
        for child in self._children_for_track_prefix(track_prefix):
            child.update_weights_from_ipc(
                peft_config=peft_config,
                base_sync_done=base_sync_done,
                use_shm=use_shm,
            )

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
        track_prefix: str = "",
    ) -> None:
        self._require_track_prefix("update_weights_from_tensor", track_prefix)
        for child in self._children_for_track_prefix(track_prefix):
            child.update_weights_from_tensor(
                serialized_named_tensors=serialized_named_tensors,
                target_modules=target_modules,
                load_format=load_format,
                flush_cache=flush_cache,
            )

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, Any],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        demuxed = self._demux_by_prefix(lora_tensors)
        if not demuxed:
            raise ValueError(
                "AgenticImageRolloutEngine.set_lora_from_tensors requires child-prefixed tensor keys; "
                f"expected prefixes {sorted(self._child_by_name)}."
            )
        for child_name, child_tensors in demuxed.items():
            child = self._child_by_name[child_name]
            if child.is_offloaded:
                child.onload_weights()
            child.set_lora_from_tensors(adapter_name, child_tensors, peft_config=peft_config)


__all__ = ["AgenticImageRolloutEngine"]
