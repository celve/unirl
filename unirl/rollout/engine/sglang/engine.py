"""``sglang`` engine core — wiring + delegation only.

A thin core over the backend seam: it names no concrete model (the adapter,
picked from the registry by ``config.model_family``, owns the
``Sample`` → ``Sample`` conversion) and no concrete transport (the seam
owns the SRT runtime — server subprocess + HTTP, or the in-process Engine,
picked by ``config.backend``). Weight sync is a :class:`WeightSync` component
constructed over the seam; the offload lifecycle (the two staged flags) lives
directly on the engine. The frozen ``base.py`` surface is implemented as thin
forwards here — they must be real class attributes anyway (``Worker.call``
dispatches by name; ``@distributed`` binds the most-derived attribute) — which
also absorbs the surface quirks (``track_prefix``) so the component keeps clean
signatures.

One-shot construction: after ``__init__`` returns, the SRT server is spawned and
healthy and the engine is usable. ``generate`` / ``sleep`` / ``wake_up``
re-apply ``@distributed`` (the decorator is not inherited — see ``base.py``).
No environment mutation happens here — the spawn-scoped env the SRT
subprocesses need is quarantined in the backends' ``boot``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from unirl.config.require import require
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.rollout.engine.base import BaseSingleTurnRolloutEngine
from unirl.rollout.engine.sglang.adapters import PreparedInputs, get_adapter
from unirl.rollout.engine.sglang.backends import HTTPBackend, NativeBackend
from unirl.rollout.engine.sglang.config import SGLangEngineConfig, SGLangPorts
from unirl.rollout.engine.sglang.utils import resolve_sampling
from unirl.rollout.engine.sglang.weight_sync import WeightSync
from unirl.types.primitives import Texts
from unirl.types.sample import Sample, _part_with_field
from unirl.types.sampling import BaseSamplingParams

logger = logging.getLogger(__name__)


class SGLangRolloutEngine(BaseSingleTurnRolloutEngine):
    """LLM/VLM rollout engine backed by a SGLang SRT server (v2 layout)."""

    _component_name = "sglang"

    def __init__(
        self,
        config: SGLangEngineConfig,
        *,
        device: Optional[torch.device] = None,
        strategy: Any = None,
        rank: Optional[int] = None,
        model_config: Optional[Any] = None,
        ports: Optional[SGLangPorts] = None,
    ) -> None:
        require(
            isinstance(config, SGLangEngineConfig),
            f"SGLangRolloutEngine requires SGLangEngineConfig; got {type(config).__name__}",
        )
        # LLM engine carries its own model path on the config; the diffusion
        # engine takes it from model_config. Log if a caller supplied one so
        # the divergence is visible.
        if model_config is not None:
            logger.debug(
                "SGLangRolloutEngine: model_config provided but ignored — "
                "LLM engine uses config.pretrained_model_ckpt_path",
            )
        del strategy  # LLM rollout has no SDE strategy

        self.cfg = config
        self.rank = rank
        self._device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_offloaded = False
        self._weights_onloaded_for_sync = False

        engine_kwargs: Dict[str, Any] = dict(config.engine_kwargs or {})

        # Tokenizer (+ AutoProcessor for VLM) — the encoding I/O the engine
        # owns, injected into the adapter so its conversion methods stay pure.
        # The processor encodes multimodal prompts the SAME way the trainside
        # replay does (it expands the single image placeholder and emits
        # pixel_values / image_grid_thw), keeping rollout and replay
        # token-for-token aligned.
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_ckpt_path, trust_remote_code=True)
        self._tokenizer = tokenizer
        processor = None
        if config.image_token is not None:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(config.pretrained_model_ckpt_path, trust_remote_code=True)

        # Adapter (the only read of a model knob) — owns the conversion.
        self.adapter = get_adapter(config.model_family)(config, model_config, tokenizer=tokenizer, processor=processor)

        logger.info(
            "Initializing sglang engine (rank=%s, model_family=%s, model=%s, tp=%s)",
            rank,
            config.model_family,
            config.pretrained_model_ckpt_path,
            config.tp_size,
        )

        # Ports — engine-reserved on this node at the last moment before the
        # spawn (both backends: nccl_port de-syncs colocated engines). Tests
        # inject a fixed set.
        if ports is None:
            ports = SGLangPorts.reserve()

        # Backend (the seam) — booted from the config-spelled intent.
        intent = config.server_intent(ports=ports, extra=self.adapter.boot_kwargs())
        concurrency = int(engine_kwargs.get("concurrency", config.concurrency))
        if config.backend == "native":
            self._backend = NativeBackend.boot(intent, concurrency=concurrency)
        else:
            # The address peers reach this server at (the bind host is usually
            # the 0.0.0.0 wildcard). Node-identity discovery, not runtime I/O —
            # and HTTP-only: it exists to build the client base_url.
            bind_host = str(engine_kwargs.get("host") or config.host or "0.0.0.0")
            advertise_host = engine_kwargs.get("advertise_host")
            if not advertise_host:
                try:
                    import ray

                    advertise_host = ray.util.get_node_ip_address()
                except Exception:
                    advertise_host = bind_host if bind_host not in ("0.0.0.0", "") else "127.0.0.1"

            self._backend = HTTPBackend.boot(
                intent,
                advertise_host=str(advertise_host),
                concurrency=concurrency,
                health_timeout_s=float(engine_kwargs.get("health_timeout_s", 300.0)),
            )

        # Weight sync — owns all sync/LoRA state, over the live seam.
        self._weight_sync = WeightSync(
            self._backend,
            uses_lora=bool(engine_kwargs.get("enable_lora", False)),
        )

        # The backend owns its runtime concurrency. The engine only owns policy
        # provenance and delegates generation through the seam.
        self._weight_version = 0

    # ------------------------------------------------------------------ #
    # Generation — sync whole-Sample path, safe for concurrent callers
    # ------------------------------------------------------------------ #

    def _prepare_generation(self, sample: Sample) -> Any:
        require(
            int(sample.parts[-1].batch_size) > 0,
            "SGLangRolloutEngine.generate requires a non-empty Sample (gen batch_size > 0)",
        )
        sampling = resolve_sampling(self.cfg, sample)
        prepared = self.adapter.build_inputs(sample, sampling=sampling)
        active_adapter = self._weight_sync.active_adapter
        if active_adapter:
            for payload in prepared.wire:
                payload["lora_path"] = active_adapter
        return prepared

    def _finish_generation(self, sample: Sample, prepared: Any, raw: List[Any]) -> Sample:
        return self._stamp_weight_version(self.adapter.build_response(sample, prepared, raw))

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, sample: Sample) -> Sample:
        """Generate one whole Sample synchronously through the backend seam.

        Safe for concurrent callers (the agentic drain calls it from one
        thread per trajectory): prepare/finish are pure per-call, and the
        backend keeps concurrent wires in flight together.
        """
        prepared = self._prepare_generation(sample)
        raw = self._backend.generate(prepared.wire)
        return self._finish_generation(sample, prepared, raw)

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def continue_generation(
        self,
        sample: Sample,
        *,
        prefix: str,
        sampling_params: Optional[BaseSamplingParams] = None,
        stop: Optional[List[str]] = None,
    ) -> Sample:
        """Continue the current assistant message after an injected decoder prefix.

        This is deliberately a second physical SRT request, but it does not render
        another chat turn.  For each frontier row its wire input is the *exact*
        prompt ids stored by the first decode, followed by that decode's response
        ids (with terminal EOS removed), followed by ``prefix`` ids.  Only the
        newly sampled suffix becomes the new generated Part's segment/log-probs;
        therefore every stored old log-prob still belongs to a real policy action.

        The decoded primitive prepends ``prefix`` to the returned suffix because
        terminal reward extraction reads the final generated Part rather than its
        prompt condition.  ``</answer>`` is the default stop and stop trimming is
        forcibly disabled so the scorer sees a complete tagged span.
        """
        require(bool(sample.parts), "SGLang continuation requires a non-empty Sample")
        previous = sample.parts[-1]
        require(previous.is_gen, "SGLang continuation requires a generated frontier Part")
        require(bool(prefix), "SGLang continuation requires a non-empty decoder prefix")

        segment = previous.segment
        require(
            segment is not None
            and getattr(segment, "tokens", None) is not None
            and getattr(segment, "cu_seqlens", None) is not None,
            "SGLang continuation requires packed frontier response tokens",
        )
        prompt = (previous.conditions or {}).get("prompt")
        require(
            prompt is not None
            and getattr(prompt, "input_ids", None) is not None
            and getattr(prompt, "attention_mask", None) is not None,
            "SGLang continuation requires the frontier's exact prompt condition",
        )

        params = sampling_params or previous.sampling_params
        require(params is not None, "SGLang continuation requires sampling params")
        request = sample.fork(1, sampling_params=params)
        sampling = resolve_sampling(self.cfg, request)
        require(sampling.n == 1, f"SGLang continuation requires branch-1 generation; got n={sampling.n}")

        encoded_prefix = self._tokenizer.encode(prefix, add_special_tokens=False)
        if hasattr(encoded_prefix, "tolist"):
            encoded_prefix = encoded_prefix.tolist()
        prefix_ids = [int(token_id) for token_id in encoded_prefix]
        require(bool(prefix_ids), "SGLang continuation prefix encoded to zero tokens")

        prompt_ids = prompt.input_ids
        prompt_mask = prompt.attention_mask
        batch_size = int(previous.batch_size)
        require(
            int(prompt_ids.shape[0]) == batch_size and int(prompt_mask.shape[0]) == batch_size,
            "SGLang continuation prompt condition is not frontier-row aligned",
        )
        cu = [int(x) for x in segment.cu_seqlens.tolist()]
        require(
            len(cu) == batch_size + 1,
            "SGLang continuation packed response offsets are not frontier-row aligned",
        )

        eos = getattr(self._tokenizer, "eos_token_id", None)
        eos_ids = (
            {int(x) for x in eos} if isinstance(eos, (list, tuple, set)) else ({int(eos)} if eos is not None else set())
        )
        full_inputs: List[List[int]] = []
        for row in range(batch_size):
            real_prompt = prompt_ids[row][prompt_mask[row].to(dtype=torch.bool)].tolist()
            response = segment.tokens[cu[row] : cu[row + 1]].tolist()
            # An EOS was sampled to end the first request; retaining it before the
            # injected prefix would ask the model to continue *after* end-of-sequence.
            while response and int(response[-1]) in eos_ids:
                response.pop()
            full_inputs.append([int(x) for x in real_prompt] + [int(x) for x in response] + prefix_ids)

        block = dict(sampling.block)
        block["stop"] = list(stop) if stop is not None else ["</answer>"]
        block["no_stop_trim"] = True
        # ``request`` is branch-1, but spell this invariant on the wire too so a
        # future sampling resolver change cannot fan out a repair continuation.
        block["n"] = 1
        wire = [
            {
                "input_ids": ids,
                "sampling_params": dict(block),
                "return_logprob": sampling.return_logprob,
                "logprob_start_len": 0,
            }
            for ids in full_inputs
        ]
        active_adapter = self._weight_sync.active_adapter
        if active_adapter:
            for payload in wire:
                payload["lora_path"] = active_adapter
        prepared = PreparedInputs(
            wire=wire,
            prompt_token_ids=full_inputs,
            resolved_n=1,
        )

        raw = self._backend.generate(wire)
        continued = self._finish_generation(request, prepared, raw)
        frontier = continued.parts[-1]
        frontier_text = frontier.primitives.get("text")
        require(
            isinstance(frontier_text, Texts),
            "SGLang decoder-prefix continuation requires a text response",
        )
        visible = _part_with_field(
            frontier,
            "primitives",
            {**frontier.primitives, "text": Texts(texts=[prefix + text for text in frontier_text.texts])},
        )
        metadata = []
        prior_metadata = frontier.metadata or [{} for _ in range(frontier.batch_size)]
        for row in range(frontier.batch_size):
            item = dict(prior_metadata[row] or {}) if row < len(prior_metadata) else {}
            item.update(
                {
                    "answer_injected": True,
                    "format_repair": True,
                    "decoder_prefix": prefix,
                }
            )
            metadata.append(item)
        visible = _part_with_field(visible, "metadata", metadata)
        return continued.with_parts([*continued.parts[:-1], visible])

    # ── control plane — sync; reached via the raw Worker.call RPC ──────────
    def abort(self, ids: Optional[List[str]] = None) -> List[Sample]:
        """Abort in-flight generation (best-effort). Partials surface via the
        pending ``generate`` returns, so this returns ``[]``."""
        del ids
        self._backend.abort(abort_all=True)
        return []

    def pause(self) -> None:
        self._backend.pause()

    def resume(self) -> None:
        self._backend.resume()

    # ------------------------------------------------------------------ #
    # Lifecycle — the offload flags live here; decorators re-applied
    # (base.py footgun)
    # ------------------------------------------------------------------ #

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sleep(self, tags: Optional[List[str]] = None) -> None:
        """Release GPU memory (offload).

        Flushes the cache first; sglang's release only fully frees the KV
        pool when the scheduler has no pending references.

        ``tags`` selects which sglang SRT memory regions to release (e.g.
        ``["weights"]``). ``None`` releases everything. Called again while
        offloaded (post-sync re-offload), it releases the weights that
        ``onload_weights`` restored — or no-ops if they never were.
        """
        release_tags = None if tags is None or len(tags) == 0 else list(tags)
        if release_tags is None and self._is_offloaded:
            if not self._weights_onloaded_for_sync:
                return
            release_tags = ["weights"]
        if release_tags is None or "kv_cache" in release_tags:
            self._backend.flush_cache()
        self._backend.release_memory(tags=release_tags)
        self._is_offloaded = True
        self._weights_onloaded_for_sync = False
        # Releasing weights frees the SRT LoRA pool; the adapter must be
        # re-pushed (set_lora_from_tensors) before it can be referenced again.
        if release_tags is None or "weights" in release_tags:
            self._weight_sync.mark_weights_released()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def wake_up(self, tags: Optional[List[str]] = None) -> None:
        """Resume GPU memory.

        Can be called multiple times with different tag subsets for a staged
        resume — e.g. ``wake_up(tags=["weights"])`` to allow weight sync, then
        ``wake_up(tags=["kv_cache", "cuda_graph"])`` before generation.
        """
        full_wake = tags is None or len(tags) == 0
        resume_tags = None if full_wake else list(tags)
        if resume_tags is None:
            if not self._is_offloaded:
                return
            if self._weights_onloaded_for_sync:
                resume_tags = ["kv_cache", "cuda_graph"]
        self._backend.resume_memory(tags=resume_tags)
        if full_wake:
            self._is_offloaded = False
            self._weights_onloaded_for_sync = False
        elif "weights" in resume_tags:
            self._weights_onloaded_for_sync = True

    def onload_weights(self, *, track_prefix: str = "") -> None:
        """Resume only model weights so tensor/NCCL sync can update them."""
        del track_prefix
        if not self._is_offloaded:
            return
        if self._weights_onloaded_for_sync:
            return
        self._backend.resume_memory(tags=["weights"])
        self._weights_onloaded_for_sync = True

    @property
    def is_offloaded(self) -> bool:
        return self._is_offloaded

    def health_check(self) -> bool:
        if self._is_offloaded:
            return True
        return self._backend.ping()

    def shutdown(self) -> None:
        self._backend.shutdown()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Weight sync — frozen base.py surface; thin forwards to the component.
    # Un-decorated: reached per worker via the raw ``Worker.call`` RPC, not
    # through ``@distributed``. ``track_prefix`` is absorbed here.
    # ------------------------------------------------------------------ #

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
        track_prefix: str = "",
    ) -> None:
        """Update weights from serialized tensors via the seam.

        ``target_modules`` is intentionally NOT forwarded — the diffusion-side
        default ``["transformer"]`` doesn't match LLM module naming. Omitting
        the field lets the SRT server accept all incoming weights correctly.
        """
        del target_modules, track_prefix
        self._weight_sync.update_weights_from_tensor(
            serialized_named_tensors=serialized_named_tensors,
            load_format=load_format,
            flush_cache=flush_cache,
        )
        self._weight_version += 1  # weights changed → bump the version stamped onto gens

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
        del track_prefix
        self._weight_sync.init_weights_update_group(
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
        """Receive weights via NCCL broadcast from training actors.

        ``target_modules`` is intentionally NOT forwarded (see
        :meth:`update_weights_from_tensor` for rationale).
        """
        del target_modules, track_prefix
        self._weight_sync.update_weights_from_distributed(
            names=names,
            dtypes=dtypes,
            shapes=shapes,
            group_name=group_name,
            flush_cache=flush_cache,
        )
        self._weight_version += 1  # weights changed → bump the version stamped onto gens

    def destroy_weights_update_group(
        self,
        *,
        group_name: str,
        track_prefix: str = "",
    ) -> None:
        del track_prefix
        self._weight_sync.destroy_weights_update_group(group_name=group_name)

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        self._weight_sync.set_lora_from_tensors(adapter_name, lora_tensors, peft_config=peft_config)

    @property
    def lora_dirty(self) -> bool:
        """True when LoRA is in use but the adapter must be (re)pushed before generate."""
        return self._weight_sync.lora_dirty

    # ``update_weights_from_ipc`` is deliberately NOT defined — the base raises
    # NotImplementedError (SGLang has no bucketed-IPC receiver).


__all__ = ["SGLangRolloutEngine"]
