"""DiffusionTurnBatcher — coalesce concurrent in-loop image turns (LIN-577).

**Why this exists.** The agentic drain runs ``per_worker_concurrency`` trajectories
on their own threads. The AR inner absorbs that concurrency for free — its backend
"keeps the in-flight requests batching together"
(:class:`~unirl.rollout.engine.agentic.engine.AgenticRolloutEngine`). The diffusion
engines do **not**: both ``SGLangDiffusionRolloutEngine`` and
``TrainsideRolloutEngine`` guard ``generate`` with a mutex that *serializes*
concurrent callers ("the lock serializes concurrent generate callers"). So K drain
threads each taking an image turn would cost K sequential diffusion passes of batch
1, where the terminal (v1) shape costs one pass of batch K.

This is the in-process analogue of what the AR backend gives us: requests are
gathered for a short window — or until ``max_batch`` of them have parked — and
issued as **one** ``generate``, then split back to their callers.

**Bucketing is not optional.** FLUX.2-Klein serves t2i and ti2i off one checkpoint,
branching on :meth:`Sample.has_image_input` — a per-*Sample* decision. A batch is
therefore homogeneous by construction: text-only requests and image-conditioned
requests go to separate ``generate`` calls. Mixing them would silently take one
branch for both.

**Re-rooting is not optional either.** The ti2i adapter requires *exactly* one text
turn and one image turn, which a multi-turn trajectory never satisfies. Each request
is re-rooted onto a fresh single input Part carrying ``{"text", "image"}``; since
:meth:`Sample.turns` emits one turn per modality in ``PRIMITIVE_MODALITY_ORDER``,
that Part renders as exactly 1 text + 1 image. The caller writes the returned block
back onto the real trajectory lineage.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, List, Optional

from unirl.config.require import require
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Part, Sample

logger = logging.getLogger(__name__)

#: Sentinel posted by :meth:`DiffusionTurnBatcher.stop` to unblock the collector.
_STOP = object()


@dataclass
class _Request:
    """One trajectory's image turn, waiting on the collector thread."""

    text: str
    image: Optional[Images]
    event: threading.Event = dc_field(default_factory=threading.Event)
    part: Optional[Part] = None
    error: Optional[BaseException] = None


class DiffusionTurnBatcher:
    """Batch concurrent per-trajectory image turns into single ``generate`` calls."""

    def __init__(
        self,
        diffusion: Any,
        diffusion_sampling: Any,
        *,
        max_batch: int,
        window_s: float = 0.05,
        request_timeout_s: float = 1800.0,
    ) -> None:
        self._diffusion = diffusion
        self._diff_sp = diffusion_sampling
        self._max_batch = max(1, int(max_batch))
        self._window_s = max(0.0, float(window_s))
        self._request_timeout_s = float(request_timeout_s)
        self._q: "queue.Queue[Any]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._lifecycle = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle — one collector thread per drive
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the collector thread. Idempotent."""
        with self._lifecycle:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._loop, name="diffusion-turn-batcher", daemon=True)
            self._thread.start()

    def stop(self, timeout_s: float = 60.0) -> None:
        """Stop the collector and fail any still-pending request. Idempotent."""
        with self._lifecycle:
            thread = self._thread
            self._thread = None
        if thread is None:
            return
        self._q.put(_STOP)
        thread.join(timeout_s)
        if thread.is_alive():
            logger.warning("DiffusionTurnBatcher: collector did not stop within %.1fs", timeout_s)

    # ------------------------------------------------------------------
    # Caller side — one blocking call per image turn, on the drain thread
    # ------------------------------------------------------------------

    def generate(self, text: str, image: Optional[Images] = None) -> Part:
        """Render one image; block until this request's batch completes.

        ``image`` present selects the ti2i branch (it must hold exactly one image —
        one trajectory is one row in the drain). Returns the filled diffusion Part;
        the caller writes it back onto its own trajectory lineage.
        """
        require(bool(text and text.strip()), "DiffusionTurnBatcher.generate: empty prompt")
        if image is not None:
            require(
                len(image) == 1,
                f"DiffusionTurnBatcher.generate: expected exactly 1 source image per request, got {len(image)}",
            )
        self.start()  # a caller must never block on a collector that was never started
        req = _Request(text=text, image=image)
        self._q.put(req)
        if not req.event.wait(self._request_timeout_s):
            raise TimeoutError(f"DiffusionTurnBatcher: image turn exceeded {self._request_timeout_s:.0f}s")
        if req.error is not None:
            raise req.error
        require(req.part is not None, "DiffusionTurnBatcher: batch completed without a result Part")
        return req.part

    # ------------------------------------------------------------------
    # Collector
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        stopping = False
        try:
            while not stopping:
                item = self._q.get()
                if item is _STOP:
                    return
                batch: List[_Request] = [item]
                # Gather whatever else parks within the window, up to max_batch. The
                # window is small next to a denoise pass, so a lone trajectory pays
                # ~nothing while a full drain pays one pass instead of K.
                deadline = time.monotonic() + self._window_s
                while len(batch) < self._max_batch:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        nxt = self._q.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if nxt is _STOP:
                        # Serve the batch we already have, then exit on the next pass.
                        stopping = True
                        break
                    batch.append(nxt)
                self._run_batch(batch)
        except BaseException:  # noqa: BLE001 — the collector must never die silently
            logger.exception("DiffusionTurnBatcher: collector crashed; failing pending requests")
            raise
        finally:
            self._drain_failed()

    def _run_batch(self, batch: List[_Request]) -> None:
        """Split by modality (t2i vs ti2i cannot share a Sample) and serve each."""
        for bucket in ([r for r in batch if r.image is None], [r for r in batch if r.image is not None]):
            if bucket:
                self._serve(bucket)

    def _serve(self, reqs: List[_Request]) -> None:
        try:
            for req, part in zip(reqs, self._generate_bucket(reqs)):
                req.part = part
        except BaseException as exc:  # noqa: BLE001 — one bad batch must not hang its callers
            logger.warning("DiffusionTurnBatcher: batch of %d failed: %s", len(reqs), exc, exc_info=True)
            for req in reqs:
                req.error = exc
        finally:
            for req in reqs:
                req.event.set()

    def _generate_bucket(self, reqs: List[_Request]) -> List[Part]:
        """One ``generate`` for the whole bucket; returns one filled Part per request."""
        k = len(reqs)
        primitives: dict = {"text": Texts(texts=[r.text for r in reqs])}
        if reqs[0].image is not None:
            # One image per request, in request order — row-aligned to the prompts.
            primitives["image"] = Images.from_list([img for r in reqs for img in r.image.to_list()])
        root = Part.input(sample_ids=[f"draw{i}" for i in range(k)], primitives=primitives)
        shell = root.fork(1, sampling_params=self._diff_sp)
        out = self._diffusion.generate(Sample(parts=[root, shell]))
        require(
            len(out.parts[-1].sample_ids) == k,
            f"DiffusionTurnBatcher: diffusion returned {len(out.parts[-1].sample_ids)} rows; expected {k}",
        )
        samples = out.split()
        require(len(samples) == k, f"DiffusionTurnBatcher: split yielded {len(samples)} groups; expected {k}")
        return [s.parts[-1] for s in samples]

    def _drain_failed(self) -> None:
        """Release anyone still parked after the collector exits."""
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                return
            if item is _STOP:
                continue
            item.error = RuntimeError("DiffusionTurnBatcher: collector stopped before this image turn ran")
            item.event.set()


__all__ = ["DiffusionTurnBatcher"]
