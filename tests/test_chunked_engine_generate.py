"""Unit tests for ``chunked_engine_generate`` and ``chunked_decode_latents``.

Covers both helpers in ``diffusionrl.samplers.engine``. Verifies:

  - fast-path: ``chunk_size`` ``None`` or ``>= n_prompts`` calls the
    underlying engine method exactly once.
  - chunked path: prompts/latents are sliced into mini-batches of
    ``chunk_size`` (last chunk may be smaller), the underlying engine
    method is called ``ceil(n / chunk_size)`` times, and per-chunk
    outputs are concat'd along dim-0 in input order.
  - determinism: chunked vs unchunked runs over the same input produce
    bit-identical outputs (latents / decoded tensors / rewards /
    forwarded prompt ids).
  - fail-fast: ``chunk_size`` ``0`` / negative / non-int / empty
    inputs all raise ``ValueError`` (or ``TypeError`` for non-tensor
    decode input).

Both helpers are engine-agnostic: we mock ``BaseRolloutEngine`` with a
deterministic stub so the tests run on CPU/MPS without any real model.
"""

from __future__ import annotations

from typing import List

import pytest
import torch

from diffusionrl.samplers.engine import (
    BaseRolloutEngine,
    chunked_decode_latents,
    chunked_engine_generate,
)
from diffusionrl.types.prompts import Prompts
from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.sample import RolloutSamples
from diffusionrl.types.sampling import SamplingParams


def _prompt_id_value(pid: str) -> float:
    """Deterministic float keyed on prompt id; used to label per-prompt latents."""
    return float(int.from_bytes(pid.encode("utf-8"), byteorder="big", signed=False) % 9973)


def _build_request(*, n_prompts: int) -> RolloutRequest:
    prompts = Prompts.from_unique_prompts(
        [f"prompt-{i}" for i in range(n_prompts)],
        prompt_ids=[f"pid-{i}" for i in range(n_prompts)],
    )
    sampling_params = SamplingParams(
        num_inference_steps=2,
        guidance_scale=1.0,
        height=8,
        width=8,
        num_frames=1,
        seed=0,
    )
    return RolloutRequest(prompts=prompts, sampling_params=sampling_params)


class _DeterministicEngine(BaseRolloutEngine):
    """Mock engine that emits per-prompt-id deterministic ``RolloutSamples``.

    Records the prompt-id list of every ``generate`` call and the per-chunk
    batch size of every ``decode_latents`` call so the tests can assert
    chunk boundaries and ordering.
    """

    def __init__(self) -> None:
        super().__init__(config=None)
        self._is_initialized = True
        self.calls: List[List[str]] = []
        self.decode_calls: List[int] = []

    def initialize(self, device: torch.device) -> None:  # pragma: no cover - unused
        self._is_initialized = True

    def update_weights(self, state_dict) -> None:  # pragma: no cover - unused
        pass

    def generate(self, request: RolloutRequest) -> RolloutSamples:
        prompt_ids = list(request.prompts.prompt_ids)
        self.calls.append(prompt_ids)
        latents = torch.tensor([[_prompt_id_value(pid)] for pid in prompt_ids], dtype=torch.float32)
        rewards = torch.tensor([_prompt_id_value(pid) / 10.0 for pid in prompt_ids], dtype=torch.float32)
        return RolloutSamples(
            latents=latents,
            timesteps=torch.zeros(2),
            prompts=request.prompts,
            rewards=rewards,
        )

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Pure-tensor stand-in for VAE decode: returns ``latents * 2 + 1``.

        Deterministic and embarrassingly parallel along dim 0, so chunked vs
        unchunked outputs are bit-identical when concatenated.
        """
        self.decode_calls.append(int(latents.shape[0]))
        return latents.to(dtype=torch.float32) * 2.0 + 1.0


def _run(engine: _DeterministicEngine, *, n_prompts: int, chunk_size):
    request = _build_request(n_prompts=n_prompts)
    return chunked_engine_generate(engine, request, chunk_size=chunk_size)


def test_fast_path_when_chunk_size_is_none() -> None:
    engine = _DeterministicEngine()
    out = _run(engine, n_prompts=4, chunk_size=None)
    assert len(engine.calls) == 1
    assert engine.calls[0] == [f"pid-{i}" for i in range(4)]
    assert out.batch_size == 4
    assert out.latents.shape == (4, 1)


def test_fast_path_when_chunk_size_ge_n_prompts() -> None:
    engine = _DeterministicEngine()
    out = _run(engine, n_prompts=4, chunk_size=4)
    assert len(engine.calls) == 1
    assert out.batch_size == 4

    engine = _DeterministicEngine()
    out = _run(engine, n_prompts=4, chunk_size=999)
    assert len(engine.calls) == 1
    assert out.batch_size == 4


def test_chunked_path_slices_at_chunk_boundary() -> None:
    engine = _DeterministicEngine()
    out = _run(engine, n_prompts=5, chunk_size=2)
    # ceil(5 / 2) = 3 chunks: 2, 2, 1
    assert [len(c) for c in engine.calls] == [2, 2, 1]
    assert engine.calls[0] == ["pid-0", "pid-1"]
    assert engine.calls[1] == ["pid-2", "pid-3"]
    assert engine.calls[2] == ["pid-4"]
    assert out.batch_size == 5


def test_chunked_path_preserves_prompt_order_and_values() -> None:
    n = 6
    request = _build_request(n_prompts=n)
    expected = torch.tensor([[_prompt_id_value(pid)] for pid in request.prompts.prompt_ids], dtype=torch.float32)

    engine_full = _DeterministicEngine()
    out_full = chunked_engine_generate(engine_full, request, chunk_size=None)

    engine_chunked = _DeterministicEngine()
    out_chunked = chunked_engine_generate(engine_chunked, request, chunk_size=1)

    assert len(engine_full.calls) == 1
    assert len(engine_chunked.calls) == n
    assert torch.equal(out_full.latents, expected)
    assert torch.equal(out_chunked.latents, out_full.latents)
    assert torch.equal(out_chunked.rewards, out_full.rewards)
    assert list(out_chunked.prompts.prompt_ids) == list(out_full.prompts.prompt_ids)


def test_chunk_size_zero_is_rejected() -> None:
    engine = _DeterministicEngine()
    with pytest.raises(ValueError, match="must be a positive int"):
        _run(engine, n_prompts=3, chunk_size=0)


def test_negative_chunk_size_is_rejected() -> None:
    engine = _DeterministicEngine()
    with pytest.raises(ValueError, match="must be a positive int"):
        _run(engine, n_prompts=3, chunk_size=-1)


def test_non_int_chunk_size_is_rejected() -> None:
    engine = _DeterministicEngine()
    with pytest.raises(ValueError, match="must be a positive int"):
        _run(engine, n_prompts=3, chunk_size=1.5)


def test_empty_prompts_is_rejected() -> None:
    engine = _DeterministicEngine()
    empty = Prompts.from_unique_prompts([])
    sampling_params = SamplingParams(
        num_inference_steps=2,
        guidance_scale=1.0,
        height=8,
        width=8,
        num_frames=1,
        seed=0,
    )
    request = RolloutRequest(prompts=empty, sampling_params=sampling_params)
    with pytest.raises(ValueError, match="non-empty request.prompts.prompts"):
        chunked_engine_generate(engine, request, chunk_size=2)


# ---------------------------------------------------------------------------
# chunked_decode_latents tests
# ---------------------------------------------------------------------------


def _make_decode_input(*, n: int) -> torch.Tensor:
    """Return a deterministic [n, 2, 3] latent tensor whose values encode the row index."""
    return torch.arange(n * 2 * 3, dtype=torch.float32).reshape(n, 2, 3)


def test_decode_fast_path_when_chunk_size_is_none() -> None:
    engine = _DeterministicEngine()
    latents = _make_decode_input(n=4)
    out = chunked_decode_latents(engine, latents, chunk_size=None)
    assert engine.decode_calls == [4]
    assert out.shape == (4, 2, 3)
    assert torch.equal(out, latents * 2.0 + 1.0)


def test_decode_fast_path_when_chunk_size_ge_batch() -> None:
    engine = _DeterministicEngine()
    latents = _make_decode_input(n=4)
    out = chunked_decode_latents(engine, latents, chunk_size=4)
    assert engine.decode_calls == [4]
    assert out.shape == (4, 2, 3)

    engine = _DeterministicEngine()
    out = chunked_decode_latents(engine, latents, chunk_size=999)
    assert engine.decode_calls == [4]
    assert out.shape == (4, 2, 3)


def test_decode_chunked_path_is_bit_identical_to_full_batch() -> None:
    n = 5
    latents = _make_decode_input(n=n)

    engine_full = _DeterministicEngine()
    out_full = chunked_decode_latents(engine_full, latents, chunk_size=None)

    # chunk_size=2 -> ceil(5/2)=3 chunks of sizes [2, 2, 1]
    engine_chunked = _DeterministicEngine()
    out_chunked = chunked_decode_latents(engine_chunked, latents, chunk_size=2)
    assert engine_chunked.decode_calls == [2, 2, 1]
    assert torch.equal(out_chunked, out_full)

    # chunk_size=1 -> n chunks of size 1
    engine_one = _DeterministicEngine()
    out_one = chunked_decode_latents(engine_one, latents, chunk_size=1)
    assert engine_one.decode_calls == [1] * n
    assert torch.equal(out_one, out_full)


def test_decode_chunk_size_zero_is_rejected() -> None:
    engine = _DeterministicEngine()
    with pytest.raises(ValueError, match="must be a positive int"):
        chunked_decode_latents(engine, _make_decode_input(n=3), chunk_size=0)


def test_decode_negative_chunk_size_is_rejected() -> None:
    engine = _DeterministicEngine()
    with pytest.raises(ValueError, match="must be a positive int"):
        chunked_decode_latents(engine, _make_decode_input(n=3), chunk_size=-1)


def test_decode_non_int_chunk_size_is_rejected() -> None:
    engine = _DeterministicEngine()
    with pytest.raises(ValueError, match="must be a positive int"):
        chunked_decode_latents(engine, _make_decode_input(n=3), chunk_size=1.5)


def test_decode_empty_latents_is_rejected() -> None:
    engine = _DeterministicEngine()
    empty = torch.empty(0, 2, 3, dtype=torch.float32)
    with pytest.raises(ValueError, match="non-empty latents"):
        chunked_decode_latents(engine, empty, chunk_size=2)


def test_decode_non_tensor_input_is_rejected() -> None:
    engine = _DeterministicEngine()
    with pytest.raises(TypeError, match="batched tensor"):
        chunked_decode_latents(engine, [1, 2, 3], chunk_size=2)  # type: ignore[arg-type]
