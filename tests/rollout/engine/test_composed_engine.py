"""Unit tests for :class:`ComposedRolloutEngine`.

CPU-only. Uses fake child engines with distinct typed configs to exercise:

- Config validation (``__post_init__`` invariants).
- Inspect-based child building (``_build_child``).
- Generate flow (LLM → diffusion ordering, sleep/wake transitions, lineage).
- Sleep / wake routing by ``stage_ids``.
- Weight-sync routing by ``stage_ids`` (explicit single stage required).
- Aggregations (memory, health, shutdown best-effort).

No Ray, no GPU, no real subprocess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest
import torch

from diffusionrl.rollout.engine.base import BaseEngineConfig, BaseRolloutEngine
from diffusionrl.rollout.engine.composed import (
    ComposedRolloutEngine,
    ComposedRolloutEngineConfig,
)
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp, RolloutTrack
from diffusionrl.types.sampling import (
    ARSamplingParams,
    ComposedSamplingParams,
    DiffusionSamplingParams,
    get_ar_params,
)
from diffusionrl.types.segments.latent import LatentSegment
from diffusionrl.types.segments.text import TextSegment

# ---------------------------------------------------------------------------
# Fake child engines (must be module-level for hydra.utils.get_method)
# ---------------------------------------------------------------------------


@dataclass
class _FakeLLMConfig(BaseEngineConfig):
    model_path: str = "fake-llm"
    _target_: str = f"{__name__}._FakeLLMEngine"


@dataclass
class _FakeDiffConfig(BaseEngineConfig):
    model_family: str = "sd3"
    _target_: str = f"{__name__}._FakeDiffEngine"


class _FakeChildEngine(BaseRolloutEngine):
    """Records every method call so tests can assert routing behavior.

    Subclasses override ``__init__`` to declare a typed ``config`` parameter
    so :meth:`ComposedRolloutEngine._build_child` can recover the schema via
    ``inspect``.
    """

    name = "fake"

    def __init__(self, *, config: Any = None, **deps: Any) -> None:
        self.config = config
        self.deps = deps
        self.calls: List[Dict[str, Any]] = []
        self._is_offloaded = False
        # Per-test override: a callable taking the req and returning a RolloutResp.
        self.gen_factory = None

    # -- recorded methods --------------------------------------------------

    def shutdown(self) -> None:
        self.calls.append({"name": "shutdown"})

    def sleep(self, stage_ids: Optional[List[int]] = None) -> None:
        self.calls.append({"name": "sleep", "stage_ids": stage_ids})
        self._is_offloaded = True

    def wake_up(self, stage_ids: Optional[List[int]] = None) -> None:
        self.calls.append({"name": "wake_up", "stage_ids": stage_ids})
        self._is_offloaded = False

    def health_check(self) -> bool:
        return getattr(self, "_health", True)

    def get_memory_info(self) -> Dict[str, float]:
        return {"allocated_gb": 1.0, "cached_gb": 2.0}

    @property
    def is_offloaded(self) -> bool:
        return self._is_offloaded

    def generate(self, req: RolloutReq) -> RolloutResp:
        self.calls.append(
            {
                "name": "generate",
                "sample_ids": list(req.sample_ids),
                "primitives_text": list(req.primitives["text"].texts) if "text" in req.primitives else None,
                "sampling_params": req.sampling_params,
                "request_conditions": dict(req.request_conditions),
            }
        )
        if self.gen_factory is None:
            raise RuntimeError(f"{type(self).__name__}.generate: no gen_factory set")
        return self.gen_factory(req)

    # -- recorded weight-sync methods -------------------------------------

    def update_weights_from_ipc(self, **kwargs: Any) -> None:
        self.calls.append({"name": "update_weights_from_ipc", **kwargs})

    def init_weights_update_group(self, **kwargs: Any) -> None:
        self.calls.append({"name": "init_weights_update_group", **kwargs})

    def update_weights_from_distributed(self, **kwargs: Any) -> None:
        self.calls.append({"name": "update_weights_from_distributed", **kwargs})

    def destroy_weights_update_group(self, **kwargs: Any) -> None:
        self.calls.append({"name": "destroy_weights_update_group", **kwargs})

    def set_lora_from_tensors(self, adapter_name: str, lora_tensors: Any, **kwargs: Any) -> None:
        self.calls.append(
            {
                "name": "set_lora_from_tensors",
                "adapter_name": adapter_name,
                "n_tensors": len(lora_tensors) if lora_tensors is not None else 0,
                **kwargs,
            }
        )

    def update_weights_from_tensor(self, **kwargs: Any) -> None:
        self.calls.append({"name": "update_weights_from_tensor", **kwargs})


class _FakeLLMEngine(_FakeChildEngine):
    """LLM child fake; declares typed ``config: _FakeLLMConfig``."""

    name = "llm"

    def __init__(self, config: _FakeLLMConfig, **deps: Any) -> None:
        super().__init__(config=config, **deps)


class _FakeDiffEngine(_FakeChildEngine):
    """Diffusion child fake; declares typed ``config: _FakeDiffConfig``."""

    name = "diffusion"

    def __init__(self, config: _FakeDiffConfig, **deps: Any) -> None:
        super().__init__(config=config, **deps)


def _fake_composed_config(*, sleep_diffusion_on_start: bool = True) -> ComposedRolloutEngineConfig:
    return ComposedRolloutEngineConfig(
        ar=_FakeLLMConfig(model_path="fake-llm-path"),
        diffusion=_FakeDiffConfig(model_family="sd3"),
        sleep_diffusion_on_start=sleep_diffusion_on_start,
    )


# ---------------------------------------------------------------------------
# Fake response factories — produce single-track resps with given cardinality
# ---------------------------------------------------------------------------


def _make_llm_resp_factory(*, p_times_n: int):
    """Return a factory that produces a single-track LLM resp of P*N samples."""

    def factory(req: RolloutReq) -> RolloutResp:
        ar = get_ar_params(req.sampling_params)
        n = int(ar.samples_per_prompt) if ar is not None else 1
        p = int(req.batch_size)
        assert p * n == p_times_n, f"factory mismatch: P={p}, N={n}, expected {p_times_n}"
        sample_ids = [f"{sid}#{k}" for sid in req.sample_ids for k in range(n)]
        decoded_texts = [f"PE_for_{sid}_{k}" for sid in req.sample_ids for k in range(n)]
        seg = TextSegment(
            tokens=torch.zeros(0, dtype=torch.long),
            log_probs=torch.zeros(0, dtype=torch.float32),
            sample_indices=torch.arange(p_times_n, dtype=torch.long),
        )
        track = RolloutTrack(
            sample_ids=sample_ids,
            parent_ids=None,
            parent_track=None,
            segment=seg,
            decoded=Texts(texts=decoded_texts),
        )
        return RolloutResp(tracks={"text": track})

    return factory


def _make_diff_resp_factory(*, total: int):
    """Return a factory that produces a single-track diffusion resp of P*N*M samples."""

    def factory(req: RolloutReq) -> RolloutResp:
        assert int(req.batch_size) == total, (
            f"diff factory mismatch: req.batch_size={int(req.batch_size)} expected {total}"
        )
        sample_ids = list(req.sample_ids)
        seg = LatentSegment(
            latents=torch.zeros((total, 1, 4, 8, 8)),
            log_probs=torch.zeros((total, 1)),
            sigmas=torch.linspace(1.0, 0.0, 2),
            sample_indices=torch.arange(total, dtype=torch.long),
        )
        track = RolloutTrack(
            sample_ids=sample_ids,
            parent_ids=None,
            parent_track=None,
            segment=seg,
            decoded=None,
        )
        return RolloutResp(tracks={"image": track})

    return factory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_engine(*, sleep_diffusion_on_start: bool = True) -> ComposedRolloutEngine:
    cfg = _fake_composed_config(sleep_diffusion_on_start=sleep_diffusion_on_start)
    return ComposedRolloutEngine(
        cfg,
        device=torch.device("cpu"),
        strategy=None,
        rank=0,
        model_config=None,
    )


def _build_req(*, prompts: List[str], n: int, m: int) -> RolloutReq:
    return RolloutReq(
        sample_ids=[f"p{i}" for i in range(len(prompts))],
        group_ids=[f"p{i}" for i in range(len(prompts))],
        primitives={"text": Texts(texts=list(prompts))},
        request_conditions={},
        sampling_params=ComposedSamplingParams(
            diffusion=DiffusionSamplingParams(samples_per_prompt=m),
            ar=ARSamplingParams(samples_per_prompt=n),
        ),
    )


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_config_target_resolves_to_engine():
    cfg = _fake_composed_config()
    assert getattr(cfg, "_target_").endswith("ComposedRolloutEngine")


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


def test_ctor_accepts_all_four_runtime_kwargs():
    cfg = _fake_composed_config()
    engine = ComposedRolloutEngine(
        cfg,
        device=torch.device("cpu"),
        strategy="fake-strategy",
        rank=2,
        model_config={"key": "value"},
    )
    # Both children should have received their deps.
    assert engine._ar.deps["device"] == torch.device("cpu")
    assert engine._ar.deps["rank"] == 2
    assert engine._ar.deps["strategy"] is None  # AR gets strategy=None.
    assert engine._ar.deps["model_config"] == {"key": "value"}
    assert engine._diffusion.deps["device"] == torch.device("cpu")
    assert engine._diffusion.deps["strategy"] == "fake-strategy"
    assert engine._diffusion.deps["rank"] == 2
    assert engine._diffusion.deps["model_config"] == {"key": "value"}


def test_ctor_sleeps_diffusion_on_start_when_flag_set():
    engine = _build_engine(sleep_diffusion_on_start=True)
    assert engine._diffusion.is_offloaded is True
    sleep_calls = [c for c in engine._diffusion.calls if c["name"] == "sleep"]
    assert len(sleep_calls) == 1
    # AR not slept.
    assert not any(c["name"] == "sleep" for c in engine._ar.calls)


def test_ctor_does_not_sleep_diffusion_when_flag_unset():
    engine = _build_engine(sleep_diffusion_on_start=False)
    assert not any(c["name"] == "sleep" for c in engine._diffusion.calls)
    assert not any(c["name"] == "sleep" for c in engine._ar.calls)


def test_build_child_inspect_recovers_typed_config():
    engine = _build_engine()
    assert isinstance(engine._ar.config, _FakeLLMConfig)
    assert engine._ar.config.model_path == "fake-llm-path"
    assert isinstance(engine._diffusion.config, _FakeDiffConfig)
    assert engine._diffusion.config.model_family == "sd3"


def test_build_child_rejects_missing_target():
    cfg = _fake_composed_config()
    object.__setattr__(
        cfg,
        "ar",
        {"model_path": "x"},  # no _target_
    )
    with pytest.raises(Exception):
        ComposedRolloutEngine(cfg, device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# Generate tests
# ---------------------------------------------------------------------------


def test_generate_calls_llm_then_diffusion_in_order():
    P, N, M = 2, 3, 2
    engine = _build_engine()
    engine._ar.gen_factory = _make_llm_resp_factory(p_times_n=P * N)
    engine._diffusion.gen_factory = _make_diff_resp_factory(total=P * N * M)

    req = _build_req(prompts=["a", "b"], n=N, m=M)
    engine.generate(req)

    # LLM first, then diffusion.
    llm_gen_idx = [i for i, c in enumerate(engine._ar.calls) if c["name"] == "generate"]
    diff_gen_idx = [i for i, c in enumerate(engine._diffusion.calls) if c["name"] == "generate"]
    assert len(llm_gen_idx) == 1
    assert len(diff_gen_idx) == 1

    # LLM saw P prompts.
    llm_gen_call = engine._ar.calls[llm_gen_idx[0]]
    assert llm_gen_call["sample_ids"] == ["p0", "p1"]
    assert llm_gen_call["primitives_text"] == ["a", "b"]
    assert llm_gen_call["sampling_params"].samples_per_prompt == N

    # Diffusion saw P*N*M expanded prompts.
    diff_gen_call = engine._diffusion.calls[diff_gen_idx[0]]
    assert len(diff_gen_call["sample_ids"]) == P * N * M
    assert len(diff_gen_call["primitives_text"]) == P * N * M


def test_generate_returns_two_track_resp_with_correct_lineage():
    P, N, M = 2, 3, 2
    engine = _build_engine()
    engine._ar.gen_factory = _make_llm_resp_factory(p_times_n=P * N)
    engine._diffusion.gen_factory = _make_diff_resp_factory(total=P * N * M)

    req = _build_req(prompts=["a", "b"], n=N, m=M)
    resp = engine.generate(req)

    assert set(resp.tracks.keys()) == {"ar", "diffusion"}
    assert resp.tracks["ar"].parent_track is None
    assert resp.tracks["diffusion"].parent_track == "ar"
    assert len(resp.tracks["ar"].sample_ids) == P * N
    assert len(resp.tracks["diffusion"].sample_ids) == P * N * M
    # Lineage FK passes (no exception from __post_init__).
    # parent_ids of AR reference the request prompts.
    assert resp.tracks["ar"].parent_ids == ["p0", "p0", "p0", "p1", "p1", "p1"]
    # parent_ids of diffusion reference the AR sample_ids.
    assert all(p in resp.tracks["ar"].sample_ids for p in resp.tracks["diffusion"].parent_ids)


def test_generate_sleep_wake_transitions():
    P, N, M = 1, 2, 2
    engine = _build_engine(sleep_diffusion_on_start=False)  # clean slate
    engine._ar.gen_factory = _make_llm_resp_factory(p_times_n=P * N)
    engine._diffusion.gen_factory = _make_diff_resp_factory(total=P * N * M)

    engine.generate(_build_req(prompts=["a"], n=N, m=M))

    # Look at sleep/wake calls on each child relative to its generate call.
    llm_seq = [c["name"] for c in engine._ar.calls]
    diff_seq = [c["name"] for c in engine._diffusion.calls]

    # LLM: wake_up before generate, sleep after.
    assert llm_seq.index("wake_up") < llm_seq.index("generate") < llm_seq.index("sleep")
    # Diffusion: sleep first (before LLM stage), wake_up before generate.
    assert diff_seq.index("sleep") < diff_seq.index("wake_up") < diff_seq.index("generate")


def test_generate_uses_n_from_sampling_params_ar():
    P, N, M = 1, 4, 1
    engine = _build_engine()
    engine._ar.gen_factory = _make_llm_resp_factory(p_times_n=P * N)
    engine._diffusion.gen_factory = _make_diff_resp_factory(total=P * N * M)
    engine.generate(_build_req(prompts=["a"], n=N, m=M))
    llm_gen = next(c for c in engine._ar.calls if c["name"] == "generate")
    assert llm_gen["sampling_params"].samples_per_prompt == N


def test_generate_propagates_pe_texts_via_primitives_text():
    """PE texts must ride on diffusion sub-request's primitives['text'], NOT conditions."""
    P, N, M = 1, 2, 3
    engine = _build_engine()
    engine._ar.gen_factory = _make_llm_resp_factory(p_times_n=P * N)
    engine._diffusion.gen_factory = _make_diff_resp_factory(total=P * N * M)
    engine.generate(_build_req(prompts=["a"], n=N, m=M))
    diff_gen = next(c for c in engine._diffusion.calls if c["name"] == "generate")
    # PE texts should appear, each replicated M times consecutively (group-by-parent).
    expected_texts = ["PE_for_p0_0"] * M + ["PE_for_p0_1"] * M
    assert diff_gen["primitives_text"] == expected_texts
    # Caller-supplied request_conditions is empty in this test; the composed
    # engine propagates the dict by value (see propagation test below).
    assert diff_gen["request_conditions"] == {}


def test_generate_propagates_request_conditions_to_both_children():
    """``request_conditions`` (e.g. ``initial_latents`` for I2V/I2I/resume) must
    flow into both the LLM and diffusion sub-requests. Previously they were
    dropped with ``request_conditions={}``, which silently invalidated any
    caller-supplied seed conditions on the composed engine entry point.
    """
    from diffusionrl.types.conditions import ImageLatentCondition

    P, N, M = 1, 2, 1
    engine = _build_engine()
    engine._ar.gen_factory = _make_llm_resp_factory(p_times_n=P * N)
    engine._diffusion.gen_factory = _make_diff_resp_factory(total=P * N * M)

    seed_latents = torch.arange(P * N * M * 4, dtype=torch.float32).reshape(P * N * M, 1, 2, 2)
    seed_cond = ImageLatentCondition(latents=seed_latents)
    req = RolloutReq(
        sample_ids=[f"p{i}" for i in range(P)],
        group_ids=[f"p{i}" for i in range(P)],
        primitives={"text": Texts(texts=["a"])},
        request_conditions={"initial_latents": seed_cond},
        sampling_params=ComposedSamplingParams(
            diffusion=DiffusionSamplingParams(samples_per_prompt=M),
            ar=ARSamplingParams(samples_per_prompt=N),
        ),
    )
    engine.generate(req)

    llm_gen = next(c for c in engine._ar.calls if c["name"] == "generate")
    diff_gen = next(c for c in engine._diffusion.calls if c["name"] == "generate")
    assert "initial_latents" in llm_gen["request_conditions"], "LLM child must receive caller conditions"
    assert "initial_latents" in diff_gen["request_conditions"], "diffusion child must receive caller conditions"
    # Same condition value flows through (identity-preserving propagation).
    assert llm_gen["request_conditions"]["initial_latents"] is seed_cond
    assert diff_gen["request_conditions"]["initial_latents"] is seed_cond


def test_generate_lifts_llm_child_conditions_onto_track():
    """The LLM child's track.conditions (e.g. the chat-template-formatted
    prompt input_ids that SGLangLLMRolloutEngine packs as a
    TextTokenCondition) must reach the final llm_track so ARGRPO /
    Qwen3ARStage.replay can teacher-force at train time. Without this lift,
    train-side conditions['prompt'] is None and replay raises.
    """
    from diffusionrl.types.conditions import TextTokenCondition

    P, N, M = 2, 2, 1
    engine = _build_engine()

    def llm_factory(req: RolloutReq) -> RolloutResp:
        ar = get_ar_params(req.sampling_params)
        n = int(ar.samples_per_prompt) if ar is not None else 1
        total = int(req.batch_size) * n
        sample_ids = [f"{sid}#{k}" for sid in req.sample_ids for k in range(n)]
        decoded_texts = [f"PE_for_{sid}_{k}" for sid in req.sample_ids for k in range(n)]
        # Stub prompt input_ids: distinct values per sample so the assertion
        # confirms the right tensors flow through (not e.g. an accidental zero
        # tensor from the shell).
        input_ids = torch.tensor(
            [[10 + i, 11 + i, 12 + i] for i in range(total)],
            dtype=torch.long,
        )
        attention_mask = torch.ones((total, 3), dtype=torch.long)
        seg = TextSegment(
            tokens=torch.zeros(0, dtype=torch.long),
            log_probs=torch.zeros(0, dtype=torch.float32),
            sample_indices=torch.arange(total, dtype=torch.long),
        )
        track = RolloutTrack(
            sample_ids=sample_ids,
            parent_ids=None,
            parent_track=None,
            conditions={
                "prompt": TextTokenCondition(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            },
            segment=seg,
            decoded=Texts(texts=decoded_texts),
        )
        return RolloutResp(tracks={"text": track})

    engine._ar.gen_factory = llm_factory
    engine._diffusion.gen_factory = _make_diff_resp_factory(total=P * N * M)

    resp = engine.generate(_build_req(prompts=["a", "b"], n=N, m=M))
    llm_track = resp.tracks["ar"]
    assert "prompt" in llm_track.conditions
    prompt_cond = llm_track.conditions["prompt"]
    assert isinstance(prompt_cond, TextTokenCondition)
    # Tensor flowed through unchanged.
    assert prompt_cond.input_ids.shape == (P * N, 3)
    assert prompt_cond.input_ids[0].tolist() == [10, 11, 12]
    assert prompt_cond.input_ids[-1].tolist() == [10 + P * N - 1, 11 + P * N - 1, 12 + P * N - 1]


def test_generate_rejects_empty_req():
    engine = _build_engine()
    empty_req = RolloutReq(sample_ids=[], group_ids=[], primitives={})
    with pytest.raises(Exception):
        engine.generate(empty_req)


def test_generate_rejects_missing_text_primitive():
    engine = _build_engine()
    bad_req = RolloutReq(
        sample_ids=["p0"],
        group_ids=["p0"],
        primitives={},  # no 'text'
        sampling_params=ComposedSamplingParams(
            diffusion=DiffusionSamplingParams(samples_per_prompt=1),
            ar=ARSamplingParams(samples_per_prompt=1),
        ),
    )
    with pytest.raises(Exception):
        engine.generate(bad_req)


# ---------------------------------------------------------------------------
# Sleep / wake routing tests
# ---------------------------------------------------------------------------


def test_sleep_no_args_sleeps_both_children():
    engine = _build_engine(sleep_diffusion_on_start=False)
    engine.sleep()
    assert any(c["name"] == "sleep" for c in engine._ar.calls)
    assert any(c["name"] == "sleep" for c in engine._diffusion.calls)


def test_wake_up_no_args_wakes_both_children():
    engine = _build_engine(sleep_diffusion_on_start=False)
    engine.wake_up()
    assert any(c["name"] == "wake_up" for c in engine._ar.calls)
    assert any(c["name"] == "wake_up" for c in engine._diffusion.calls)


def test_onload_weights_without_track_prefix_wakes_both_children():
    engine = _build_engine(sleep_diffusion_on_start=False)
    engine.onload_weights()
    assert any(c["name"] == "wake_up" for c in engine._ar.calls)
    assert any(c["name"] == "wake_up" for c in engine._diffusion.calls)


def test_onload_weights_with_track_prefix_wakes_one_child():
    engine = _build_engine(sleep_diffusion_on_start=False)
    engine.onload_weights(track_prefix="ar")
    assert any(c["name"] == "wake_up" for c in engine._ar.calls)
    assert not any(c["name"] == "wake_up" for c in engine._diffusion.calls)


# ---------------------------------------------------------------------------
# Weight-sync routing tests
# ---------------------------------------------------------------------------

_UNSUPPORTED_COMPOSED_SYNC_METHODS = [
    ("update_weights_from_ipc", {}),
    (
        "init_weights_update_group",
        dict(master_address="x", master_port=0, rank_offset=0, world_size=1, group_name="g"),
    ),
    (
        "update_weights_from_distributed",
        dict(names=[], dtypes=[], shapes=[], group_name="g"),
    ),
    ("destroy_weights_update_group", dict(group_name="g")),
]


@pytest.mark.parametrize("method,kwargs", _UNSUPPORTED_COMPOSED_SYNC_METHODS)
def test_composed_rejects_ipc_and_nccl_sync(method, kwargs):
    engine = _build_engine()
    fn = getattr(engine, method)
    with pytest.raises(NotImplementedError):
        fn(**kwargs)


def test_update_weights_from_tensor_requires_track_prefix():
    engine = _build_engine()
    with pytest.raises(ValueError, match="track_prefix"):
        engine.update_weights_from_tensor(serialized_named_tensors=[])


def test_update_weights_from_tensor_routes_diffusion_track():
    engine = _build_engine()
    engine.update_weights_from_tensor(serialized_named_tensors=[], track_prefix="diffusion")
    assert any(c["name"] == "update_weights_from_tensor" for c in engine._diffusion.calls)
    assert not any(c["name"] == "update_weights_from_tensor" for c in engine._ar.calls)


def test_update_weights_from_tensor_routes_ar_track():
    engine = _build_engine()
    engine.update_weights_from_tensor(serialized_named_tensors=[], track_prefix="ar")
    assert any(c["name"] == "update_weights_from_tensor" for c in engine._ar.calls)
    assert not any(c["name"] == "update_weights_from_tensor" for c in engine._diffusion.calls)


def test_update_weights_from_tensor_unknown_track_raises():
    engine = _build_engine()
    with pytest.raises(ValueError, match="unknown"):
        engine.update_weights_from_tensor(serialized_named_tensors=[], track_prefix="unknown")


def test_set_lora_from_tensors_routes_prefixed_tensors_to_correct_child():
    engine = _build_engine()
    engine.set_lora_from_tensors("default", {"ar.a": torch.zeros(1)})
    llm_call = next(c for c in engine._ar.calls if c["name"] == "set_lora_from_tensors")
    assert llm_call["adapter_name"] == "default"
    assert llm_call["n_tensors"] == 1
    # Diffusion never received it.
    assert not any(c["name"] == "set_lora_from_tensors" for c in engine._diffusion.calls)


def test_set_lora_from_tensors_requires_child_prefixes():
    engine = _build_engine()
    with pytest.raises(ValueError, match="child-prefixed"):
        engine.set_lora_from_tensors("default", {"a": torch.zeros(1)})


# ---------------------------------------------------------------------------
# Structural parity tests
# ---------------------------------------------------------------------------


def test_class_has_required_lifecycle_methods():
    for attr in ["shutdown", "sleep", "wake_up", "health_check", "is_offloaded", "get_memory_info"]:
        assert hasattr(ComposedRolloutEngine, attr), attr


def test_class_has_all_weight_sync_methods():
    for attr in [
        "update_weights_from_ipc",
        "init_weights_update_group",
        "update_weights_from_distributed",
        "destroy_weights_update_group",
        "set_lora_from_tensors",
        "update_weights_from_tensor",
    ]:
        assert hasattr(ComposedRolloutEngine, attr), attr


# ---------------------------------------------------------------------------
# Aggregation tests
# ---------------------------------------------------------------------------


def test_shutdown_calls_each_child_even_if_one_raises():
    engine = _build_engine(sleep_diffusion_on_start=False)
    # Make LLM child raise on shutdown.
    original_shutdown = engine._ar.shutdown

    def boom() -> None:
        original_shutdown()
        raise RuntimeError("boom")

    engine._ar.shutdown = boom  # type: ignore[method-assign]
    engine.shutdown()  # must not propagate
    assert any(c["name"] == "shutdown" for c in engine._ar.calls)
    assert any(c["name"] == "shutdown" for c in engine._diffusion.calls)


def test_get_memory_info_sums_per_child():
    engine = _build_engine(sleep_diffusion_on_start=False)
    info = engine.get_memory_info()
    # Each fake reports {allocated_gb: 1.0, cached_gb: 2.0}; sum across both.
    assert info == {"allocated_gb": 2.0, "cached_gb": 4.0}


def test_health_check_all_must_be_healthy():
    engine = _build_engine()
    assert engine.health_check() is True
    engine._ar._health = False  # type: ignore[attr-defined]
    assert engine.health_check() is False


def test_is_offloaded_reflects_all_children():
    engine = _build_engine(sleep_diffusion_on_start=False)
    assert engine.is_offloaded is False
    engine.sleep()
    assert engine.is_offloaded is True
    engine.wake_up()
    assert engine.is_offloaded is False
