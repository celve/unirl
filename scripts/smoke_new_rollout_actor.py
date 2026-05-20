"""Smoke for the new ``RolloutReq``/``RolloutResp`` rollout-side stack.

Exercises the wiring added by the new-actor migration without requiring a
real engine / GPU. A stub :class:`BaseRolloutEngine` subclass returns a
synthetic ``RolloutResp`` shaped like vllm-omni's HI3 t2i output (image
segment with latents/sigmas/sde_logp, decoded pixel tensor, sample / group
ids); a stub reward service returns a constant reward; the smoke drives:

  - :func:`chunked_engine_generate_req` against the stub
  - :meth:`RolloutPlan.shard_grouped_req` to confirm group-aligned sharding
  - :meth:`RolloutResp.split` for per-group fan-out
  - :class:`NewRolloutPipelineMixin` end-to-end (generate_buffered →
    attach_reward → compute_advantages → run_rollout_pipeline)

PASS criteria: the fused ``run_rollout_pipeline`` returns one ``RolloutResp``
per group, each carrying populated ``rewards`` and ``advantages``, and the
final ``RolloutResp.concat`` round-trips through the legacy compat helper
(``resp_to_samples``) so a downstream caller could feed it to
``RolloutResponse.to_training_batch`` once a stub ``ForwardContext`` is
attached.

Run as::

    python scripts/smoke_new_rollout_actor.py

No external dependencies, no Ray, no model. The full vllm-omni / HI3
integration test still lives in ``smoke_vllm_omni_rollout_replay.py``
(replay-parity smoke against the real engine).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from diffusionrl.ray.mixins.new_rollout_pipeline import NewRolloutPipelineMixin
from diffusionrl.rollout.engine import chunked_engine_generate_req
from diffusionrl.rollout.engine.base import BaseRolloutEngine
from diffusionrl.rollout.plan import RolloutPlan
from diffusionrl.transfer.buffer import Buffer
from diffusionrl.types.primitives import Images, Texts
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp
from diffusionrl.types.segments import LatentSegment

# ---------------------------------------------------------------------------
# Stub engine + reward
# ---------------------------------------------------------------------------


class _StubEngine(BaseRolloutEngine):
    """In-process engine that returns a synthetic image-modality RolloutResp.

    Per-sample latent value seeded by sample_id so chunking determinism can be
    sanity-checked: ``latents[i, t, 0, 0, 0] = sid_value(sample_ids[i]) + t``.
    Each ``generate`` call records the sample-id list so the smoke can assert
    chunk boundaries.
    """

    def __init__(
        self,
        *,
        num_inference_steps: int = 4,
        height: int = 8,
        width: int = 8,
    ) -> None:
        self.num_inference_steps = int(num_inference_steps)
        self.height = int(height)
        self.width = int(width)
        self.calls: List[List[str]] = []

    def shutdown(self) -> None:
        pass

    def generate(self, req: RolloutReq) -> RolloutResp:
        self.calls.append(list(req.sample_ids))
        n = len(req.sample_ids)
        T = self.num_inference_steps
        # latents shape: [n, T+1, 1, h, w]; deterministic per sample id + step
        sid_vals = torch.tensor(
            [float(int.from_bytes(s.encode("utf-8"), "big") % 9973) for s in req.sample_ids],
            dtype=torch.float32,
        )
        latents = sid_vals.view(n, 1, 1, 1, 1).expand(n, T + 1, 1, self.height, self.width).clone()
        for t in range(T + 1):
            latents[:, t] = latents[:, t] + float(t)

        sigmas = torch.linspace(1.0, 0.0, T + 1)
        sde_logp = torch.zeros(n, T)
        sde_indices = torch.arange(T)
        sample_indices = torch.arange(n)

        seg = LatentSegment(
            latents=latents,
            sigmas=sigmas,
            sde_logp=sde_logp,
            sde_indices=sde_indices,
            sample_indices=sample_indices,
        )
        # Decoded pixels: [n, 3, h, w] in [0, 1]
        pixels = torch.ones(n, 3, self.height, self.width) * 0.5
        decoded = Images(pixels=pixels)
        return RolloutResp(
            conditions={},
            rollout_traces={"image": seg},
            decoded={"image": decoded},
            sample_ids=list(req.sample_ids),
            group_ids=list(req.group_ids),
        )


@dataclass
class _StubRewardResponse:
    rewards: List[float]
    component_rewards: Dict[str, List[float]]
    successes: List[bool]
    errors: List[Optional[str]]


@dataclass
class _StubRewardService:
    preferred_input_kind: str = "image"

    def compute_rewards(self, request: Any) -> _StubRewardResponse:
        n = len(getattr(request, "prompts", []) or [])
        return _StubRewardResponse(
            rewards=[1.0 + 0.1 * i for i in range(n)],
            component_rewards={"stub_component": [1.0] * n},
            successes=[True] * n,
            errors=[None] * n,
        )

    def offload(self) -> None:
        pass

    def onload(self) -> None:
        pass


@dataclass
class _StubRewardPipeline:
    """Reward pipeline with the same surface area attach_reward calls."""

    reward_service: _StubRewardService = field(default_factory=_StubRewardService)

    @property
    def preferred_input_kind(self) -> str:
        return self.reward_service.preferred_input_kind

    def score_and_attach(self, response: Any) -> Any:
        from diffusionrl.reward.pipeline import (
            _build_request_for_samples,
            _read_reward_payload,
        )

        if _read_reward_payload(response.samples) is not None:
            raise RuntimeError("Stub rejects pre-computed rewards.")
        prompts = response.request.prompts
        request = _build_request_for_samples(
            reward_input_kind=self.preferred_input_kind,
            samples_per_prompt=int(response.request.sampling_params.num_samples_per_prompt),
            sampler_outputs=[response.samples],
            prompts=prompts.prompts,
            prompt_ids=prompts.prompt_ids,
            sample_ids=prompts.sample_ids,
            group_ids=prompts.group_ids,
            prompt_metadata=prompts.prompt_metadata,
        )
        rr = self.reward_service.compute_rewards(request)
        response.samples.rewards = torch.tensor(rr.rewards, dtype=torch.float32)
        response.samples.component_rewards = {
            str(k): torch.tensor(list(v or []), dtype=torch.float32) for k, v in (rr.component_rewards or {}).items()
        }
        return response


@dataclass
class _StubAlgorithm:
    def compute_advantages(self, *, rewards: torch.Tensor, group_ids: List[str]) -> torch.Tensor:
        # Simple z-score per group; matches GRPO advantage shape (per-sample scalar).
        result = torch.zeros_like(rewards)
        seen: Dict[str, List[int]] = {}
        for i, g in enumerate(group_ids):
            seen.setdefault(g, []).append(i)
        for indices in seen.values():
            sub = rewards[indices]
            mean = sub.mean()
            std = sub.std() if sub.numel() > 1 else torch.tensor(1.0)
            result[indices] = (sub - mean) / (std + 1e-8)
        return result


class _StubActor(NewRolloutPipelineMixin, Buffer):
    """In-process stand-in for ``NewRolloutActor`` to exercise the mixin
    without needing Ray, Hydra, or a real engine.
    """

    def __init__(self, engine: _StubEngine, rollout_plan: RolloutPlan) -> None:
        super().__init__()
        self.engine = engine
        self._rollout_plan = rollout_plan
        self.algorithm = _StubAlgorithm()
        self._reward_pipeline: Optional[_StubRewardPipeline] = None

    def _ensure_reward_pipeline(self) -> _StubRewardPipeline:
        if self._reward_pipeline is None:
            self._reward_pipeline = _StubRewardPipeline()
        return self._reward_pipeline

    def generate(self, req: RolloutReq) -> RolloutResp:
        return chunked_engine_generate_req(
            self.engine,
            req,
            chunk_size=self._rollout_plan.forward_batch_size,
        )


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def _build_req(*, prompts_per_group: int, num_groups: int) -> RolloutReq:
    sample_ids: List[str] = []
    group_ids: List[str] = []
    texts: List[str] = []
    for g in range(num_groups):
        for k in range(prompts_per_group):
            sid = f"g{g}_s{k}"
            sample_ids.append(sid)
            group_ids.append(f"g{g}")
            texts.append(f"prompt for group {g} sample {k}")
    return RolloutReq(
        sample_ids=sample_ids,
        group_ids=group_ids,
        primitives={"text": Texts(texts=texts)},
        stage_params={
            "diffusion": {
                "height": 8,
                "width": 8,
                "num_inference_steps": 4,
                "guidance_scale": 1.0,
                "eta": 1.0,
                "seed": 0,
            }
        },
    )


def phase_a_chunked_generate() -> None:
    print("[smoke] Phase A: chunked_engine_generate_req")
    engine = _StubEngine(num_inference_steps=4, height=8, width=8)
    req = _build_req(prompts_per_group=2, num_groups=3)  # 6 samples

    # Fast path: no chunking
    resp_full = chunked_engine_generate_req(engine, req, chunk_size=None)
    assert resp_full.batch_size == 6, resp_full.batch_size
    assert len(engine.calls) == 1
    assert engine.calls[0] == req.sample_ids

    # Chunked: chunk_size=2 → 3 calls of [2, 2, 2]
    engine = _StubEngine(num_inference_steps=4, height=8, width=8)
    resp_chunked = chunked_engine_generate_req(engine, req, chunk_size=2)
    assert resp_chunked.batch_size == 6
    assert [len(c) for c in engine.calls] == [2, 2, 2]
    # Order preserved
    flat = [s for chunk in engine.calls for s in chunk]
    assert flat == req.sample_ids

    # Concat preserves sample order
    assert list(resp_chunked.sample_ids) == req.sample_ids
    print("[smoke] Phase A OK")


def phase_b_shard_grouped_req() -> None:
    print("[smoke] Phase B: RolloutPlan.shard_grouped_req")
    plan = RolloutPlan(forward_batch_size=4)
    req = _build_req(prompts_per_group=4, num_groups=3)  # 12 samples, 3 groups
    shards = plan.shard_grouped_req(req, num_actors=2, samples_per_prompt=4)
    assert len(shards) == 2, len(shards)
    # 3 groups across 2 actors → [2 groups, 1 group] = [8 samples, 4 samples]
    sizes = [s.batch_size if s is not None else 0 for s in shards]
    assert sizes == [8, 4], sizes
    # All groups whole inside their shard (no group split across actors)
    for shard in shards:
        if shard is None:
            continue
        gids = list(shard.group_ids)
        # Each unique group should fully appear in this shard
        seen: Dict[str, int] = {}
        for g in gids:
            seen[g] = seen.get(g, 0) + 1
        assert all(c == 4 for c in seen.values()), seen
    print("[smoke] Phase B OK")


def phase_c_resp_split() -> None:
    print("[smoke] Phase C: RolloutResp.split")
    engine = _StubEngine()
    req = _build_req(prompts_per_group=4, num_groups=3)  # 12 samples, 3 groups
    resp = engine.generate(req)
    shards = resp.split()
    assert len(shards) == 3, len(shards)
    for shard in shards:
        # All sample ids in shard share the same group id
        gids = set(shard.group_ids)
        assert len(gids) == 1, gids
        assert shard.batch_size == 4
        # Latents shape preserved per shard
        seg = shard.rollout_traces["image"]
        assert seg.latents.shape[0] == 4
    print("[smoke] Phase C OK")


def phase_d_run_rollout_pipeline() -> None:
    print("[smoke] Phase D: NewRolloutPipelineMixin.run_rollout_pipeline (fused)")
    engine = _StubEngine(num_inference_steps=4, height=8, width=8)
    plan = RolloutPlan(forward_batch_size=None)
    actor = _StubActor(engine=engine, rollout_plan=plan)

    req = _build_req(prompts_per_group=4, num_groups=2)  # 8 samples, 2 groups
    responses = actor.run_rollout_pipeline(req)

    assert len(responses) == 2, len(responses)
    for resp in responses:
        assert resp.batch_size == 4
        # rewards / advantages populated
        assert resp.rewards is not None and resp.rewards.shape == (4,)
        assert resp.advantages is not None and resp.advantages.shape == (4,)
        # component_rewards populated by the stub
        assert resp.component_rewards is not None
        assert "stub_component" in resp.component_rewards
        # group_ids consistent
        assert len(set(resp.group_ids)) == 1
        # reward_compute_s stamped (per-actor sum, so same across shards)
    actor_total = responses[0].reward_compute_s
    assert all(abs(r.reward_compute_s - actor_total) < 1e-9 for r in responses)

    # Cross-shard advantage normalization: per-group mean ≈ 0 (z-score)
    for resp in responses:
        assert abs(float(resp.advantages.mean().item())) < 1e-5

    # Ensure handle state cleaned up
    assert getattr(actor, "_handle_state", {}) == {}
    print("[smoke] Phase D OK")


def phase_e_resp_concat_and_compat() -> None:
    print("[smoke] Phase E: RolloutResp.concat round-trip via resp_to_samples")
    from diffusionrl.rollout.engine.types_compat import resp_to_samples
    from diffusionrl.types.prompts import Prompts
    from diffusionrl.types.request import RolloutRequest
    from diffusionrl.types.sampling import SamplingParams

    engine = _StubEngine()
    plan = RolloutPlan(forward_batch_size=None)
    actor = _StubActor(engine=engine, rollout_plan=plan)
    req = _build_req(prompts_per_group=2, num_groups=2)  # 4 samples
    responses = actor.run_rollout_pipeline(req)
    combined = RolloutResp.concat(responses)
    assert combined.batch_size == 4
    assert combined.rewards is not None and combined.rewards.shape == (4,)
    assert combined.advantages is not None and combined.advantages.shape == (4,)

    # Bridge to legacy samples (same path NewRolloutPipeline.convert_training_data uses).
    sids = list(combined.sample_ids)
    legacy_request = RolloutRequest(
        prompts=Prompts(
            prompts=[f"text:{s}" for s in sids],
            prompt_ids=list(sids),
            sample_ids=list(sids),
            group_ids=list(combined.group_ids),
            noise_group_ids=list(sids),
            prompt_metadata=[{} for _ in sids],
        ),
        sampling_params=SamplingParams(num_samples_per_prompt=2),
    )
    legacy_samples = resp_to_samples(combined, request=legacy_request)
    assert legacy_samples.latents.shape[0] == 4
    assert legacy_samples.decoded_images is not None
    assert len(legacy_samples.decoded_images) == 4
    print("[smoke] Phase E OK")


def main() -> int:
    phase_a_chunked_generate()
    phase_b_shard_grouped_req()
    phase_c_resp_split()
    phase_d_run_rollout_pipeline()
    phase_e_resp_concat_and_compat()
    print("[smoke] ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
