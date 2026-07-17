from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from unirl.trainer.sao_async import AsyncSAOTrainer, _trajectory_versions, _TrajectoryBuffer
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams
from unirl.types.segments import TextSegment


def _trajectory(root: str, versions: list[int]) -> Sample:
    parts = [Part.input([root])]
    parent = root
    for index, version in enumerate(versions):
        sample_id = f"{parent}/0"
        parts.append(
            Part(
                sample_ids=[sample_id],
                segment=TextSegment.pack(
                    tokens=[torch.tensor([index + 1])],
                    log_probs=[torch.tensor([-0.1])],
                    loss_mask=[torch.ones(1)],
                ),
                sampling_params=ARSamplingParams(samples_per_prompt=1),
                weight_version=version,
            )
        )
        parent = sample_id
    return Sample(parts=parts)


def _root(trajectory: Sample) -> str:
    return trajectory.parts[0].sample_ids[0]


def test_fifo_buffer_uses_oldest_turn_version_for_eviction() -> None:
    buffer = _TrajectoryBuffer()
    buffer.put(_trajectory("first", [2, 5]), completion_version=5, completion_id=0)
    buffer.put(_trajectory("second", [4]), completion_version=4, completion_id=1)

    # current=6, lag=2 evicts the first trajectory because its oldest turn is
    # version 2, even though it completed under version 5.
    picked = buffer.drain_fifo(1, current_version=6, max_oldest_version_lag=2)
    assert picked is not None
    assert [_root(item) for item in picked] == ["second"]
    assert buffer.evicted == 1


def test_fifo_waits_for_full_batch_and_preserves_completion_order() -> None:
    buffer = _TrajectoryBuffer()
    buffer.put(_trajectory("a", [0]), completion_version=0, completion_id=8)
    assert buffer.drain_fifo(2, current_version=0) is None
    buffer.put(_trajectory("b", [0]), completion_version=0, completion_id=2)

    picked = buffer.drain_fifo(2, current_version=0)
    assert picked is not None
    assert [_root(item) for item in picked] == ["a", "b"]


def test_trajectory_versions_fall_back_only_without_turn_stamps() -> None:
    assert _trajectory_versions(_trajectory("a", [7, 3, 9]), fallback=100) == (3, 9)
    assert _trajectory_versions(Sample.request(Part.input(["empty"])), fallback=11) == (11, 11)


def test_final_poll_reaps_completion_before_a_drained_drive_is_reset() -> None:
    trajectory = _trajectory("late", [0])

    class _Rollout:
        polls = 0

        def poll(self):
            self.polls += 1
            return [[]] if self.polls == 1 else [[trajectory]]

        @staticmethod
        def drained():
            return [True]

    trainer = AsyncSAOTrainer.__new__(AsyncSAOTrainer)
    trainer.rollout = _Rollout()
    trainer._buffer = _TrajectoryBuffer()
    trainer._weight_version = 0
    trainer._max_oldest_version_lag = None
    trainer._gen_id = 0
    trainer._invalid_trajectories = 0
    trainer._pending_carried = []
    trainer._submit_drive = lambda *_args, **_kwargs: pytest.fail(
        "the final completion must be consumed before a refill"
    )

    assert trainer._next_n(1, rollout_id=0) == [trajectory]


def _valid_sampling() -> tuple[dict, dict, dict]:
    actor = {"algorithm": {"sampling_temperature": 1.0}}
    rollout = {
        "config": {
            "episode_sampling": {
                "samples_per_prompt": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 0,
            }
        }
    }
    sampling = {
        "samples_per_prompt": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
    }
    return actor, rollout, sampling


def test_sampling_contract_accepts_exact_single_rollout_fidelity() -> None:
    actor, rollout, sampling = _valid_sampling()
    AsyncSAOTrainer._validate_sampling(object(), actor_cfg=actor, rollout_cfg=rollout, sampling_cfg=sampling)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("sampling", "samples_per_prompt"), 2, "exactly one"),
        (("sampling", "top_p"), 0.95, "top_p"),
        (("rollout", "top_k"), 20, "top_k"),
        (("actor", "temperature"), 0.8, "temperatures must match"),
    ],
)
def test_sampling_contract_fails_loudly(path: tuple[str, str], value: float, message: str) -> None:
    actor, rollout, sampling = _valid_sampling()
    owner, field = path
    if owner == "sampling":
        sampling[field] = value
    elif owner == "rollout":
        rollout["config"]["episode_sampling"][field] = value
    else:
        actor["algorithm"]["sampling_temperature"] = value

    with pytest.raises(ValueError, match=message):
        AsyncSAOTrainer._validate_sampling(object(), actor_cfg=actor, rollout_cfg=rollout, sampling_cfg=sampling)


class _SavingBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str]] = []

    def save(self, path: str, *, step: int, mode: str) -> None:
        self.calls.append((path, step, mode))


class _LoadingBackend:
    def __init__(self, step: int) -> None:
        self.step = int(step)
        self.paths: list[str] = []

    def load(self, path: str) -> int:
        self.paths.append(path)
        return self.step


class _WeightSync:
    def __init__(self) -> None:
        self.calls = 0

    def sync(self) -> None:
        self.calls += 1


def test_checkpoint_envelope_separates_actor_and_critic(tmp_path: Path) -> None:
    trainer = AsyncSAOTrainer.__new__(AsyncSAOTrainer)
    trainer.actor_backend = _SavingBackend()
    trainer.critic_backend = _SavingBackend()
    trainer.wandb_logger = SimpleNamespace(run_id="run", optimizer_step=17)
    trainer._submitted_prompt_batches = 9
    trainer._weight_version = 4

    trainer._save_sao_checkpoint(
        rollout_id=1,
        num_rollouts=2,
        save_interval=1,
        save_dir=str(tmp_path),
        save_mode="full",
    )

    root = tmp_path / "checkpoint-2"
    assert trainer.actor_backend.calls == [(str(root / "actor"), 2, "full")]
    assert trainer.critic_backend.calls == [(str(root / "critic"), 2, "full")]
    state = json.loads((root / "trainer_state.json").read_text())
    assert state == {
        "wandb_run_id": "run",
        "optimizer_step": 17,
        "submitted_prompt_batches": 9,
        "weight_version": 4,
    }


def test_checkpoint_resume_starts_fresh_rollout_version_epoch(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    root = tmp_path / "checkpoint-40"
    root.mkdir()
    (root / "trainer_state.json").write_text(
        json.dumps(
            {
                "submitted_prompt_batches": 7,
                "weight_version": 37,
            }
        )
    )

    trainer = AsyncSAOTrainer.__new__(AsyncSAOTrainer)
    trainer.actor_backend = _LoadingBackend(step=40)
    trainer.critic_backend = _LoadingBackend(step=40)
    trainer._submitted_prompt_batches = 0
    trainer._weight_version = 99
    trainer._resume_state = {}

    start = trainer._load_sao_checkpoint(str(root), num_rollouts=100)

    assert start == 40
    assert trainer._submitted_prompt_batches == 7
    assert trainer._weight_version == 0
    assert "intentionally not restored" in caplog.text

    trainer.weight_sync = _WeightSync()
    trainer._sync_resumed_rollout(resumed=True)
    assert trainer.weight_sync.calls == 1
    assert trainer._weight_version == 1

    # A trajectory emitted by the freshly synced engine must not be compared
    # against checkpoint version 37 and evicted as historically stale.
    buffer = _TrajectoryBuffer()
    buffer.put(_trajectory("fresh", [1]), completion_version=1, completion_id=0)
    picked = buffer.drain_fifo(1, current_version=trainer._weight_version, max_oldest_version_lag=2)
    assert picked is not None
    assert [_root(item) for item in picked] == ["fresh"]
    assert buffer.evicted == 0
