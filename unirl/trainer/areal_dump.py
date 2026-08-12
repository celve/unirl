"""Reviewable trajectory artifacts for AReaL deep-research runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from unirl.types.primitives import Texts
from unirl.types.sample import Sample

_HARNESS_FIELDS = (
    "protocol",
    "termination_reason",
    "prediction",
    "answer_tag_found",
    "policy_call_count",
    "prompt_tokens_at_rescue",
    "rescue_prompt_tokens",
    "tool_calls",
    "search_calls",
    "visit_calls",
    "retry_count",
    "elapsed_seconds",
    "controller_error_type",
    "controller_error_message",
)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _text_hash(texts: Sequence[str]) -> str:
    payload = json.dumps(list(texts), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _trajectory_row(
    trajectory: Sample,
    *,
    rollout_id: int,
    group_index: int,
    completion_index: int,
    reward: float,
) -> dict[str, Any]:
    if not trajectory.parts or trajectory.parts[0].batch_size != 1:
        raise ValueError("trajectory dump requires one rooted trajectory")
    root = trajectory.parts[0]
    root_metadata = root.metadata[0] if root.metadata else {}
    sibling_index = (root_metadata or {}).get("sibling_index", completion_index)
    ar_control = dict(root.control.get("ar") or {})
    turns = []
    assistant_texts = []
    for turn_index, part in enumerate(trajectory.parts):
        primitive = part.primitives.get("text")
        if primitive is None:
            continue
        if not isinstance(primitive, Texts) or len(primitive.texts) != 1:
            raise ValueError("trajectory dump requires one text row per turn")
        text = primitive.texts[0] or ""
        role = part.resolved_role()
        turns.append(
            {
                "turn": turn_index,
                "role": role,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "generated": bool(part.is_gen),
                "output_version": part.output_version,
            }
        )
        if role == "assistant":
            assistant_texts.append(text)

    last_metadata = trajectory.parts[-1].metadata[0] if trajectory.parts[-1].metadata else {}
    harness = last_metadata.get("harness") if isinstance(last_metadata, Mapping) else None
    safe_harness = {
        key: _json_value(harness.get(key)) for key in _HARNESS_FIELDS if isinstance(harness, Mapping) and key in harness
    }
    transcript_texts = [f"{turn['role']}\n{turn['text']}" for turn in turns]
    return {
        "schema_version": 1,
        "rollout_id": int(rollout_id),
        "group_index": int(group_index),
        "sibling_index": int(sibling_index),
        "completion_index": int(completion_index),
        "root_id": root.sample_ids[0],
        "leaf_id": trajectory.parts[-1].sample_ids[0],
        "sampling_seed_base": _json_value(ar_control.get("sampling_seed_base")),
        "reward": reward if math.isfinite(reward) else None,
        "harness_status": trajectory.parts[-1].harness_status,
        "ground_truth": _json_value((root_metadata or {}).get("answer")),
        "source_row": _json_value((root_metadata or {}).get("source_row")),
        "harness": safe_harness,
        "assistant_sha256": _text_hash(assistant_texts),
        "trajectory_sha256": _text_hash(transcript_texts),
        "turns": turns,
    }


class ARealTrajectoryDumper:
    """Persist every scored trajectory and a compact sibling-diversity summary."""

    def __init__(self, directory: str) -> None:
        if not str(directory or "").strip():
            raise ValueError("AReaL trajectory_dump_dir must be non-empty")
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        probe = self.directory / f".write-probe-{os.getpid()}"
        with open(probe, "x", encoding="utf-8") as handle:
            handle.write("ok\n")
            handle.flush()
            os.fsync(handle.fileno())
        probe.unlink()

    def dump(self, groups: Sequence[Sequence[Sample]], rewards: torch.Tensor, *, rollout_id: int) -> dict[str, Any]:
        """Write one atomic JSONL plus summary before the optimizer update."""
        values = rewards.detach().to(dtype=torch.float32, device="cpu").flatten().tolist()
        expected = sum(len(group) for group in groups)
        if len(values) != expected:
            raise ValueError(f"trajectory dump has {expected} trajectories but {len(values)} rewards")

        data_path = self.directory / f"rollout_{int(rollout_id):04d}.jsonl"
        summary_path = self.directory / f"rollout_{int(rollout_id):04d}.summary.json"
        if data_path.exists() or summary_path.exists():
            raise FileExistsError(f"trajectory dump for rollout {rollout_id} already exists")

        temp_path = self.directory / f".{data_path.name}.{os.getpid()}.tmp"
        digest = hashlib.sha256()
        group_summaries = []
        reward_values = []
        failed = 0
        offset = 0
        try:
            descriptor = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                for group_index, group in enumerate(groups):
                    assistant_hashes = []
                    trajectory_hashes = []
                    group_rewards = []
                    for completion_index, trajectory in enumerate(group):
                        reward = float(values[offset])
                        offset += 1
                        row = _trajectory_row(
                            trajectory,
                            rollout_id=rollout_id,
                            group_index=group_index,
                            completion_index=completion_index,
                            reward=reward,
                        )
                        encoded = (
                            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                        ).encode("utf-8")
                        handle.write(encoded)
                        digest.update(encoded)
                        assistant_hashes.append(row["assistant_sha256"])
                        trajectory_hashes.append(row["trajectory_sha256"])
                        if row["reward"] is None:
                            failed += 1
                        else:
                            group_rewards.append(float(row["reward"]))
                            reward_values.append(float(row["reward"]))
                    group_summaries.append(
                        {
                            "group_index": group_index,
                            "root_id": group[0].parts[0].sample_ids[0] if group else None,
                            "trajectories": len(group),
                            "unique_assistant_outputs": len(set(assistant_hashes)),
                            "unique_trajectories": len(set(trajectory_hashes)),
                            "rewards": group_rewards,
                        }
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, data_path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise

        summary = {
            "schema_version": 1,
            "rollout_id": int(rollout_id),
            "data_file": data_path.name,
            "data_sha256": digest.hexdigest(),
            "data_bytes": data_path.stat().st_size,
            "groups": len(groups),
            "trajectories": expected,
            "failed_trajectories": failed,
            "reward_mean": sum(reward_values) / len(reward_values) if reward_values else None,
            "fully_identical_assistant_groups": sum(
                int(item["trajectories"] > 1 and item["unique_assistant_outputs"] == 1) for item in group_summaries
            ),
            "group_summaries": group_summaries,
        }
        self._write_summary(summary_path, summary)
        return summary

    @staticmethod
    def _write_summary(path: Path, summary: Mapping[str, Any]) -> None:
        temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        payload = (json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
        try:
            descriptor = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise


__all__ = ["ARealTrajectoryDumper"]
