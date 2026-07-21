"""Trajectory dump — persist every rollout's per-trajectory conversation for debugging.

Opt-in via ``$TRAJ_DUMP_DIR``: when set, :func:`maybe_dump_trajectories` writes one
JSONL file per rollout (``rollout_<id>.jsonl``) with **one line per trajectory**
(GRPO sibling). Each line carries the decoded multi-turn conversation (prompt →
assistant/tool turns → final answer), the per-turn token counts, the trajectory's
scalar reward + GRPO advantage, the reference answer, an explicit terminal
``has_answer_tag`` diagnostic, and a pre-rendered ``transcript`` string for
eyeballing. Unset → a no-op (zero overhead).

Granularity: **rollout → file**, **sample (trajectory) → line**, **turn (Part) →
an element of the line's ``turns`` array**.

Failure-isolated by contract: a debug dump must never sink a training step, so the
whole write is wrapped and only logged on error. Non-finite (NaN) rewards — the
crashed-trajectory marker (see ``AgenticTrainer._group_advantages``) — serialize as
``reward: null`` with ``crashed: true`` rather than breaking JSON.

Env knobs:
- ``TRAJ_DUMP_DIR``       : output directory (absolute recommended — training runs
                            under a Hydra-changed cwd). Unset disables the dump.
- ``TRAJ_DUMP_MAX_CHARS`` : per-turn text cap (default ``0`` = full fidelity). When
                            set >0, over-long turns are trimmed and flagged
                            ``"truncated": true`` — never silently.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional

from unirl.types.sample import Sample

logger = logging.getLogger(__name__)

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _turn_text(primitive: Any) -> Optional[str]:
    """The first row's text of a Part's primitive, or ``None`` (undecoded / non-text)."""
    texts = getattr(primitive, "texts", None)
    if not texts:
        return None
    return texts[0]


def _turn_tokens(part: Any) -> Optional[int]:
    """Row-0 token count from the Part's encoded ``segment.lengths``, or ``None``."""
    lengths = getattr(getattr(part, "segment", None), "lengths", None)
    if lengths is None or not hasattr(lengths, "numel") or lengths.numel() == 0:
        return None
    return int(lengths[0].item())


def _finite_or_none(x: Any) -> Optional[float]:
    """``float(x)`` if finite, else ``None`` — so a NaN/inf reward stays valid JSON."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if math.isfinite(xf) else None


def _clip(text: Optional[str], max_chars: int) -> tuple[Optional[str], bool]:
    """Trim ``text`` to ``max_chars`` (0 = unlimited); return ``(text, truncated)``."""
    if text is None or max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars] + f"…[+{len(text) - max_chars} chars]", True


def _render_trajectory(
    traj: Sample,
    *,
    rollout_id: int,
    traj_index: int,
    reward: Any,
    advantage: Any,
    group_id: str,
    max_chars: int,
) -> Dict[str, Any]:
    """One trajectory :class:`Sample` → a JSON-serializable dump record.

    Walks the trajectory's ordered ``parts`` (the turns): root input (``user``),
    then the interleaved ``assistant`` generations and ``tool`` observations, ending
    at the final ``assistant`` answer. Each part contributes ``{index, role, is_gen,
    tokens, truncated, text}``.
    """
    parts = list(traj.parts)
    turns: List[Dict[str, Any]] = []
    for i, part in enumerate(parts):
        text, truncated = _clip(_turn_text(part.primitive), max_chars)
        answer_injected = any(bool((m or {}).get("answer_injected")) for m in (part.metadata or []))
        turns.append(
            {
                "index": i,
                "role": part.resolved_role(),
                "is_gen": bool(part.is_gen),
                "tokens": _turn_tokens(part),
                "truncated": truncated,
                "answer_injected": answer_injected,
                "text": text,
            }
        )

    # Final answer: the TERMINAL generation's ``<answer>…</answer>`` (else its
    # text). Do not skip an empty terminal turn and accidentally report an older
    # tool call as the answer; reward grading uses this same terminal-turn rule.
    gen_parts = traj.gen_parts()
    last_gen = _turn_text(gen_parts[-1].primitive) if gen_parts else None
    final_answer: Optional[str] = None
    has_answer_tag = False
    if last_gen is not None:
        m = list(_ANSWER_RE.finditer(last_gen))
        has_answer_tag = bool(m)
        final_answer = m[-1].group(1).strip() if m else last_gen.strip()

    root = parts[0] if parts else None
    reward_f = _finite_or_none(reward)
    record: Dict[str, Any] = {
        "rollout_id": rollout_id,
        "traj_index": traj_index,
        "root_id": root.sample_ids[0] if (root is not None and root.sample_ids) else None,
        "leaf_id": parts[-1].sample_ids[0] if (parts and parts[-1].sample_ids) else None,
        "group_id": group_id,
        "reward": reward_f,
        "crashed": reward_f is None,  # NaN reward = crashed trajectory (env bug, not a policy outcome)
        "advantage": _finite_or_none(advantage),
        "num_turns": len(traj.gen_parts()),  # assistant turns actually trained
        "num_logical_turns": sum(
            1
            for part in traj.gen_parts()
            if not any(bool((m or {}).get("answer_injected")) for m in (part.metadata or []))
        ),
        "num_parts": len(parts),
        "answer_injected": any(t["answer_injected"] for t in turns),
        "prompt": _turn_text(root.primitive) if root is not None else None,
        "reference_answer": (root.metadata[0] or {}).get("answer") if (root is not None and root.metadata) else None,
        "has_answer_tag": has_answer_tag,
        "final_answer": final_answer,
        "turns": turns,
    }
    record["transcript"] = _render_transcript(record)
    return record


def _render_transcript(record: Dict[str, Any]) -> str:
    """A human-readable transcript of a dump record (embedded in the JSON line)."""
    header = (
        f"# rollout {record['rollout_id']} · traj {record['traj_index']} · {record['leaf_id']}\n"
        f"# reward={record['reward']} advantage={record['advantage']} "
        f"turns={record['num_turns']} crashed={record['crashed']} "
        f"has_answer_tag={record['has_answer_tag']}\n"
        f"# reference_answer: {record['reference_answer']}"
    )
    blocks = [header]
    for t in record["turns"]:
        tok = f" · {t['tokens']} tok" if t["tokens"] is not None else ""
        trunc = " · TRUNCATED" if t["truncated"] else ""
        blocks.append(f"===== [{t['index']}] {t['role']}{tok}{trunc} =====\n{t['text'] or ''}")
    return "\n\n".join(blocks)


def maybe_dump_trajectories(
    trajs: List[Sample],
    rewards: Any,
    advantages: Any,
    group_ids: List[str],
    *,
    rollout_id: int,
    dump_dir: Optional[str] = None,
) -> None:
    """Write this rollout's trajectories to ``<dir>/rollout_<id>.jsonl`` (one line per
    trajectory) when ``$TRAJ_DUMP_DIR`` (or ``dump_dir``) is set; else a no-op.

    ``rewards``/``advantages`` are per-trajectory scalars aligned to ``trajs`` (a
    tensor or any sequence). Never raises — a dump failure is logged, not propagated.
    """
    out_dir = dump_dir or os.environ.get("TRAJ_DUMP_DIR")
    if not out_dir or not trajs:
        return
    try:
        max_chars = int(os.environ.get("TRAJ_DUMP_MAX_CHARS", "0") or 0)
        rewards_l = rewards.detach().to("cpu").tolist() if hasattr(rewards, "detach") else list(rewards)
        adv_l = advantages.detach().to("cpu").tolist() if hasattr(advantages, "detach") else list(advantages)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"rollout_{rollout_id:06d}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for i, traj in enumerate(trajs):
                record = _render_trajectory(
                    traj,
                    rollout_id=rollout_id,
                    traj_index=i,
                    reward=rewards_l[i] if i < len(rewards_l) else float("nan"),
                    advantage=adv_l[i] if i < len(adv_l) else float("nan"),
                    group_id=group_ids[i] if i < len(group_ids) else "",
                    max_chars=max_chars,
                )
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("trajectory dump: rollout %d — wrote %d trajectories to %s", rollout_id, len(trajs), path)
    except Exception:  # noqa: BLE001 — a debug dump must never sink a training step
        logger.warning("trajectory dump failed for rollout %d", rollout_id, exc_info=True)


__all__ = ["maybe_dump_trajectories"]
