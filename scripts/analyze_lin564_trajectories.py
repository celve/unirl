#!/usr/bin/env python3
"""Read-only trajectory evaluator for the LIN-564 deep-research experiments.

The input is one or more unirl rollout JSONL files (or directories containing
``rollout_*.jsonl``).  The script never writes beside, edits, or otherwise
mutates an input.  By default it emits a machine-readable JSON document on
stdout and a compact comparison table on stderr::

    python scripts/analyze_lin564_trajectories.py /path/to/traj/

The evaluator deliberately reconstructs actions from the dumped conversation,
rather than trusting raw ``<tool_call>`` counts:

* ``raw`` is every ``<tool_call>`` opener, including malformed calls;
* ``executed`` is the environment-selected (last parseable) call from a
  generated turn immediately followed by a ``role=tool`` observation;
* ``clean`` is an executed search/visit from a generation containing exactly
  one valid call and whose observation is not an error.

This distinction matters for LIN-564: a generation can contain hundreds of raw
calls while the environment executes at most one of them.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Iterator, Mapping, Sequence

_TOOL_OPEN_RE = re.compile(r"<tool_call\b[^>]*>", re.IGNORECASE)
_TOOL_CLOSE_RE = re.compile(r"</tool_call\s*>", re.IGNORECASE)
_ENV_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_ENV_TOOL_OPEN_RE = re.compile(r"<tool_call>\s*(\{)", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_NAME_RE = re.compile(r'["\']name["\']\s*:\s*["\']([^"\']+)["\']', re.IGNORECASE)

_BUDGET_SIGNALS = (
    "you have now reached the maximum context length you can handle",
    "reached the maximum context length",
    "stop making tool calls and, based on all the information above",
)
_ERROR_OBSERVATION_PATTERNS = (
    re.compile(r"^\s*error\s*:", re.IGNORECASE),
    re.compile(r"\[(?:search|visit)\]\s*(?:error|failed|empty)", re.IGNORECASE),
    re.compile(r"webpage content could not be accessed", re.IGNORECASE),
)


@dataclass(frozen=True)
class EvalConfig:
    """Thresholds that are not represented explicitly in a trajectory dump."""

    turn_cap: int = 100
    generation_token_limit: int = 8192
    capped_generation_fraction: float = 0.95
    trim_turns_at: int = 20
    neither_run_length: int = 3

    @property
    def capped_generation_tokens(self) -> int:
        return math.ceil(self.generation_token_limit * self.capped_generation_fraction)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _nonnegative_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _text(turn: Mapping[str, Any]) -> str:
    value = turn.get("text")
    return value if isinstance(value, str) else ""


def _role(turn: Mapping[str, Any]) -> str:
    value = turn.get("role")
    return str(value).lower() if value is not None else ""


def _is_gen(turn: Mapping[str, Any]) -> bool:
    if "is_gen" in turn:
        return bool(turn.get("is_gen"))
    return _role(turn) == "assistant"


def _balanced_json(text: str, start: int, *, stop: int | None = None) -> str | None:
    """Return a string-aware brace-balanced JSON object starting at ``start``."""

    limit = len(text) if stop is None else min(len(text), stop)
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, limit):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
        elif not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
    return None


def _normalize_call(raw_json: str | None) -> dict[str, Any] | None:
    if raw_json is None:
        return None
    try:
        payload = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    function = payload.get("function")
    function = function if isinstance(function, dict) else {}
    name = payload.get("name") or function.get("name")
    arguments = payload.get("arguments")
    if arguments is None:
        arguments = function.get("arguments")
    if arguments is None:
        arguments = {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    if not isinstance(name, str) or not name.strip() or not isinstance(arguments, dict):
        return None
    return {"name": name.strip().lower(), "arguments": arguments}


def _raw_call_chunks(text: str) -> Iterator[tuple[int, int, str]]:
    """Yield one bounded text chunk per raw opener, including malformed calls."""

    openers = list(_TOOL_OPEN_RE.finditer(text))
    for index, opener in enumerate(openers):
        next_open = openers[index + 1].start() if index + 1 < len(openers) else len(text)
        close = _TOOL_CLOSE_RE.search(text, opener.end(), next_open)
        end = close.start() if close is not None else next_open
        yield opener.end(), end, text[opener.end() : end]


def _all_valid_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for start, end, chunk in _raw_call_chunks(text):
        brace_rel = chunk.find("{")
        if brace_rel < 0:
            continue
        raw = _balanced_json(text, start + brace_rel, stop=end)
        call = _normalize_call(raw)
        if call is not None:
            calls.append(call)
    return calls


def _raw_call_names(text: str) -> list[str | None]:
    """Best-effort names aligned one-to-one with raw call openers."""

    names: list[str | None] = []
    for start, end, chunk in _raw_call_chunks(text):
        brace_rel = chunk.find("{")
        call = None
        if brace_rel >= 0:
            call = _normalize_call(_balanced_json(text, start + brace_rel, stop=end))
        if call is not None:
            names.append(str(call["name"]).lower())
            continue
        match = _NAME_RE.search(chunk)
        names.append(match.group(1).strip().lower() if match else None)
    return names


def _environment_call(text: str) -> dict[str, Any] | None:
    """Mirror ``ToolEnvironment.parse_tool_call`` for the action actually selected."""

    raw_json: str | None = None
    matches = list(_ENV_TOOL_CALL_RE.finditer(text))
    if matches:
        raw_json = matches[-1].group(1).strip()
    else:
        opener = _ENV_TOOL_OPEN_RE.search(text)
        if opener is not None:
            raw_json = _balanced_json(text, opener.start(1))
    return _normalize_call(raw_json)


def _has_answer(text: str, *, require_nonempty: bool) -> bool:
    matches = list(_ANSWER_RE.finditer(text))
    if not matches:
        return False
    return any(match.group(1).strip() for match in matches) if require_nonempty else True


def _proper_answer(record: Mapping[str, Any], gen_turns: Sequence[Mapping[str, Any]]) -> bool:
    """A non-empty closed answer on the terminal generation.

    New dumps carry authoritative ``has_answer_tag`` and ``final_answer`` fields;
    older dumps fall back to parsing terminal generated text.
    """

    terminal = _text(gen_turns[-1]) if gen_turns else ""
    if "has_answer_tag" in record:
        if not bool(record.get("has_answer_tag")):
            return False
        final_answer = record.get("final_answer")
        if isinstance(final_answer, str):
            return bool(final_answer.strip())
    return _has_answer(terminal, require_nonempty=True)


def _is_error_observation(text: str) -> bool:
    if not text.strip():
        return True
    return any(pattern.search(text) for pattern in _ERROR_OBSERVATION_PATTERNS)


def _contains_budget_signal(turns: Sequence[Mapping[str, Any]], record: Mapping[str, Any]) -> bool:
    corpus = "\n".join(_text(turn) for turn in turns).lower()
    # ``transcript`` is a compatibility fallback for older/clipped dumps.  Do not
    # append it when turns are complete: that would merely duplicate the text.
    if not corpus and isinstance(record.get("transcript"), str):
        corpus = str(record["transcript"]).lower()
    return any(signal in corpus for signal in _BUDGET_SIGNALS)


def _linear_percentile(values: Sequence[float | int], percentile: float) -> float | None:
    """NumPy-compatible linear percentile without a NumPy dependency."""

    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: Sequence[int | float], percentiles: Sequence[int]) -> dict[str, Any]:
    if not values:
        result: dict[str, Any] = {"n": 0, "total": 0, "mean": None, "min": None, "max": None}
        result.update({f"p{p}": None for p in percentiles})
        return result
    result = {
        "n": len(values),
        "total": sum(values),
        "mean": fmean(values),
        "min": min(values),
        "max": max(values),
    }
    result.update({f"p{p}": _linear_percentile(values, p) for p in percentiles})
    return result


def _safe_ratio(numerator: float | int, denominator: float | int) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _counter(names: Iterable[str | None]) -> dict[str, int]:
    result = {"all": 0, "search": 0, "visit": 0, "other_named": 0, "unnamed": 0}
    for name in names:
        result["all"] += 1
        if name == "search":
            result["search"] += 1
        elif name == "visit":
            result["visit"] += 1
        elif name is None:
            result["unnamed"] += 1
        else:
            result["other_named"] += 1
    return result


def _add_counter(target: dict[str, int], source: Mapping[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def analyze_trajectory(record: Mapping[str, Any], config: EvalConfig = EvalConfig()) -> dict[str, Any]:
    """Reduce one dump record to the fields needed by the rollout evaluator."""

    raw_turns = record.get("turns")
    turns = [turn for turn in raw_turns if isinstance(turn, dict)] if isinstance(raw_turns, list) else []
    gen_indices = [index for index, turn in enumerate(turns) if _is_gen(turn)]
    gen_turns = [turns[index] for index in gen_indices]
    gen_tokens_by_turn = [_nonnegative_int(turn.get("tokens")) for turn in gen_turns]
    generated_tokens = sum(value or 0 for value in gen_tokens_by_turn)

    recorded_turns = _nonnegative_int(record.get("num_turns"))
    num_turns = recorded_turns if recorded_turns is not None else len(gen_turns)
    turn_count_mismatch = recorded_turns is not None and recorded_turns != len(gen_turns)

    raw = {"all": 0, "search": 0, "visit": 0, "other_named": 0, "unnamed": 0}
    executed = {"all": 0, "search": 0, "visit": 0, "other_named": 0}
    clean = {"all": 0, "search": 0, "visit": 0}
    raw_calls_per_gen: list[int] = []
    valid_calls = 0
    malformed_calls = 0
    error_observations = 0
    neither_flags: list[bool] = []

    for index in gen_indices:
        text = _text(turns[index])
        raw_names = _raw_call_names(text)
        valid = _all_valid_calls(text)
        raw_count = len(raw_names)
        valid_count = len(valid)
        raw_calls_per_gen.append(raw_count)
        _add_counter(raw, _counter(raw_names))
        valid_calls += valid_count
        malformed_calls += max(0, raw_count - valid_count)

        next_turn = turns[index + 1] if index + 1 < len(turns) else None
        followed_by_tool = isinstance(next_turn, dict) and _role(next_turn) == "tool"
        selected = _environment_call(text)
        if followed_by_tool and selected is not None:
            name = str(selected["name"]).lower()
            executed["all"] += 1
            if name in ("search", "visit"):
                executed[name] += 1
            else:
                executed["other_named"] += 1
            observation = _text(next_turn)
            observation_error = _is_error_observation(observation)
            error_observations += int(observation_error)
            if valid_count == 1 and name in ("search", "visit") and not observation_error:
                clean["all"] += 1
                clean[name] += 1

        answered = _has_answer(text, require_nonempty=False)
        neither_flags.append(not answered and not followed_by_tool)

    longest_neither_run = 0
    current_neither_run = 0
    for is_neither in neither_flags:
        current_neither_run = current_neither_run + 1 if is_neither else 0
        longest_neither_run = max(longest_neither_run, current_neither_run)

    proper_answer = _proper_answer(record, gen_turns)
    spam = any(count > 1 for count in raw_calls_per_gen)
    capped_generation = any(
        token is not None and token >= config.capped_generation_tokens for token in gen_tokens_by_turn
    )
    max_turn = num_turns >= config.turn_cap
    budget = _contains_budget_signal(turns, record)
    no_answer = not proper_answer
    neither_run = longest_neither_run >= config.neither_run_length
    loop = budget or max_turn or no_answer or neither_run
    pathology = spam or capped_generation or loop

    return {
        "reward": _finite_float(record.get("reward")),
        "advantage": _finite_float(record.get("advantage")),
        "num_turns": num_turns,
        "turn_count_mismatch": turn_count_mismatch,
        "generated_tokens": generated_tokens,
        "missing_gen_token_turns": sum(token is None for token in gen_tokens_by_turn),
        "proper_answer": proper_answer,
        "raw": raw,
        "valid_calls": valid_calls,
        "malformed_calls": malformed_calls,
        "executed": executed,
        "clean": clean,
        "error_observations": error_observations,
        "raw_calls_per_gen": raw_calls_per_gen,
        "longest_neither_run": longest_neither_run,
        "flags": {
            "spam": spam,
            "capped_generation": capped_generation,
            "max_turn": max_turn,
            "budget": budget,
            "no_answer": no_answer,
            "neither_run": neither_run,
            "loop": loop,
            "pathology": pathology,
            "malformed": malformed_calls > 0,
            "tool_error": error_observations > 0,
        },
    }


def _flag_metrics(rows: Sequence[Mapping[str, Any]], flag: str, total_tokens: int) -> dict[str, Any]:
    selected = [row for row in rows if bool(row["flags"].get(flag))]
    token_count = sum(int(row["generated_tokens"]) for row in selected)
    return {
        "n": len(selected),
        "rate": _safe_ratio(len(selected), len(rows)),
        "generated_tokens": token_count,
        "token_mass": _safe_ratio(token_count, total_tokens),
    }


def _advantage_cohort(rows: Sequence[Mapping[str, Any]], *, at_least_two_clean: bool) -> dict[str, Any]:
    cohort = [
        row
        for row in rows
        if ((int(row["clean"]["all"]) >= 2) == at_least_two_clean)
        and row["advantage"] is not None
        and int(row["generated_tokens"]) > 0
    ]
    token_count = sum(int(row["generated_tokens"]) for row in cohort)
    weighted_advantage = sum(float(row["advantage"]) * int(row["generated_tokens"]) for row in cohort)
    rewards = [float(row["reward"]) for row in cohort if row["reward"] is not None]
    return {
        "n": len(cohort),
        "generated_tokens": token_count,
        "A_tok": _safe_ratio(weighted_advantage, token_count),
        "mean_advantage": fmean(float(row["advantage"]) for row in cohort) if cohort else None,
        "mean_reward": fmean(rewards) if rewards else None,
    }


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    config: EvalConfig = EvalConfig(),
    source: str | None = None,
    rollout_id: Any = None,
) -> dict[str, Any]:
    """Compute a JSON-serializable rollout summary."""

    rows = [analyze_trajectory(record, config) for record in records]
    rewards = [float(row["reward"]) for row in rows if row["reward"] is not None]
    strict_rewards = [
        float(row["reward"]) if bool(row["proper_answer"]) else 0.0 for row in rows if row["reward"] is not None
    ]
    total_tokens = sum(int(row["generated_tokens"]) for row in rows)
    reward_token_rows = [row for row in rows if row["reward"] is not None and int(row["generated_tokens"]) > 0]
    reward_token_denominator = sum(int(row["generated_tokens"]) for row in reward_token_rows)
    weighted_reward = sum(float(row["reward"]) * int(row["generated_tokens"]) for row in reward_token_rows)
    weighted_strict_reward = sum(
        (float(row["reward"]) if bool(row["proper_answer"]) else 0.0) * int(row["generated_tokens"])
        for row in reward_token_rows
    )

    turns = [int(row["num_turns"]) for row in rows]
    turns_trimmed = [turn for turn in turns if turn <= config.trim_turns_at]
    turn_distribution = _distribution(turns, (50, 75, 90, 95, 99))
    turn_distribution.update(
        {
            # ``trim20`` drops the >20-turn tail; ``capped20`` winsorizes it.
            # Both are explicit in JSON so the shorthand table is not ambiguous.
            "trimmed_at": config.trim_turns_at,
            "trimmed_mean": fmean(turns_trimmed) if turns_trimmed else None,
            "trimmed_n": len(turns_trimmed),
            "winsorized_mean": fmean(min(turn, config.trim_turns_at) for turn in turns) if turns else None,
            # Stable aliases requested by the LIN-564 comparison dashboard. The
            # default threshold is 20; ``trimmed_*`` above remains truthful when
            # a caller intentionally overrides it.
            "trim20_mean": fmean(turns_trimmed) if turns_trimmed else None,
            "trim20_n": len(turns_trimmed),
            "capped20_mean": fmean(min(turn, config.trim_turns_at) for turn in turns) if turns else None,
        }
    )

    raw_total = {"all": 0, "search": 0, "visit": 0, "other_named": 0, "unnamed": 0}
    executed_total = {"all": 0, "search": 0, "visit": 0, "other_named": 0}
    clean_total = {"all": 0, "search": 0, "visit": 0}
    raw_calls_per_gen: list[int] = []
    for row in rows:
        _add_counter(raw_total, row["raw"])
        _add_counter(executed_total, row["executed"])
        _add_counter(clean_total, row["clean"])
        raw_calls_per_gen.extend(int(value) for value in row["raw_calls_per_gen"])

    for totals in (raw_total, executed_total, clean_total):
        totals["per_trajectory_mean"] = _safe_ratio(totals["all"], len(rows))

    positive_rows = [
        row
        for row in rows
        if row["advantage"] is not None and float(row["advantage"]) > 0 and row["generated_tokens"] > 0
    ]
    positive_tokens = sum(int(row["generated_tokens"]) for row in positive_rows)
    positive_pathology_tokens = sum(
        int(row["generated_tokens"]) for row in positive_rows if bool(row["flags"]["pathology"])
    )
    positive_adv_mass = sum(float(row["advantage"]) * int(row["generated_tokens"]) for row in positive_rows)
    positive_adv_pathology_mass = sum(
        float(row["advantage"]) * int(row["generated_tokens"])
        for row in positive_rows
        if bool(row["flags"]["pathology"])
    )

    flag_names = (
        "spam",
        "capped_generation",
        "max_turn",
        "budget",
        "no_answer",
        "neither_run",
        "loop",
        "pathology",
        "malformed",
        "tool_error",
    )
    proper_count = sum(bool(row["proper_answer"]) for row in rows)
    summary = {
        "source": source,
        "rollout_id": rollout_id,
        "n": len(rows),
        "n_finite_reward": len(rewards),
        "reward": {
            "mean": fmean(rewards) if rewards else None,
            "strict_mean": fmean(strict_rewards) if strict_rewards else None,
            "token_weighted_mean": _safe_ratio(weighted_reward, reward_token_denominator),
            "strict_token_weighted_mean": _safe_ratio(weighted_strict_reward, reward_token_denominator),
        },
        "turns": turn_distribution,
        "generated_tokens": {
            **_distribution([int(row["generated_tokens"]) for row in rows], (50, 75, 90, 95, 99)),
            "missing_turn_token_counts": sum(int(row["missing_gen_token_turns"]) for row in rows),
        },
        "proper_answer": {"n": proper_count, "rate": _safe_ratio(proper_count, len(rows))},
        "tools": {
            "raw": raw_total,
            "valid_calls": sum(int(row["valid_calls"]) for row in rows),
            "malformed_calls": sum(int(row["malformed_calls"]) for row in rows),
            "executed": executed_total,
            "clean": clean_total,
            "error_observations": sum(int(row["error_observations"]) for row in rows),
            "raw_per_executed": {
                "all": _safe_ratio(raw_total["all"], executed_total["all"]),
                "search": _safe_ratio(raw_total["search"], executed_total["search"]),
                "visit": _safe_ratio(raw_total["visit"], executed_total["visit"]),
            },
            "raw_calls_per_generation": _distribution(raw_calls_per_gen, (50, 90, 95, 99)),
        },
        "flags": {name: _flag_metrics(rows, name, total_tokens) for name in flag_names},
        "positive_advantage_pathology": {
            "positive_n": len(positive_rows),
            "positive_generated_tokens": positive_tokens,
            "pathology_generated_tokens": positive_pathology_tokens,
            "pathology_token_share": _safe_ratio(positive_pathology_tokens, positive_tokens),
            "positive_advantage_token_mass": positive_adv_mass,
            "pathology_positive_advantage_token_mass": positive_adv_pathology_mass,
            "pathology_mass_share": _safe_ratio(positive_adv_pathology_mass, positive_adv_mass),
        },
        "advantage_by_clean_retrieval": {
            "clean_ge_2": _advantage_cohort(rows, at_least_two_clean=True),
            "clean_lt_2": _advantage_cohort(rows, at_least_two_clean=False),
        },
        "diagnostics": {
            "turn_count_mismatches": sum(bool(row["turn_count_mismatch"]) for row in rows),
            "max_longest_neither_run": max((int(row["longest_neither_run"]) for row in rows), default=0),
        },
    }
    return summary


def discover_rollout_files(inputs: Sequence[str | Path]) -> list[Path]:
    """Resolve input files without creating, touching, or rewriting any path."""

    found: list[Path] = []
    for raw_path in inputs:
        path = Path(raw_path).expanduser()
        if path.is_file():
            if path.suffix != ".jsonl":
                raise ValueError(f"expected a .jsonl file: {path}")
            found.append(path.resolve())
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)
        matches = sorted(path.rglob("rollout_*.jsonl"))
        if not matches:
            matches = sorted(path.rglob("*.jsonl"))
        found.extend(match.resolve() for match in matches)
    # Stable de-duplication, preserving the user's input order.
    return list(dict.fromkeys(found))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def _rollout_id(record: Mapping[str, Any], path: Path) -> Any:
    if record.get("rollout_id") is not None:
        return record["rollout_id"]
    match = re.search(r"rollout_(\d+)", path.stem)
    return int(match.group(1)) if match else path.stem


def analyze_paths(paths: Sequence[Path], config: EvalConfig = EvalConfig()) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in paths:
        records = load_jsonl(path)
        by_rollout: dict[Any, list[dict[str, Any]]] = {}
        for record in records:
            by_rollout.setdefault(_rollout_id(record, path), []).append(record)
        if not by_rollout:
            reports.append(summarize_records([], config=config, source=str(path), rollout_id=path.stem))
        else:
            for rollout_id, rollout_records in by_rollout.items():
                reports.append(
                    summarize_records(rollout_records, config=config, source=str(path), rollout_id=rollout_id)
                )
    return reports


def _fmt_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _fmt_percent(value: Any) -> str:
    return "-" if value is None else f"{100.0 * float(value):.1f}"


def render_table(reports: Sequence[Mapping[str, Any]]) -> str:
    """A deliberately compact dashboard; the JSON carries every exact field."""

    headers = (
        "roll",
        "n",
        "R",
        "Rstrict",
        "Rtok",
        "Tmean",
        "T50",
        "T90",
        "Tmax",
        "Tcap20",
        "tokmean",
        "ans%",
        "raw/exe/clean",
        "raw:exe",
        "call99/max",
        "spam%",
        "cap%",
        "loop%",
        "path%/tok%",
        "+Apath%",
        "Atok>=2/<2",
    )
    body: list[tuple[str, ...]] = []
    rollout_id_counts: dict[str, int] = {}
    for report in reports:
        key = str(report.get("rollout_id", "?"))
        rollout_id_counts[key] = rollout_id_counts.get(key, 0) + 1
    for report in reports:
        rollout_label = str(report.get("rollout_id", "?"))
        if rollout_id_counts[rollout_label] > 1:
            source = Path(str(report.get("source") or "unknown"))
            run_label = source.parent.parent.name if source.parent.name in {"traj", "rollout"} else source.parent.name
            rollout_label = f"{run_label}:{rollout_label}"
        reward = report["reward"]
        turns = report["turns"]
        tokens = report["generated_tokens"]
        tools = report["tools"]
        flags = report["flags"]
        positive = report["positive_advantage_pathology"]
        cohorts = report["advantage_by_clean_retrieval"]
        body.append(
            (
                rollout_label,
                str(report["n"]),
                _fmt_number(reward["mean"]),
                _fmt_number(reward["strict_mean"]),
                _fmt_number(reward["token_weighted_mean"]),
                _fmt_number(turns["mean"], 2),
                _fmt_number(turns["p50"], 1),
                _fmt_number(turns["p90"], 1),
                _fmt_number(turns["max"], 0),
                _fmt_number(turns["capped20_mean"], 2),
                _fmt_number(tokens["mean"], 0),
                _fmt_percent(report["proper_answer"]["rate"]),
                f"{tools['raw']['all']}/{tools['executed']['all']}/{tools['clean']['all']}",
                _fmt_number(tools["raw_per_executed"]["all"], 2),
                f"{_fmt_number(tools['raw_calls_per_generation']['p99'], 1)}/"
                f"{_fmt_number(tools['raw_calls_per_generation']['max'], 0)}",
                _fmt_percent(flags["spam"]["rate"]),
                _fmt_percent(flags["capped_generation"]["rate"]),
                _fmt_percent(flags["loop"]["rate"]),
                f"{_fmt_percent(flags['pathology']['rate'])}/{_fmt_percent(flags['pathology']['token_mass'])}",
                _fmt_percent(positive["pathology_mass_share"]),
                f"{_fmt_number(cohorts['clean_ge_2']['A_tok'])}/{_fmt_number(cohorts['clean_lt_2']['A_tok'])}",
            )
        )

    widths = [len(header) for header in headers]
    for row in body:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def format_row(row: Sequence[str]) -> str:
        return "  ".join(cell.rjust(width) for cell, width in zip(row, widths))

    lines = [format_row(headers), format_row(tuple("-" * width for width in widths))]
    lines.extend(format_row(row) for row in body)
    return "\n".join(lines)


def _definitions(config: EvalConfig) -> dict[str, Any]:
    return {
        "strict_reward": "reward, but zero unless the terminal generation has a non-empty closed <answer>",
        "token_weighted_reward": "mean reward weighted by total generated tokens per trajectory",
        "turns_trimmed_mean": f"mean after dropping trajectories with turns > {config.trim_turns_at}",
        "turns_capped20_mean": f"mean(min(turns, {config.trim_turns_at})); the robust turn-growth endpoint",
        "raw_call": "every <tool_call> opener, including malformed calls",
        "executed_call": "environment-selected last parseable call immediately followed by role=tool",
        "clean_call": "executed search/visit with exactly one valid call in the generation and non-error observation",
        "spam": "any generated turn containing more than one raw call",
        "capped_generation": f"any generated turn with tokens >= {config.capped_generation_tokens} "
        f"(ceil({config.capped_generation_fraction} * {config.generation_token_limit}))",
        "max_turn": f"num_turns >= {config.turn_cap}",
        "budget": "force-answer/context-budget warning text appears in the trajectory",
        "neither_run": f">= {config.neither_run_length} consecutive generated turns with neither answer nor following tool observation",
        "loop": "budget OR max_turn OR no proper answer OR neither_run",
        "pathology": "spam OR capped_generation OR loop",
        "token_mass": "share of all generated tokens belonging to trajectories carrying the flag",
        "positive_advantage_pathology_mass": "share of sum(max(advantage,0)*generated_tokens) from pathological trajectories",
        "A_tok": "sum(advantage*generated_tokens)/sum(generated_tokens) in the stated clean-call cohort",
        "percentiles": "linear interpolation at (n-1)*q, matching numpy.percentile default",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="rollout JSONL file(s) or directories")
    parser.add_argument("--turn-cap", type=int, default=100, help="max-turn loop threshold (default: 100)")
    parser.add_argument(
        "--generation-token-limit", type=int, default=8192, help="per-generation configured token limit"
    )
    parser.add_argument(
        "--capped-generation-fraction",
        type=float,
        default=0.95,
        help="fraction of per-generation limit that marks a capped generation (default: .95)",
    )
    parser.add_argument("--trim-turns-at", type=int, default=20, help="turn threshold used by trimmed/capped means")
    parser.add_argument("--neither-run-length", type=int, default=3, help="consecutive NEITHER turns marking a loop")
    parser.add_argument("--json-only", action="store_true", help="suppress the stderr table")
    parser.add_argument("--table-only", action="store_true", help="print only the table on stdout")
    parser.add_argument("--compact-json", action="store_true", help="emit one-line JSON instead of indented JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.json_only and args.table_only:
        raise SystemExit("--json-only and --table-only are mutually exclusive")
    if args.turn_cap < 1 or args.generation_token_limit < 1 or args.trim_turns_at < 0:
        raise SystemExit("turn/token thresholds must be positive (trim threshold may be zero)")
    if not 0 < args.capped_generation_fraction <= 1:
        raise SystemExit("--capped-generation-fraction must be in (0, 1]")
    if args.neither_run_length < 1:
        raise SystemExit("--neither-run-length must be positive")

    config = EvalConfig(
        turn_cap=args.turn_cap,
        generation_token_limit=args.generation_token_limit,
        capped_generation_fraction=args.capped_generation_fraction,
        trim_turns_at=args.trim_turns_at,
        neither_run_length=args.neither_run_length,
    )
    paths = discover_rollout_files(args.inputs)
    if not paths:
        raise SystemExit("no rollout JSONL files found")
    reports = analyze_paths(paths, config)
    document = {
        "schema_version": 1,
        "definitions": _definitions(config),
        "config": {
            "turn_cap": config.turn_cap,
            "generation_token_limit": config.generation_token_limit,
            "capped_generation_fraction": config.capped_generation_fraction,
            "capped_generation_tokens": config.capped_generation_tokens,
            "trim_turns_at": config.trim_turns_at,
            "neither_run_length": config.neither_run_length,
        },
        "files": [str(path) for path in paths],
        "rollouts": reports,
    }

    table = render_table(reports)
    if args.table_only:
        print(table)
    else:
        print(json.dumps(document, ensure_ascii=False, indent=None if args.compact_json else 2, sort_keys=True))
        if not args.json_only:
            print(table, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
