"""Synthetic contracts for the standalone LIN-564 trajectory evaluator."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_lin564_trajectories.py"
_SPEC = importlib.util.spec_from_file_location("analyze_lin564_trajectories", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
analyzer = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = analyzer
_SPEC.loader.exec_module(analyzer)


def _turn(role: str, text: str, *, tokens: int = 0) -> dict:
    return {"role": role, "is_gen": role == "assistant", "tokens": tokens, "text": text}


def _record(index: int, reward: float, advantage: float, turns: list[dict], *, tagged: bool) -> dict:
    gens = [turn for turn in turns if turn["is_gen"]]
    final = gens[-1]["text"] if gens else ""
    return {
        "rollout_id": 0,
        "traj_index": index,
        "reward": reward,
        "advantage": advantage,
        "num_turns": len(gens),
        "has_answer_tag": tagged,
        "final_answer": "ok" if tagged else final,
        "turns": turns,
    }


def _synthetic_records() -> list[dict]:
    search = '<tool_call>{"name":"search","arguments":{"query":["q"]}}</tool_call>'
    visit = '<tool_call>{"name":"visit","arguments":{"url":"u","goal":"g"}}</tool_call>'
    return [
        _record(
            0,
            1.0,
            2.0,
            [
                _turn("user", "q"),
                _turn("assistant", search, tokens=10),
                _turn("tool", "Results for q", tokens=50),
                _turn("assistant", visit, tokens=20),
                _turn("tool", "Evidence: useful", tokens=50),
                _turn("assistant", "<answer>ok</answer>", tokens=5),
            ],
            tagged=True,
        ),
        _record(
            1,
            1.0,
            1.0,
            [
                _turn("user", "q"),
                _turn("assistant", search + visit, tokens=8000),
                _turn("tool", "Evidence: the last (visit) call ran", tokens=20),
                _turn("assistant", "thinking one", tokens=3),
                _turn("assistant", "thinking two", tokens=4),
                _turn("assistant", "thinking three", tokens=5),
            ],
            tagged=False,
        ),
        _record(
            2,
            0.0,
            -1.0,
            [
                _turn("user", "q"),
                _turn("assistant", search, tokens=10),
                _turn("tool", "Results for q", tokens=10),
                _turn(
                    "user",
                    "You have now reached the maximum context length you can handle. Stop making tool calls.",
                ),
                _turn("assistant", "<answer>ok</answer>", tokens=10),
            ],
            tagged=True,
        ),
        _record(
            3,
            1.0,
            0.5,
            [
                _turn("user", "q"),
                _turn("assistant", '<tool_call>{"name":"search", bad json}', tokens=10),
                _turn("assistant", "<answer>ok</answer>", tokens=10),
            ],
            tagged=True,
        ),
    ]


def test_rollout_metrics_distinguish_raw_executed_clean_and_pathology() -> None:
    records = _synthetic_records()
    config = analyzer.EvalConfig(turn_cap=4)
    report = analyzer.summarize_records(records, config=config, rollout_id=0)

    assert report["n"] == 4
    assert report["reward"]["mean"] == pytest.approx(0.75)
    assert report["reward"]["strict_mean"] == pytest.approx(0.5)
    assert report["reward"]["token_weighted_mean"] == pytest.approx(8067 / 8087)
    assert report["proper_answer"] == {"n": 3, "rate": pytest.approx(0.75)}

    assert report["tools"]["raw"]["all"] == 6
    # The malformed fourth record still exposes a best-effort raw name.
    assert report["tools"]["raw"]["search"] == 4
    assert report["tools"]["raw"]["visit"] == 2
    assert report["tools"]["malformed_calls"] == 1
    assert report["tools"]["executed"]["all"] == 4
    assert report["tools"]["executed"]["search"] == 2
    assert report["tools"]["executed"]["visit"] == 2
    assert report["tools"]["clean"]["all"] == 3
    assert report["tools"]["raw_per_executed"]["all"] == pytest.approx(1.5)
    assert report["tools"]["raw_calls_per_generation"]["max"] == 2

    assert report["flags"]["spam"]["n"] == 1
    assert report["flags"]["capped_generation"]["n"] == 1
    assert report["flags"]["max_turn"]["n"] == 1
    assert report["flags"]["budget"]["n"] == 1
    assert report["flags"]["no_answer"]["n"] == 1
    assert report["flags"]["neither_run"]["n"] == 1
    assert report["flags"]["loop"]["n"] == 2
    assert report["flags"]["pathology"]["n"] == 2
    assert report["flags"]["pathology"]["token_mass"] == pytest.approx(8032 / 8087)

    positive = report["positive_advantage_pathology"]
    assert positive["pathology_mass_share"] == pytest.approx(8012 / 8092)
    assert report["advantage_by_clean_retrieval"]["clean_ge_2"]["A_tok"] == pytest.approx(2.0)
    assert report["advantage_by_clean_retrieval"]["clean_lt_2"]["A_tok"] == pytest.approx(8002 / 8052)


def test_cli_emits_json_stdout_and_table_stderr(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "rollout_000000.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in _synthetic_records()), encoding="utf-8")

    assert analyzer.main([str(path)]) == 0
    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert document["schema_version"] == 1
    assert document["rollouts"][0]["n"] == 4
    assert "raw/exe/clean" in captured.err
    assert "Tcap20" in captured.err
    assert "Atok>=2/<2" in captured.err
