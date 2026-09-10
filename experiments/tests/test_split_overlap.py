"""E0 gate: the split-overlap audit catches both reformatted and re-ID'd duplicates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.audit_split_overlap import load_split, main, normalize_prompt, prompt_key  # noqa: E402


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_normalization_collapses_case_punctuation_and_whitespace():
    assert normalize_prompt("What is  2+2?") == normalize_prompt("what is 2 + 2")
    assert normalize_prompt("Café") == normalize_prompt("cafe")


def test_prompt_key_is_stable_and_distinguishing():
    assert prompt_key("solve for x") == prompt_key("Solve for X.")
    assert prompt_key("solve for x") != prompt_key("solve for y")


def test_txt_split_reads_one_prompt_per_line(tmp_path):
    path = tmp_path / "test.txt"
    path.write_text("a red cube\na blue sphere\n")
    prompts, ids = load_split(path, prompt_fields=["prompt"], id_fields=["problem_id"])
    assert len(prompts) == 2 and ids == set()


def test_clean_splits_pass(tmp_path, capsys):
    _write_jsonl(tmp_path / "train.jsonl", [{"prompt": "add 1 and 2", "problem_id": "p1"}])
    _write_jsonl(tmp_path / "test.jsonl", [{"prompt": "multiply 3 by 4", "problem_id": "p2"}])
    code = main([f"--split=train={tmp_path/'train.jsonl'}", f"--split=test={tmp_path/'test.jsonl'}"])
    assert code == 0
    assert json.loads(capsys.readouterr().out.split("\nOVERLAP")[0])["clean"] is True


def test_reformatted_duplicate_is_caught(tmp_path):
    _write_jsonl(tmp_path / "train.jsonl", [{"prompt": "What is 2+2?", "problem_id": "p1"}])
    _write_jsonl(tmp_path / "test.jsonl", [{"prompt": "what is  2 + 2", "problem_id": "p999"}])
    assert main([f"--split=train={tmp_path/'train.jsonl'}", f"--split=test={tmp_path/'test.jsonl'}"]) == 1


def test_reused_problem_id_is_caught(tmp_path):
    _write_jsonl(tmp_path / "train.jsonl", [{"prompt": "wholly different text", "problem_id": "p1"}])
    _write_jsonl(tmp_path / "test.jsonl", [{"prompt": "another unrelated prompt", "problem_id": "p1"}])
    assert main([f"--split=train={tmp_path/'train.jsonl'}", f"--split=test={tmp_path/'test.jsonl'}"]) == 1


def test_three_splits_are_compared_pairwise(tmp_path, capsys):
    _write_jsonl(tmp_path / "train.jsonl", [{"prompt": "alpha", "problem_id": "p1"}])
    _write_jsonl(tmp_path / "val.jsonl", [{"prompt": "beta", "problem_id": "p2"}])
    _write_jsonl(tmp_path / "test.jsonl", [{"prompt": "beta", "problem_id": "p3"}])
    assert main([
        f"--split=train={tmp_path/'train.jsonl'}",
        f"--split=val={tmp_path/'val.jsonl'}",
        f"--split=test={tmp_path/'test.jsonl'}",
    ]) == 1
    report = json.loads(capsys.readouterr().out.split("\nOVERLAP")[0])
    assert len(report["overlaps"]) == 3
    offending = [o for o in report["overlaps"] if o["shared_prompt_count"]]
    assert [o["pair"] for o in offending] == ["test|val"]


def test_empty_extraction_fails_loudly(tmp_path):
    _write_jsonl(tmp_path / "bad.jsonl", [{"unexpected_field": "x"}])
    try:
        load_split(tmp_path / "bad.jsonl", prompt_fields=["prompt"], id_fields=["problem_id"])
    except ValueError as exc:
        assert "no prompts or problem ids" in str(exc)
    else:
        raise AssertionError("expected a ValueError for an unextractable split")
