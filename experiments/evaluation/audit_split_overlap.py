#!/usr/bin/env python
"""Train/validation/test overlap audit on normalized prompts and problem IDs.

RQ1's endpoint is only meaningful if the held-out sets are actually held out. The
description requires this audit on normalized prompts *and* problem IDs, because a
reformatted duplicate defeats an ID-only check and a reused ID defeats a text-only
one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[^\w\s]")


def normalize_prompt(text: str) -> str:
    """Casefold, strip accents and punctuation, collapse whitespace — a duplicate-catching key."""
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().casefold()
    return _WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", folded)).strip()


def prompt_key(text: str) -> str:
    return hashlib.sha256(normalize_prompt(text).encode()).hexdigest()[:16]


def _iter_rows(path: Path) -> Iterable[Dict[str, object]]:
    """Read a split as JSONL objects, or as one prompt per line for a plain .txt list."""
    with path.open() as handle:
        for lineno, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            if path.suffix == ".txt":
                yield {"prompt": line, "problem_id": None}
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: not valid JSON ({exc})") from exc
            yield row


def _extract(row: Dict[str, object], *, prompt_fields: Sequence[str], id_fields: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    prompt = next((str(row[f]) for f in prompt_fields if f in row and row[f] is not None), None)
    problem_id = next((str(row[f]) for f in id_fields if f in row and row[f] is not None), None)
    return prompt, problem_id


def load_split(
    path: Path,
    *,
    prompt_fields: Sequence[str],
    id_fields: Sequence[str],
) -> Tuple[Dict[str, str], Set[str]]:
    """Return normalized-prompt-key → original text, plus the split's problem IDs."""
    prompts: Dict[str, str] = {}
    ids: Set[str] = set()
    for row in _iter_rows(path):
        prompt, problem_id = _extract(row, prompt_fields=prompt_fields, id_fields=id_fields)
        if prompt is not None:
            prompts.setdefault(prompt_key(prompt), prompt)
        if problem_id is not None:
            ids.add(problem_id)
    if not prompts and not ids:
        raise ValueError(f"{path}: no prompts or problem ids extracted — check --prompt-fields/--id-fields")
    return prompts, ids


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", action="append", required=True, metavar="NAME=PATH", help="repeatable, e.g. train=train.jsonl")
    parser.add_argument("--prompt-fields", default="prompt,question,text,problem")
    parser.add_argument("--id-fields", default="problem_id,id,uid,index")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=10)
    args = parser.parse_args(argv)

    prompt_fields = [f.strip() for f in args.prompt_fields.split(",") if f.strip()]
    id_fields = [f.strip() for f in args.id_fields.split(",") if f.strip()]

    splits: Dict[str, Tuple[Dict[str, str], Set[str]]] = {}
    for entry in args.split:
        if "=" not in entry:
            raise SystemExit(f"--split expects NAME=PATH, got {entry!r}")
        name, _, raw_path = entry.partition("=")
        splits[name] = load_split(Path(raw_path), prompt_fields=prompt_fields, id_fields=id_fields)

    names = sorted(splits)
    overlaps: List[Dict[str, object]] = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            left_prompts, left_ids = splits[left]
            right_prompts, right_ids = splits[right]
            shared_prompts = sorted(set(left_prompts) & set(right_prompts))
            shared_ids = sorted(left_ids & right_ids)
            overlaps.append(
                {
                    "pair": f"{left}|{right}",
                    "shared_prompt_count": len(shared_prompts),
                    "shared_problem_id_count": len(shared_ids),
                    "shared_prompt_examples": [left_prompts[k] for k in shared_prompts[: args.max_examples]],
                    "shared_problem_id_examples": shared_ids[: args.max_examples],
                }
            )

    report = {
        "splits": {
            name: {
                "unique_normalized_prompts": len(prompts),
                "unique_problem_ids": len(ids),
            }
            for name, (prompts, ids) in splits.items()
        },
        "overlaps": overlaps,
        "clean": all(o["shared_prompt_count"] == 0 and o["shared_problem_id_count"] == 0 for o in overlaps),
    }

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload)

    if not report["clean"]:
        print("\nOVERLAP AUDIT FAILED — held-out evaluation is contaminated on these pairs:")
        for overlap in overlaps:
            if overlap["shared_prompt_count"] or overlap["shared_problem_id_count"]:
                print(
                    f"  - {overlap['pair']}: {overlap['shared_prompt_count']} prompts, "
                    f"{overlap['shared_problem_id_count']} problem ids"
                )
        return 1
    print("\nOVERLAP AUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
