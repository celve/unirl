#!/usr/bin/env python3
"""Convert the Geometry3K dataset into JSONL + flat images for TextPromptDataset.

Source layout (flat — all problems in one directory):
    geometry3k/{id}/
        data.json        # id, annotat_text, choices, answer, precise_value,
                         #   problem_type_graph, problem_type_goal, data_type
        img_diagram.png

Source layout (split subdirectories):
    geometry3k/{split}/{id}/
        data.json
        img_diagram.png

Target layout:
    {out_dir}/
        train.jsonl
        val.jsonl
        test.jsonl
        images/
            0.png
            1.png
            ...

Usage:
    python scripts/convert_geo3k.py --src /path/to/geometry3k --out /path/to/output
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ANSWER_LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}
SPLIT_RANGES = {
    "train": range(0, 2101),
    "val": range(2101, 2401),
    "test": range(2401, 3002),
}
SPLIT_NAMES = ("train", "val", "test")


def _load_problem(problem_dir: Path) -> Dict[str, Any]:
    """Load data.json from a single problem directory."""
    data_path = problem_dir / "data.json"
    with open(data_path, "r") as f:
        return json.load(f)


def _format_prompt(data: Dict[str, Any]) -> str:
    """Build the multiple-choice prompt string."""
    choices = data["choices"]
    text = data["annotat_text"]
    return (
        f"Look at the geometry diagram. {text}\n\n"
        f"A) {choices[0]}\n"
        f"B) {choices[1]}\n"
        f"C) {choices[2]}\n"
        f"D) {choices[3]}\n\n"
        "Answer with the letter only."
    )


def _build_record(data: Dict[str, Any], problem_id: int) -> Dict[str, Any]:
    """Build one JSONL record from a parsed data.json."""
    choices = data["choices"]
    answer_letter = data["answer"]
    answer_index = ANSWER_LETTER_TO_INDEX.get(answer_letter.upper(), -1)
    answer_value = choices[answer_index] if 0 <= answer_index < len(choices) else ""

    return {
        "prompt": _format_prompt(data),
        "prompt_id": f"geo3k:{problem_id}",
        "media_refs": [
            {
                "modality": "image",
                "role": "condition",
                "uri": f"images/{problem_id}.png",
            }
        ],
        "metadata": {
            "answer": answer_letter,
            "answer_value": answer_value,
            "choices": choices,
            "problem_type_graph": data.get("problem_type_graph", ""),
            "problem_type_goal": data.get("problem_type_goal", ""),
            "source_id": problem_id,
        },
    }


def _discover_flat_layout(src: Path) -> Dict[str, List[Tuple[int, Path]]]:
    """Try the flat layout: all problems directly under src, split via data_type field."""
    split_problems: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
    found_any = False

    for entry in sorted(src.iterdir()):
        if not entry.is_dir() or not entry.name.isdigit():
            continue
        data_path = entry / "data.json"
        if not data_path.exists():
            continue

        found_any = True
        with open(data_path, "r") as f:
            data = json.load(f)

        problem_id = int(entry.name)

        # Determine split from data_type field or fall back to ID ranges.
        data_type = data.get("data_type", "").strip().lower()
        if data_type in SPLIT_NAMES:
            split_name = data_type
        else:
            split_name = _split_from_id(problem_id)

        if split_name is not None:
            split_problems[split_name].append((problem_id, entry))

    if not found_any:
        return {}
    return dict(split_problems)


def _discover_split_layout(src: Path) -> Dict[str, List[Tuple[int, Path]]]:
    """Try the split-subdirectory layout: src/{train,val,test}/{id}/."""
    split_problems: Dict[str, List[Tuple[int, Path]]] = {}

    for split_name in SPLIT_NAMES:
        split_dir = src / split_name
        if not split_dir.is_dir():
            continue
        problems: List[Tuple[int, Path]] = []
        for entry in sorted(split_dir.iterdir()):
            if not entry.is_dir() or not entry.name.isdigit():
                continue
            if (entry / "data.json").exists():
                problems.append((int(entry.name), entry))
        if problems:
            split_problems[split_name] = problems

    return split_problems


def _split_from_id(problem_id: int) -> str | None:
    """Determine split from problem ID using the canonical ranges."""
    for split_name, id_range in SPLIT_RANGES.items():
        if problem_id in id_range:
            return split_name
    return None


def convert(src: Path, out: Path) -> None:
    """Run the full conversion."""
    # Discover problems — try flat layout first, then split subdirectories.
    split_problems = _discover_flat_layout(src)
    layout_name = "flat"
    if not split_problems:
        split_problems = _discover_split_layout(src)
        layout_name = "split-subdirectory"

    if not split_problems:
        print(
            f"ERROR: No problems found under {src}. Expected numbered directories "
            "containing data.json either directly or under train/val/test subdirectories.",
            file=sys.stderr,
        )
        sys.exit(1)

    total_problems = sum(len(v) for v in split_problems.values())
    print(f"Detected {layout_name} layout with {total_problems} problems across {len(split_problems)} split(s).")

    # Prepare output directories.
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Process each split.
    grand_total = 0
    for split_name in SPLIT_NAMES:
        problems = split_problems.get(split_name, [])
        if not problems:
            print(f"  {split_name:>5s}: 0 problems (skipped)")
            continue

        problems.sort(key=lambda x: x[0])
        jsonl_path = out / f"{split_name}.jsonl"
        written = 0
        skipped = 0

        with open(jsonl_path, "w") as f:
            for problem_id, problem_dir in problems:
                try:
                    data = _load_problem(problem_dir)
                except (json.JSONDecodeError, OSError) as exc:
                    print(f"  WARNING: skipping {problem_dir}: {exc}", file=sys.stderr)
                    skipped += 1
                    continue

                # Copy image.
                src_img = problem_dir / "img_diagram.png"
                dst_img = images_dir / f"{problem_id}.png"
                if src_img.exists():
                    shutil.copy2(src_img, dst_img)
                else:
                    print(f"  WARNING: missing image for problem {problem_id}", file=sys.stderr)

                record = _build_record(data, problem_id)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

        print(
            f"  {split_name:>5s}: {written} problems written to {jsonl_path}"
            + (f" ({skipped} skipped)" if skipped else "")
        )
        grand_total += written

    print(f"\nDone. {grand_total} total records across {len(split_problems)} splits written to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Geometry3K dataset to JSONL + flat images for TextPromptDataset."
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Path to the source geometry3k directory.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to the output directory.",
    )
    args = parser.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()

    if not src.is_dir():
        print(f"ERROR: source directory does not exist: {src}", file=sys.stderr)
        sys.exit(1)

    convert(src, out)


if __name__ == "__main__":
    main()
