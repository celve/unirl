"""Prepare ASearcher LRM35k deep-research data as deterministic JSONL.

The default source is ASearcher LRM35k. Each row preserves the source question
and answer value, adds a stable prompt ID, and writes an adjacent conversion
manifest consumed by the deep-research recipe.

  python -m unirl.utils.prepare_asearcher \
    --source /path/to/ASearcher-LRM-35k.jsonl \
    --out-dir data/asearcher
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, deque
from typing import Any, Dict, Iterator, Optional, Sequence

_QUESTION_KEYS = ("question", "prompt", "query")
_ANSWER_KEYS = ("answer", "gt", "golden_answers", "ground_truth", "label")
_ID_KEYS = ("query_id", "id", "qid")


def _first_present(row: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_source(
    source: Optional[str],
    hf_name: str,
    split: str,
    revision: Optional[str],
) -> Iterator[Dict[str, Any]]:
    if source is not None:
        if not os.path.isfile(source):
            raise FileNotFoundError(f"ASearcher source not found: {source}")
        with open(source, encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"ASearcher source line {line_number} must be a JSON object")
                yield row
        return

    from datasets import load_dataset  # lazy: only needed when downloading

    kwargs: Dict[str, Any] = {"split": split, "streaming": True}
    if revision is not None:
        kwargs["revision"] = revision
    for row in load_dataset(hf_name, **kwargs):
        yield dict(row)


def _answer_kind(answer: Any) -> str:
    if not isinstance(answer, (list, tuple)):
        return "scalar"
    if len(answer) == 0:
        return "empty_list"
    if len(answer) == 1:
        return "singleton_list"
    return "multi_answer_list"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare lossless ASearcher LRM35k JSONL.")
    parser.add_argument("--out-dir", default="data/asearcher")
    parser.add_argument("--source", default=None, help="local ASearcher JSONL (otherwise download from HF)")
    parser.add_argument("--hf-name", default="inclusionAI/ASearcher-train-data")
    parser.add_argument("--split", default="ASearcherLRM35k")
    parser.add_argument("--revision", default=None, help="optional immutable Hugging Face dataset revision")
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--limit", type=int, default=0, help="cap written rows (0 = all)")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "train.jsonl")
    manifest_path = args.manifest_path or f"{out_path}.manifest.json"
    source_path = os.path.abspath(args.source) if args.source is not None else None

    seen = 0
    written = 0
    rejected: Counter[str] = Counter()
    answer_types: Counter[str] = Counter()
    prompt_ids: set[str] = set()
    first_ids: list[str] = []
    last_ids: deque[str] = deque(maxlen=5)
    prompt_digest = hashlib.sha256()

    temporary_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=args.out_dir,
            prefix=".asearcher-",
            suffix=".jsonl",
            delete=False,
        ) as output:
            temporary_path = output.name
            for source_row, row in enumerate(_iter_source(source_path, args.hf_name, args.split, args.revision)):
                seen += 1
                question = _first_present(row, _QUESTION_KEYS)
                if not isinstance(question, str) or question == "":
                    rejected["missing_or_invalid_question"] += 1
                    continue

                answer = _first_present(row, _ANSWER_KEYS)
                if answer is None:
                    rejected["missing_answer"] += 1
                    continue

                source_id = _first_present(row, _ID_KEYS)
                prompt_id = f"source-row:{source_row}" if source_id is None or str(source_id) == "" else str(source_id)
                if prompt_id in prompt_ids:
                    raise ValueError(f"duplicate resolved prompt_id {prompt_id!r} at source row {source_row}")
                prompt_ids.add(prompt_id)

                converted = {
                    "prompt": question,
                    "prompt_id": prompt_id,
                    "metadata": {"answer": answer, "source_row": source_row},
                }
                output.write(json.dumps(converted, ensure_ascii=False, separators=(",", ":")) + "\n")

                prompt_bytes = question.encode("utf-8")
                prompt_digest.update(len(prompt_bytes).to_bytes(8, "big"))
                prompt_digest.update(prompt_bytes)
                answer_types[_answer_kind(answer)] += 1
                if len(first_ids) < 5:
                    first_ids.append(prompt_id)
                last_ids.append(prompt_id)
                written += 1
                if args.limit and written >= args.limit:
                    break

        os.replace(temporary_path, out_path)
        temporary_path = None
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)

    manifest = {
        "schema_version": 1,
        "dataset": {
            "kind": "local" if source_path is not None else "huggingface",
            "name": args.hf_name,
            "revision": args.revision,
            "split": args.split,
            "source_path": source_path,
            "source_sha256": _sha256_file(source_path) if source_path is not None else None,
        },
        "output_path": os.path.abspath(out_path),
        "converted_file_sha256": _sha256_file(out_path),
        "prompt_text_sha256": prompt_digest.hexdigest(),
        "seen_rows": seen,
        "written_rows": written,
        "rejected_rows": seen - written,
        "rejection_reasons": dict(sorted(rejected.items())),
        "unique_prompt_ids": len(prompt_ids),
        "first_prompt_ids": first_ids,
        "last_prompt_ids": list(last_ids),
        "answer_type_counts": dict(sorted(answer_types.items())),
    }
    with open(manifest_path, "w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")

    print(f"wrote {written}/{seen} examples -> {out_path}")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
