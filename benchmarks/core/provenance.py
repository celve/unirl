"""Generation provenance: refuse to extend outputs another configuration produced — see README.md."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

MANIFEST_NAME = "generation_manifest.json"


def fingerprint(payload: Dict[str, Any]) -> str:
    """Stable digest of a generation configuration."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _differing_fields(previous: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for key in sorted(set(previous) | set(current)):
        was, now = previous.get(key, "<absent>"), current.get(key, "<absent>")
        if was != now:
            lines.append(f"    {key}: {was!r} -> {now!r}")
    return lines


def guard(manifest_path: Path, payload: Dict[str, Any], *, outputs_exist: bool) -> None:
    """Refuse to extend outputs that a different configuration produced."""
    current = {**payload, "fingerprint": fingerprint(payload)}

    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("fingerprint") != current["fingerprint"]:
            fields = _differing_fields(
                {k: v for k, v in previous.items() if k != "fingerprint"}, payload
            )
            raise SystemExit(
                f"generation configuration changed since {manifest_path} was written:\n"
                + "\n".join(fields)
                + "\n  Existing outputs were produced under the previous configuration and resume "
                "matches on identity alone, so continuing here would mix two populations with "
                "counts that reconcile. Generate into a new tag instead."
            )
        return

    # A directory from before this check: its contents are unattributed, and saying so is the
    # most the manifest can honestly claim about them.
    if outputs_exist:
        print(
            f"[provenance] {manifest_path.parent} holds outputs but no manifest — they predate "
            "this check and cannot be attributed to a configuration. Recording the current one; "
            "regenerate into a fresh tag if their provenance matters."
        )
        current["preexisting_outputs_unattributed"] = True

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(current, indent=2, sort_keys=True))


__all__ = ["MANIFEST_NAME", "fingerprint", "guard"]
