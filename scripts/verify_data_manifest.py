#!/usr/bin/env python
"""Verify SHA256 checksums listed in a manifest file."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_manifest(path: Path):
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid manifest line: {line}")
        yield parts[0], parts[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify data manifest checksums")
    parser.add_argument("--manifest", required=True, help="Path to .sha256 manifest file")
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[1],
        help="Base directory for relative file paths in manifest",
    )
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    repo_root = Path(args.repo_root).resolve()

    failed = False
    for expected, rel in parse_manifest(manifest):
        target = (repo_root / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
        if not target.exists():
            print(f"MISSING  {rel}")
            failed = True
            continue
        actual = file_sha256(target)
        if actual != expected:
            print(f"MISMATCH {rel}")
            failed = True
        else:
            print(f"OK      {rel}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
