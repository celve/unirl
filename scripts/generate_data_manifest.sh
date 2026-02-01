#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/generate_data_manifest.sh <file_or_dir> [...]" >&2
  exit 1
fi

for path in "$@"; do
  if [[ -f "${path}" ]]; then
    sha256sum "${path}"
  elif [[ -d "${path}" ]]; then
    find "${path}" -type f | sort | while read -r f; do
      sha256sum "${f}"
    done
  else
    echo "Skip missing path: ${path}" >&2
  fi
done
