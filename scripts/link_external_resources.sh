#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_PARENT="$(cd "${REPO_ROOT}/.." && pwd)"
DATA_SOURCE=""
MODELS_SOURCE=""
FORCE=false

usage() {
  cat <<EOF
Usage:
  bash scripts/link_external_resources.sh [--data-source <path>] [--models-source <path>] [--force]

Defaults (if paths omitted and found):
  data source:   ${DEFAULT_PARENT}/data
  models source: ${DEFAULT_PARENT}/shared_models

Links created:
  ${REPO_ROOT}/data/external -> <data-source>
  ${REPO_ROOT}/models/local  -> <models-source>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-source)
      DATA_SOURCE="$2"; shift 2 ;;
    --models-source)
      MODELS_SOURCE="$2"; shift 2 ;;
    --force)
      FORCE=true; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ -z "${DATA_SOURCE}" && -d "${DEFAULT_PARENT}/data" ]]; then
  DATA_SOURCE="${DEFAULT_PARENT}/data"
fi
if [[ -z "${MODELS_SOURCE}" && -d "${DEFAULT_PARENT}/shared_models" ]]; then
  MODELS_SOURCE="${DEFAULT_PARENT}/shared_models"
fi

if [[ -z "${DATA_SOURCE}" && -z "${MODELS_SOURCE}" ]]; then
  echo "Nothing to link. Pass --data-source and/or --models-source." >&2
  exit 1
fi

link_path() {
  local src="$1"
  local dst="$2"

  if [[ ! -e "${src}" ]]; then
    echo "Source does not exist: ${src}" >&2
    exit 1
  fi

  if [[ -L "${dst}" ]]; then
    rm -f "${dst}"
  elif [[ -e "${dst}" ]]; then
    if [[ "${FORCE}" == "true" ]]; then
      local backup="${dst}.bak.$(date +%Y%m%d%H%M%S)"
      mv "${dst}" "${backup}"
      echo "Moved existing path to ${backup}"
    else
      echo "Target exists and is not a symlink: ${dst}" >&2
      echo "Re-run with --force to back it up automatically." >&2
      exit 1
    fi
  fi

  ln -s "${src}" "${dst}"
  echo "Linked: ${dst} -> ${src}"
}

mkdir -p "${REPO_ROOT}/data" "${REPO_ROOT}/models"

if [[ -n "${DATA_SOURCE}" ]]; then
  link_path "$(realpath "${DATA_SOURCE}")" "${REPO_ROOT}/data/external"
fi

if [[ -n "${MODELS_SOURCE}" ]]; then
  link_path "$(realpath "${MODELS_SOURCE}")" "${REPO_ROOT}/models/local"
fi
