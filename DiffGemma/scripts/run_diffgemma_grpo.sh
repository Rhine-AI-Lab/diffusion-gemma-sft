#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIFFGEMMA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${DIFFGEMMA_DIR}:${PYTHONPATH:-}"

python -m diffgemma_trl.train "$@"
