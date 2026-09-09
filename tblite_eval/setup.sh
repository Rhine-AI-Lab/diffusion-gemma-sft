#!/usr/bin/env bash
# One-time setup for the OpenThoughts-TBLite terminal-agent eval.
#   1. Installs the `harbor` CLI (https://github.com/laude-institute/harbor)
#      as an isolated uv tool, so it does not touch the project's
#      torch/transformers environment.
#   2. Clones the task dataset (https://github.com/open-thoughts/OpenThoughts-TBLite)
#      into tasks/. It is not published under a Harbor Hub dataset name, so we
#      point `harbor run --path` directly at the cloned directory instead of
#      `--dataset`. Harbor treats any directory of `<task>/task.toml`
#      subfolders as an implicit local dataset.
set -euo pipefail
cd "$(dirname "$0")"

if command -v uv >/dev/null 2>&1; then
  uv tool install harbor
else
  echo "ERROR: uv not found. Install uv first (https://docs.astral.sh/uv/)." >&2
  exit 127
fi

if [ ! -d tasks/.git ]; then
  git clone --depth 1 https://github.com/open-thoughts/OpenThoughts-TBLite tasks
else
  echo "tasks/ already cloned, skipping (git pull manually to update)."
fi

echo "=== harbor version ==="
harbor --version
echo "=== task count ==="
find tasks -mindepth 1 -maxdepth 1 -type d -exec test -e '{}/task.toml' ';' -print | wc -l
