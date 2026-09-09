#!/bin/bash
# Run all 3 wd1 countdown training runs (one per prompt style) sequentially on
# GPUs 2,3 with the stabilized config (LR 1e-6), full 1000 steps each.
# Each run uses both GPUs (data-parallel), so they run one after another.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
export GPUS="${GPUS:-2,3}"
export MAX_STEPS="${MAX_STEPS:-1000}"

for ps in countdown countdown_d2_train countdown_d2_eval; do
  echo "==================== wd1 train: ${ps} (max_steps=${MAX_STEPS}) ===================="
  PS="$ps" bash local/wd1_countdown_train.sh
  sleep 10
done
echo "==================== ALL 3 wd1 runs DONE ===================="
