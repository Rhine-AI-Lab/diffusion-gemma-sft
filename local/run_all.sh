#!/bin/bash
# Run all 6 countdown experiments locally on GPUs 2,3 (override with GPUS=...).
#   3 base-model evals + 3 wd1 training runs, one per prompt style.
# Each run already uses both GPUs (eval = sharded, train = data-parallel), so runs
# go sequentially. Evals run first (fast baselines), then the longer training runs.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
export GPUS="${GPUS:-2,3}"

STYLES=(countdown countdown_d2_train countdown_d2_eval)

echo "==================== BASE-MODEL EVALS ===================="
for ps in "${STYLES[@]}"; do
  echo "---- eval: $ps ----"
  PS="$ps" bash local/countdown_eval.sh
done

echo "==================== WD1 TRAINING ===================="
for ps in "${STYLES[@]}"; do
  echo "---- train: $ps ----"
  PS="$ps" bash local/wd1_countdown_train.sh
done

echo "==================== ALL 6 DONE ===================="
