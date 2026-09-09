#!/bin/bash
# Wait for the in-flight wd1 run on GPUs 2,3 to finish, then run the two remaining
# prompt styles sequentially with the stabilized (LR 1e-6) config.
# Run #1 (countdown) is already running as xp_wd1_countdown_cd3_gdsd.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
export GPUS="${GPUS:-2,3}"

echo "[queue] waiting for the current training to finish (GPUs ${GPUS})..."
while pgrep -f "diffgemma_trl.train" >/dev/null 2>&1; do sleep 60; done
echo "[queue] GPUs free; starting remaining runs."

for ps in countdown_d2_train countdown_d2_eval; do
  echo "==================== wd1 train: ${ps} ===================="
  PS="$ps" bash local/wd1_countdown_train.sh
  sleep 10
done
echo "==================== remaining wd1 runs DONE ===================="
