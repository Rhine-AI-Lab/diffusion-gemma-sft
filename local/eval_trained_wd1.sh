#!/bin/bash
# Wait for the wd1 training queue to finish, then eval each trained checkpoint-1000
# with its MATCHING prompt style (train/eval parity), sharded across GPUs 2,3.
# Results -> eval_out/countdown_trained_<ps>.txt (compare vs countdown_base_<ps>.txt).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
export GPUS="${GPUS:-2,3}"

echo "[eval-queue] waiting for training queue (run_all_wd1.sh) to finish..."
while pgrep -f "run_all_wd1.sh" >/dev/null 2>&1; do sleep 120; done
echo "[eval-queue] queue script done; waiting for any train procs to clear..."
while pgrep -f "diffgemma_trl.train" >/dev/null 2>&1; do sleep 60; done
echo "[eval-queue] training complete; starting trained-checkpoint evals."

for ps in countdown countdown_d2_train countdown_d2_eval; do
  CKPT="./xp_wd1_countdown_cd3_${ps}/checkpoint-1000"
  if [ ! -d "$CKPT" ]; then
    echo "[warn] missing ${CKPT}; skipping ${ps}"
    continue
  fi
  echo "==================== eval TRAINED: ${ps}  (${CKPT}) ===================="
  PS="$ps" ADAPTER="$CKPT" TAG="countdown_trained_${ps}" bash local/countdown_eval.sh
  sleep 5
done
echo "==================== trained-checkpoint evals DONE ===================="
