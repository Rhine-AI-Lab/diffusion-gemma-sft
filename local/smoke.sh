#!/bin/bash
# Tiny end-to-end smoke test on GPUs 2,3 (override with GPUS=...).
# 1) 2-step wd1 train (2-GPU, small canvas/diffusion) ; 2) limit-8 sharded eval.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
export GPUS="${GPUS:-2,3}"

echo "######## SMOKE 1/2: wd1 train (2 steps) ########"
PS=countdown MAX_STEPS=2 CANVAS=64 DIFF_STEPS=8 NUM_GEN=2 BATCH=2 \
  SAVE_STEPS=1000 SAVE_TOTAL_LIMIT=1 OUTDIR=./xp_smoke_wd1 \
  bash local/wd1_countdown_train.sh

echo "######## SMOKE 2/2: countdown eval (limit 8) ########"
PS=countdown LIMIT=8 BATCH=4 MAX_NEW_TOKENS=64 MAX_DENOISING_STEPS=8 TAG=smoke_eval \
  bash local/countdown_eval.sh

echo "######## SMOKE DONE ########"
