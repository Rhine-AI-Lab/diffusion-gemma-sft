#!/bin/bash
# Learning-rate sweep for block-SFT. For each LR in {1.5e-4, 1.5e-5, 1.5e-6}: train r1cot
# base-sft (block mode) for 500 steps on ONE GPU (runs in parallel), then merge the
# checkpoint into a full model and eval MATH-500 at g4096/c256. Reports LR x {MATH-500,
# canvas-bin fractions}. The bin fractions come from the trainer's canvas_bin_stats.json.
#
#   bash local/lr_sweep.sh
#   LRS="1.5e-5 1.5e-6" TRAIN_GPUS="1 2" bash local/lr_sweep.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
mkdir -p logs out eval_out

LRS="${LRS:-1.5e-4 1.5e-5 1.5e-6}"; read -r -a LR_ARR <<< "$LRS"
RESP="${RESP:-r1cot}"
STEPS="${STEPS:-500}"
SAVE_EVERY="${SAVE_EVERY:-250}"   # checkpoint interval (checkpoints at 250, 500, ...)
read -r -a GPU_ARR <<< "${TRAIN_GPUS:-1 2 3}"
BASE="google/diffusiongemma-26B-A4B-it"

# ---------------- Phase 1: train (parallel, single-GPU each) ----------------
echo "==== [lr_sweep] Phase 1: train ${#LR_ARR[@]} LR runs (${RESP} base-sft, ${STEPS} steps) ===="
pids=(); i=0
for lr in "${LR_ARR[@]}"; do
  gpu="${GPU_ARR[$i]}"; port=$((29600 + i))
  echo "  LR=${lr} -> GPU ${gpu} (port ${port})"
  LR="$lr" MAX_STEPS="$STEPS" SAVE_STEPS="$SAVE_EVERY" BLOCK=1 RESPONSES="$RESP" VARIANTS=base-sft \
    GPUS="$gpu" MAIN_PORT="$port" OUT_SUFFIX="_lr${lr}" REPORT_TO=none \
    bash local/math_sft_sweep.sh > "logs/lr_sweep_train_${RESP}_lr${lr}.log" 2>&1 & pids+=($!)
  i=$((i + 1))
done
tf=0; for p in "${pids[@]}"; do wait "$p" || tf=1; done
echo "==== [lr_sweep] Phase 1 done (fail=${tf}) ===="

# ---------------- Phase 2: merge + MATH-500 eval (parallel across GPUs) ----------------
echo "==== [lr_sweep] Phase 2: merge checkpoint-${STEPS} + eval MATH-500 ===="
mpids=(); i=0
for lr in "${LR_ARR[@]}"; do
  gpu="${GPU_ARR[$i]}"
  ckdir="out/openr1_math_sft_${RESP}_base-sft_block_lr${lr}/checkpoint-${STEPS}"
  merged="out/merged_lrsweep_${RESP}_lr${lr}"
  (
    if [[ ! -d "$ckdir" ]]; then echo "[lr_sweep] MISSING checkpoint: $ckdir"; exit 1; fi
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
      DIFFGEMMA_EXPERTS_IMPL=eager .venv/bin/python local/merge_lora.py "$BASE" "$ckdir" "$merged" \
      > "logs/lr_sweep_merge_lr${lr}.log" 2>&1
    DATASETS=math500 GPUS="$gpu" BASE_PORT=$((8070 + i)) \
      CKPTS="lrsweep_${RESP}_lr${lr}:${merged}" \
      bash local/eval_ckpt_g4096_c256.sh > "logs/lr_sweep_eval_lr${lr}.log" 2>&1
  ) & mpids+=($!)
  i=$((i + 1))
done
ef=0; for p in "${mpids[@]}"; do wait "$p" || ef=1; done
echo "==== [lr_sweep] Phase 2 done (fail=${ef}) ===="

# ---------------- Phase 3: report ----------------
echo
echo "================= LR SWEEP RESULTS (${RESP} base-sft, ${STEPS} steps, g4096/c256) ================="
.venv/bin/python - "$RESP" "$STEPS" "${LRS}" <<'PY'
import json, os, sys
resp, steps, lrs = sys.argv[1], sys.argv[2], sys.argv[3].split()
def score(p): return json.load(open(p))["score"] * 100 if os.path.exists(p) else None
base = score("eval_out/mathbench_math500_g4096_c256.json")
print(f"{'LR':<10} {'MATH-500':>9}   {'reason-only':>11} {'transition':>10} {'answer-only':>11}")
print(f"{'base':<10} {(f'{base:.1f}%' if base else '  ?  '):>9}   {'-':>11} {'-':>10} {'-':>11}")
for lr in lrs:
    m = score(f"eval_out/mathbench_lrsweep_{resp}_lr{lr}_math500_g4096_c256.json")
    sp = f"out/openr1_math_sft_{resp}_base-sft_block_lr{lr}/canvas_bin_stats.json"
    fr = json.load(open(sp))["fracs"] if os.path.exists(sp) else None
    ms = f"{m:.1f}%" if m is not None else "  -  "
    if fr:
        print(f"{lr:<10} {ms:>9}   {fr['reasoning_only']*100:>10.1f}% {fr['transition']*100:>9.1f}% {fr['answer_only']*100:>10.1f}%")
    else:
        print(f"{lr:<10} {ms:>9}   {'(no stats)':>11}")
PY
echo "================================================================================================"
