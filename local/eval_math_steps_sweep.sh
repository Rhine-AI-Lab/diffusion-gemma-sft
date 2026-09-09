#!/bin/bash
# Base-model MATH-500 eval at a fixed completion budget, swept over denoising steps.
# Default: max_new_tokens=256, max_denoising_steps in {128, 256}.
# Data-parallel across 2 local GPUs (default 6,7): each GPU evaluates a strided
# shard of MATH-500, then per-shard results are aggregated. Step-counts run
# sequentially (each fully occupies both GPUs); the 2 shards run concurrently.
#
#   bash local/eval_math_steps_sweep.sh
#   STEPS_LIST="64 128 256" GPUS="6,7" bash local/eval_math_steps_sweep.sh
#   STEPS_LIST=8 LIMIT=8 BATCH=4 bash local/eval_math_steps_sweep.sh   # smoke
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs eval_out

MODEL="${MODEL:-google/diffusiongemma-26B-A4B-it}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
STEPS_LIST="${STEPS_LIST:-128 256}"
BATCH="${BATCH:-16}"
LIMIT="${LIMIT:-0}"
GPUS="${GPUS:-6,7}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NSHARDS="${#GPU_ARR[@]}"

# MATH-500 comes from the hub (cached) -> stay online for datasets.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false
export DIFFGEMMA_EXPERTS_IMPL="${DIFFGEMMA_EXPERTS_IMPL:-eager}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$REPO"

for STEPS in $STEPS_LIST; do
  SUFFIX="base_${MAX_NEW_TOKENS}x${STEPS}"
  echo "==================== MATH-500 @ ${MAX_NEW_TOKENS}tok / ${STEPS} steps on GPUs ${GPUS} ===================="
  pids=()
  for sid in "${!GPU_ARR[@]}"; do
    gpu="${GPU_ARR[$sid]}"
    res="eval_out/math_${SUFFIX}.shard${sid}.json"
    rm -f "$res"
    CUDA_VISIBLE_DEVICES="$gpu" \
    python -u -m DiffGemma.diffgemma_trl.sft_eval_math \
      --model_path "${MODEL}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" --max_denoising_steps "${STEPS}" \
      --batch_size "${BATCH}" --limit "${LIMIT}" --seed 0 \
      --num_shards "${NSHARDS}" --shard_id "${sid}" --result_file "${res}" \
      > "logs/matheval_${SUFFIX}.shard${sid}.log" 2>&1 &
    pids+=($!)
  done

  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  if [[ "$fail" -ne 0 ]]; then
    echo "[error] a shard process failed; see logs/matheval_${SUFFIX}.shard*.log"; exit 1
  fi

  # Aggregate per-shard result files into one accuracy.
  python - "math_${SUFFIX}" "$NSHARDS" <<'PY' | tee "eval_out/math_${SUFFIX}.txt"
import json, sys
tag, nshards = sys.argv[1], int(sys.argv[2])
correct = n = 0
for sid in range(nshards):
    with open(f"eval_out/{tag}.shard{sid}.json") as f:
        d = json.load(f)
    correct += d["correct"]; n += d["n"]
print(f"MATH-500[{tag}] | model=BASE | shards={nshards} | "
      f"score = {correct/n*100:.1f}%  ({correct:.1f}/{n})")
PY
  echo
done

echo "==================== MATH STEP SWEEP DONE ===================="
for STEPS in $STEPS_LIST; do
  cat "eval_out/math_base_${MAX_NEW_TOKENS}x${STEPS}.txt"
done
