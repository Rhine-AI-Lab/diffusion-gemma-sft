#!/bin/bash
# Base-model eval sweep at max_new_tokens=500, max_denoising_steps=128.
# Datasets: MATH-500, MBPP, HumanEval (Countdown excluded per request).
# Single GPU (default 7, exclusive) — GPUs 2,3 are in use by other users.
# Needs network for HF dataset downloads; MBPP/HumanEval execute generated code.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
mkdir -p eval_out logs

MODEL="${MODEL:-google/diffusiongemma-26B-A4B-it}"
GPU="${GPU:-7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-500}"
MAX_DENOISING_STEPS="${MAX_DENOISING_STEPS:-128}"
BATCH="${BATCH:-16}"
SUFFIX="${SUFFIX:-base_500x128}"

# Need the model from cache but datasets from the hub -> stay ONLINE for datasets.
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export DIFFGEMMA_EXPERTS_IMPL=eager
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"

for ds in math mbpp humaneval; do
  echo "==================== eval ${ds} @ ${MAX_NEW_TOKENS}tok / ${MAX_DENOISING_STEPS} steps (GPU ${GPU}) ===================="
  python -u -m "DiffGemma.diffgemma_trl.sft_eval_${ds}" \
    --model_path "${MODEL}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" --max_denoising_steps "${MAX_DENOISING_STEPS}" \
    --batch_size "${BATCH}" --seed 0 \
    --result_file "eval_out/${ds}_${SUFFIX}.json" \
    2>&1 | tee "eval_out/${ds}_${SUFFIX}.txt"
  echo
done
echo "==================== SWEEP DONE ===================="
grep -hE "score = |pass@1|MATH-500|MBPP|HumanEval" eval_out/{math,mbpp,humaneval}_${SUFFIX}.txt 2>/dev/null | grep "score ="
