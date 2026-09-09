#!/bin/bash
# Local (no SLURM / no container) countdown eval of ONE prompt style.
# Data-parallel across 2 local GPUs (default 2,3): each GPU evaluates a strided
# shard of the test set, then the per-shard results are aggregated.
# Base model by default; pass ADAPTER=... to eval a trained checkpoint.
#
#   PS=countdown          bash local/countdown_eval.sh
#   PS=countdown_d2_train bash local/countdown_eval.sh
#   PS=countdown_d2_eval  bash local/countdown_eval.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs eval_out

MODEL="${MODEL:-google/diffusiongemma-26B-A4B-it}"
PS="${PS:?set PS=countdown|countdown_d2_train|countdown_d2_eval}"
ADAPTER="${ADAPTER:-}"                       # empty => base model
TEST_FILE="${TEST_FILE:-./dataset/countdown_cd3_test.jsonl}"
BATCH="${BATCH:-16}"
LIMIT="${LIMIT:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_DENOISING_STEPS="${MAX_DENOISING_STEPS:-64}"
GPUS="${GPUS:-2,3}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NSHARDS="${#GPU_ARR[@]}"
TAG="${TAG:-countdown_base_${PS}}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export DIFFGEMMA_EXPERTS_IMPL="${DIFFGEMMA_EXPERTS_IMPL:-eager}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$REPO"

ADAPTER_ARG=""
if [[ -n "${ADAPTER}" ]]; then ADAPTER_ARG="--adapter_path ${ADAPTER}"; fi

echo "==== countdown eval prompt_style=${PS} (model=${ADAPTER:-BASE}) on GPUs ${GPUS} ===="
pids=()
for sid in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$sid]}"
  res="eval_out/${TAG}.shard${sid}.json"
  rm -f "$res"
  CUDA_VISIBLE_DEVICES="$gpu" \
  python -u -m DiffGemma.diffgemma_trl.countdown_eval \
    --prompt_style "${PS}" --model_path "${MODEL}" ${ADAPTER_ARG} \
    --test_file "${TEST_FILE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" --max_denoising_steps "${MAX_DENOISING_STEPS}" \
    --batch_size "${BATCH}" --limit "${LIMIT}" --seed 0 \
    --num_shards "${NSHARDS}" --shard_id "${sid}" --result_file "${res}" \
    > "logs/cdeval_${PS}.shard${sid}.log" 2>&1 &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [[ "$fail" -ne 0 ]]; then
  echo "[error] a shard process failed; see logs/cdeval_${PS}.shard*.log"; exit 1
fi

# Aggregate the per-shard result files into one accuracy.
python - "$TAG" "$NSHARDS" "${ADAPTER:-BASE}" <<'PY' | tee "eval_out/${TAG}.txt"
import json, sys
tag, nshards, model = sys.argv[1], int(sys.argv[2]), sys.argv[3]
correct = n = 0
for sid in range(nshards):
    with open(f"eval_out/{tag}.shard{sid}.json") as f:
        d = json.load(f)
    correct += d["correct"]; n += d["n"]
print(f"Countdown-cd3[{tag}] | model={model} | shards={nshards} | "
      f"score = {correct/n*100:.1f}%  ({correct:.1f}/{n})")
PY
