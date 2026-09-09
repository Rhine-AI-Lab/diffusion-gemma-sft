#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${1:-1}"
DATASET="${DATASET:-countdown}"
MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/${DATASET}}"
GEN_LENGTH="${GEN_LENGTH:-128}"
MASTER_PORT="${MASTER_PORT:-33322}"

mkdir -p "${OUTPUT_DIR}"

ARGS=(
  --dataset "${DATASET}"
  --batch_size 1
  --gen_length "${GEN_LENGTH}"
  --output_dir "${OUTPUT_DIR}"
  --model_path "${MODEL_PATH}"
)

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  ARGS+=(--checkpoint_path "${CHECKPOINT_PATH}")
fi

torchrun \
  --nproc_per_node "${NUM_GPUS}" \
  --master_port "${MASTER_PORT}" \
  eval.py \
  "${ARGS[@]}"

python parse_and_get_acc.py --directory "${OUTPUT_DIR}"
