#!/usr/bin/env bash
set -euo pipefail

TASKS="${TASKS:-humaneval}"
MODEL_NAME="${MODEL_NAME:-llada_dist}"
DEVICE="${DEVICE:-cuda}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
GEN_LENGTHS="${GEN_LENGTHS:-256}"
STEPS="${STEPS:-128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output_lmeval/code}"

if [[ -n "${MODEL_PATHS:-}" ]]; then
  read -r -a PATHS <<< "${MODEL_PATHS}"
else
  PATHS=("${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-0}"

for model_path in "${PATHS[@]}"; do
  model_id="$(basename "${model_path}")"
  for gen_len in ${GEN_LENGTHS}; do
    current_output_path="${OUTPUT_ROOT}/${model_id}_len${gen_len}"
    mkdir -p "${current_output_path}"
    model_args="model_path=${model_path},gen_length=${gen_len},steps=${STEPS},block_length=${BLOCK_LENGTH}"

    accelerate launch "${SCRIPT_DIR}/eval_llada.py" \
      --tasks "${TASKS}" \
      --model "${MODEL_NAME}" \
      --device "${DEVICE}" \
      --model_args "${model_args}" \
      --output_path "${current_output_path}"
  done
done
