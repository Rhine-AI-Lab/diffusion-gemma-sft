#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}}"
NUM_GPUS="${2:-${NUM_GPUS:-1}}"
TASKS="${TASKS:-gsm8k math}"
GEN_LENGTHS="${GEN_LENGTHS:-512}"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval_results}"
GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))

for task in ${TASKS}; do
  for gen_length in ${GEN_LENGTHS}; do
    if [[ "${gen_length}" -eq 512 ]]; then
      batch_size=8
    else
      batch_size=16
    fi

    master_port=$((12000 + RANDOM % 20000))
    output_dir="${OUTPUT_ROOT}/${task}_gl${gen_length}"
    mkdir -p "${output_dir}"

    CUDA_VISIBLE_DEVICES="${GPU_LIST}" python -m torch.distributed.run \
      --nproc_per_node="${NUM_GPUS}" \
      --master_port="${master_port}" \
      eval.py \
      --dataset "${task}" \
      --batch_size "${batch_size}" \
      --gen_length "${gen_length}" \
      --output_dir "${output_dir}" \
      --model_path "${MODEL_PATH}"
  done
done

python eval/parse_and_get_acc.py --directory "${OUTPUT_ROOT}"
