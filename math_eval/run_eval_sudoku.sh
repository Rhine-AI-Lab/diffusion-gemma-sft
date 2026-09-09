#!/bin/bash
set -euo pipefail

MODEL_ROOT="/data/oss_bucket_0/spg/nebula_308x"

NUM_GPUS=8
GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))

declare -A CHECKPOINTS
CHECKPOINTS["sudoku"]=$(seq 2200 -200 1400)
# CHECKPOINTS["countdown"]="9200"

TASKS=("sudoku")
TRAIN_LENGTHS=(256)
GEN_LENGTHS=(128 256 512)
PSI_LIST=(10.0)
MODEL="llada"

for task in "${TASKS[@]}"; do
  ckpt_list="${CHECKPOINTS[$task]:-}"
  if [[ -z "${ckpt_list}" ]]; then
    echo "No checkpoints configured for task=${task}, skip."
    continue
  fi

  for train_length in "${TRAIN_LENGTHS[@]}"; do
    for psi in "${PSI_LIST[@]}"; do
      for gen_length in "${GEN_LENGTHS[@]}"; do
        run_name="sudoku_base_spg_mix_beta1.0_weight0.5_iter8"
        run_dir="${MODEL_ROOT}/${run_name}"
        ckpt_dir="${run_dir}/checkpoints"
        logdir="${run_dir}/gl${gen_length}/eval_results"
        mkdir -p "${logdir}"

        # batch size logic
        if [[ "${gen_length}" -eq 512 ]]; then
          batch_size=64
        else
          batch_size=128
        fi

        echo "==== task=${task} gen_length=${gen_length} run=${run_name} ===="

        for ckpt_step in ${ckpt_list}; do
          ckpt_path="${ckpt_dir}/checkpoint-${ckpt_step}"
          if [[ ! -d "${ckpt_path}" ]]; then
            echo "Missing checkpoint: ${ckpt_path} (skip)"
            continue
          fi

          master_port=$(shuf -i 10000-20000 -n 1)

          echo "Evaluating: task=${task}, ckpt=${ckpt_step}, port=${master_port}"

          CUDA_VISIBLE_DEVICES="${GPU_LIST}" python -m torch.distributed.run \
            --nproc_per_node="${NUM_GPUS}" \
            --master_port="${master_port}" \
            eval.py \
              --dataset "${task}" \
              --batch_size "${batch_size}" \
              --gen_length "${gen_length}" \
              --output_dir "${logdir}/checkpoint-${ckpt_step}" \
              --model_path "${ckpt_path}" \
            || { echo "[WARN] eval failed: task=${task} ckpt=${ckpt_step}"; continue; }
        done

        python eval/parse_and_get_acc.py --directory "${logdir}"
      done
    done
  done
done

echo "All evaluations completed!"
