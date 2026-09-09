#!/bin/bash
set -euo pipefail

# =============================================================================
# DiffusionGemma Sudoku Evaluation Script
#
# Evaluates DiffusionGemma LoRA checkpoints on 4x4 Sudoku test set.
# Supports sweeping over multiple checkpoints and generation lengths.
#
# Usage:
#   bash run_eval_diffgemma_sudoku.sh
#
# After evaluation, compute accuracy with:
#   python parse_and_get_acc.py --directory <output_dir>
# =============================================================================

# --- Configuration ---
MODEL_ROOT="/data/oss_bucket_0/diffgemma_gdsd"
BASE_MODEL_PATH="/data/oss_bucket_0/models/diffusiongemma-26B-A4B-it"

NUM_GPUS=8
GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))

# Checkpoint configuration
CHECKPOINT_STEPS=$(seq 500 100 2000)

# Experiment name (should match training run_name)
RUN_NAME="sudoku_diffgemma_gdsd_mu8_cl256_lr3e-6_kl1e-3_psi10.0_mc2"

# Eval parameters
TASKS=("sudoku")
GEN_LENGTHS=(128 256)
MAX_DENOISING_STEPS=64
TEMPERATURE=0.0
BATCH_SIZE=8
SUBSAMPLE=256

# --- Main Loop ---
for task in "${TASKS[@]}"; do
  for gen_length in "${GEN_LENGTHS[@]}"; do
    # Adjust batch size for longer generation
    if [[ "${gen_length}" -ge 256 ]]; then
      batch_size=4
    else
      batch_size=${BATCH_SIZE}
    fi

    run_dir="${MODEL_ROOT}/${RUN_NAME}"
    ckpt_dir="${run_dir}/checkpoints"
    logdir="${run_dir}/gl${gen_length}/eval_results"
    mkdir -p "${logdir}"

    echo "==== task=${task} gen_length=${gen_length} run=${RUN_NAME} ===="

    for ckpt_step in ${CHECKPOINT_STEPS}; do
      ckpt_path="${ckpt_dir}/checkpoint-${ckpt_step}"
      if [[ ! -d "${ckpt_path}" ]]; then
        echo "Missing checkpoint: ${ckpt_path} (skip)"
        continue
      fi

      # Check if result already exists
      existing_results=$(find "${logdir}/checkpoint-${ckpt_step}" -name "*_generations.json" 2>/dev/null | wc -l)
      if [[ "${existing_results}" -ge "${NUM_GPUS}" ]]; then
        echo "Results already exist for checkpoint-${ckpt_step}, skipping..."
        continue
      fi

      master_port=$(shuf -i 10000-20000 -n 1)
      output_subdir="${logdir}/checkpoint-${ckpt_step}"
      mkdir -p "${output_subdir}"

      echo "Evaluating: task=${task}, ckpt=${ckpt_step}, gen_length=${gen_length}, port=${master_port}"

      CUDA_VISIBLE_DEVICES="${GPU_LIST}" python -m torch.distributed.run \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port="${master_port}" \
        eval_diffgemma.py \
          --dataset "${task}" \
          --batch_size "${batch_size}" \
          --gen_length "${gen_length}" \
          --max_denoising_steps "${MAX_DENOISING_STEPS}" \
          --temperature "${TEMPERATURE}" \
          --model_path "${BASE_MODEL_PATH}" \
          --checkpoint_path "${ckpt_path}" \
          --output_dir "${output_subdir}" \
          --subsample "${SUBSAMPLE}" \
        || { echo "[WARN] eval failed: task=${task} ckpt=${ckpt_step}"; continue; }
    done

    # Parse results and compute accuracy
    echo "Computing accuracy for gen_length=${gen_length}..."
    python parse_and_get_acc.py --directory "${logdir}" || true
  done
done

echo "All Sudoku evaluations completed!"
