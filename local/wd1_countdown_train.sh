#!/bin/bash
# Local (no SLURM / no container) wd1 RL training on countdown for ONE prompt style.
# Data-parallel across 2 local GPUs (default 2,3) via `accelerate launch`.
#
# STABILIZED config: LR 1e-6 (the 1e-4 default collapsed the model to empty
# completions by ~step 27); generation_batch_size 16, 1 MC sample, no self-cond.
#
#   PS=countdown          bash local/wd1_countdown_train.sh
#   PS=countdown_d2_train bash local/wd1_countdown_train.sh
#   PS=countdown_d2_eval  bash local/wd1_countdown_train.sh
#
# Override knobs via env, e.g.  GPUS=4,5 MAX_STEPS=1000 LR=3e-6 bash local/wd1_countdown_train.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

MODEL="${MODEL:-google/diffusiongemma-26B-A4B-it}"
PS="${PS:?set PS=countdown|countdown_d2_train|countdown_d2_eval}"
DATASET="${DATASET:-./dataset/countdown_cd3_train.jsonl}"
OUTDIR="${OUTDIR:-./xp_wd1_countdown_cd3_${PS}}"
RUN_NAME="${RUN_NAME:-wd1_countdown_cd3_${PS}}"
GPUS="${GPUS:-2,3}"
NPROC="${NPROC:-$(awk -F, '{print NF}' <<<"$GPUS")}"   # one process per visible GPU
MAX_STEPS="${MAX_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"   # disk: keep only the most recent checkpoint
NUM_GEN="${NUM_GEN:-8}"
BATCH="${BATCH:-8}"                          # per_device_train_batch_size; divisible by NUM_GEN
GEN_BATCH="${GEN_BATCH:-16}"
CANVAS="${CANVAS:-256}"
DIFF_STEPS="${DIFF_STEPS:-64}"
LR="${LR:-1e-6}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-diffuGemma-pt}"
REPORT_TO="${REPORT_TO:-wandb}"

# Model is cached locally; stay offline (datasets are local jsonl too).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export DIFFGEMMA_EXPERTS_IMPL="${DIFFGEMMA_EXPERTS_IMPL:-eager}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="$GPUS"

# Resume from latest checkpoint, if any.
RESUME_ARG=""
LATEST="$(ls -d "${OUTDIR}"/checkpoint-* 2>/dev/null | sed 's#.*/checkpoint-##' | sort -n | tail -1 || true)"
if [[ -n "${LATEST}" ]]; then
  RESUME_ARG="--resume_from_checkpoint ${OUTDIR}/checkpoint-${LATEST}"
  echo "[resume] from ${OUTDIR}/checkpoint-${LATEST}"
fi

echo "[wd1] PS=${PS} GPUS=${GPUS} (nproc=${NPROC}) steps=${MAX_STEPS} lr=${LR} -> ${OUTDIR}"
set -x
accelerate launch --num_processes "${NPROC}" --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
  -m DiffGemma.diffgemma_trl.train \
  --rl_loss_type wd1 --wd1_temperature 1.0 \
  --model_name_or_path "${MODEL}" \
  --use_peft true --lora_r 64 --lora_alpha 128 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --use_unsloth false --use_vllm false \
  --dataset_name countdown --dataset_path "${DATASET}" --countdown_prompt_style "${PS}" \
  --output_dir "${OUTDIR}" ${RESUME_ARG} \
  --max_steps "${MAX_STEPS}" --logging_steps 1 \
  --save_strategy steps --save_steps "${SAVE_STEPS}" --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --report_to "${REPORT_TO}" --wandb_project "${WANDB_PROJECT_NAME}" --run_name "${RUN_NAME}" \
  --gradient_checkpointing false \
  --num_generations "${NUM_GEN}" --per_device_train_batch_size "${BATCH}" --gradient_accumulation_steps 1 \
  --generation_batch_size "${GEN_BATCH}" --num_iterations 1 \
  --max_prompt_length 256 --max_completion_length "${CANVAS}" \
  --diffusion_steps "${DIFF_STEPS}" --diffusion_num_mc 1 --diffusion_self_conditioning_prob 0.0 \
  --beta 0.0 --learning_rate "${LR}" --temperature 1.0 \
  2>&1 | tee "logs/${RUN_NAME}.log"
