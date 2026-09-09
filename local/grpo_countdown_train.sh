#!/bin/bash
# Local (no SLURM / no container) GRPO RL training on countdown for ONE prompt style.
# Data-parallel across 2 local GPUs via `accelerate launch`. Mirrors the stabilized
# wd1 launcher but uses --rl_loss_type grpo (clipped policy objective).
#
#   PS=countdown GPUS=1,2 bash local/grpo_countdown_train.sh
#
# GRPO notes: num_iterations=1 -> old logps = current.detach() (group-baseline PG with
# clip); epsilon defaults 0.2; beta 0.0 (no KL). LR 1e-6 default (wd1 collapsed at 1e-4).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

MODEL="${MODEL:-google/diffusiongemma-26B-A4B-it}"
PS="${PS:?set PS=countdown|countdown_d2_train|countdown_d2_eval}"
DATASET="${DATASET:-./dataset/countdown_cd3_train.jsonl}"
OUTDIR="${OUTDIR:-./xp_grpo_countdown_cd3_${PS}}"
RUN_NAME="${RUN_NAME:-grpo_countdown_cd3_${PS}}"
GPUS="${GPUS:-1,2}"
NPROC="${NPROC:-$(awk -F, '{print NF}' <<<"$GPUS")}"
MAX_STEPS="${MAX_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
NUM_GEN="${NUM_GEN:-8}"
BATCH="${BATCH:-8}"
GEN_BATCH="${GEN_BATCH:-16}"
CANVAS="${CANVAS:-256}"
DIFF_STEPS="${DIFF_STEPS:-64}"
LR="${LR:-1e-6}"
PROB_TERM="${PROB_TERM:-corrupted-ce}"   # GRPO probability term feeding the ratio
EPSILON="${EPSILON:-0.2}"                # clip range
BETA="${BETA:-0.0}"                      # KL coeff (0 = off)
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-diffuGemma-pt}"
REPORT_TO="${REPORT_TO:-wandb}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export DIFFGEMMA_EXPERTS_IMPL="${DIFFGEMMA_EXPERTS_IMPL:-eager}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="$GPUS"

RESUME_ARG=""
LATEST="$(ls -d "${OUTDIR}"/checkpoint-* 2>/dev/null | sed 's#.*/checkpoint-##' | sort -n | tail -1 || true)"
if [[ -n "${LATEST}" ]]; then
  RESUME_ARG="--resume_from_checkpoint ${OUTDIR}/checkpoint-${LATEST}"
  echo "[resume] from ${OUTDIR}/checkpoint-${LATEST}"
fi

echo "[grpo] PS=${PS} GPUS=${GPUS} (nproc=${NPROC}) steps=${MAX_STEPS} lr=${LR} prob_term=${PROB_TERM} -> ${OUTDIR}"
set -x
accelerate launch --num_processes "${NPROC}" --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
  -m DiffGemma.diffgemma_trl.train \
  --rl_loss_type grpo --grpo_prob_term "${PROB_TERM}" --epsilon "${EPSILON}" \
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
  --beta "${BETA}" --learning_rate "${LR}" --temperature 1.0 \
  2>&1 | tee "logs/${RUN_NAME}.log"
