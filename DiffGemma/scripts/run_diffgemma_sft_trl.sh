#!/usr/bin/env bash
set -euo pipefail

# DiffusionGemma SFT via the TRL/transformers Trainer (PyTorch path).
# Variants: base-sft | reweighted-ce | loo-ce | reweighted-loo-ce
#
# Example:
#   MODEL=google/diffusiongemma-26B-A4B-it DATASET=./dataset/sudoku_sft.jsonl \
#   VARIANT=reweighted-ce OUTDIR=./xp_sft_reweighted_ce_trl \
#   bash DiffGemma/scripts/run_diffgemma_sft_trl.sh

MODEL="${MODEL:-google/diffusiongemma-26B-A4B-it}"
DATASET="${DATASET:?set DATASET to an HF id or local dataset path}"
VARIANT="${VARIANT:-base-sft}"
OUTDIR="${OUTDIR:-./xp_sft_${VARIANT}_trl}"

LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
USE_UNSLOTH="${USE_UNSLOTH:-false}"
ENCODER_LOSS_WEIGHT="${ENCODER_LOSS_WEIGHT:-1.0}"   # 0.0 for decoder-only (Unsloth-style)
SELF_COND_PROB="${SELF_COND_PROB:-0.5}"
NOISE_MIN="${NOISE_MIN:-0.001}"
NOISE_MAX="${NOISE_MAX:-0.999}"
BATCH="${BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-1.5e-4}"
MAX_STEPS="${MAX_STEPS:-4000}"
PROMPT_COL="${PROMPT_COL:-prompt}"
COMPLETION_COL="${COMPLETION_COL:-completion}"

accelerate launch -m DiffGemma.diffgemma_trl.sft_train \
  --model_name_or_path "${MODEL}" \
  --dataset_name "${DATASET}" \
  --dataset_path "${DATASET}" \
  --prompt_column "${PROMPT_COL}" \
  --completion_column "${COMPLETION_COL}" \
  --sft_variant "${VARIANT}" \
  --encoder_loss_weight "${ENCODER_LOSS_WEIGHT}" \
  --decoder_loss_weight 1.0 \
  --diffusion_self_conditioning_prob "${SELF_COND_PROB}" \
  --diffusion_noise_min "${NOISE_MIN}" \
  --diffusion_noise_max "${NOISE_MAX}" \
  --use_unsloth "${USE_UNSLOTH}" \
  --use_peft true \
  --lora_r "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --unsloth_lora_rank "${LORA_RANK}" \
  --unsloth_lora_alpha "${LORA_ALPHA}" \
  --per_device_train_batch_size "${BATCH}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate "${LR}" \
  --max_steps "${MAX_STEPS}" \
  --warmup_steps 100 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --save_steps 1000 \
  --save_strategy steps \
  --report_to none \
  --remove_unused_columns false \
  --bf16 true \
  --output_dir "${OUTDIR}"
