#!/usr/bin/env bash
# wd1 RL on Countdown (cd3) with the SHARED encoder/decoder LoRA + DiffusionGemma's
# native chat template (GDSD format). Reproduces the strong countdown run (the one
# before the KL-anchor experiment): tie-lora + d1 hyperparameters on the full
# Jiayi-Pan cd3 split.
#
# Setup:
#   1) build data:  python dataset/build_countdown_data.py
#   2) train:       GPUS=6,7 bash run_countdown_tielora.sh
#   3) eval:        see the countdown_eval command at the bottom
#
# Key pieces (all in DiffGemma/diffgemma_trl):
#   --tie_encoder_decoder_lora true   share ONE LoRA across the tied enc<->dec weights
#                                     (fixes the two-adapter / merge issue, PEFT #1035)
#   --dataset_name countdown_gdsd     conversational prompt -> model's REAL chat template
#                                     (native <|channel>thought primer); train==eval
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD" TOKENIZERS_PARALLELISM=false
GPUS="${GPUS:-6,7}"
DATA="${DATA:-./dataset/countdown_cd3_train_full.jsonl}"   # or _dedup.jsonl for leakage-free
OUT="${OUT:-./xp_wd1_countdown_cd3_full_tielora}"

CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch --num_processes 2 --num_machines 1 \
  --mixed_precision bf16 --dynamo_backend no \
  -m DiffGemma.diffgemma_trl.train \
  --rl_loss_type wd1 --wd1_temperature 1.0 --tie_encoder_decoder_lora true \
  --model_name_or_path unsloth/diffusiongemma-26B-A4B-it \
  --use_peft true --lora_r 128 --lora_alpha 64 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --use_unsloth false --use_vllm false \
  --dataset_name countdown_gdsd --dataset_path "$DATA" \
  --output_dir "$OUT" \
  --max_steps 600 --logging_steps 1 --save_strategy steps --save_steps 100 \
  --report_to none \
  --gradient_checkpointing false \
  --num_generations 8 --per_device_train_batch_size 8 --gradient_accumulation_steps 1 \
  --generation_batch_size 16 --num_iterations 1 \
  --max_prompt_length 256 --max_completion_length 256 \
  --diffusion_steps 64 --diffusion_num_mc 1 --diffusion_self_conditioning_prob 0.0 \
  --beta 0.0 --learning_rate 3e-6 --max_grad_norm 0.2 --adam_beta2 0.99 --weight_decay 0.1 \
  --temperature 1.0

# Eval a checkpoint on the 256-puzzle cd3 test (native gdsd template, base model = ~62.5%):
#   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python -m DiffGemma.diffgemma_trl.countdown_eval \
#     --model_path google/diffusiongemma-26B-A4B-it \
#     --adapter_path "$OUT/checkpoint-300" --answer_format gdsd \
#     --test_file ./dataset/countdown_cd3_test.jsonl --max_denoising_steps 64 --seed 0
