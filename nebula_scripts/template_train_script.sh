#!/bin/bash

# ---- DiffGemma RL parameters ----
DATASET="gsm8k_xml"
RL_LOSS_TYPE="gdsd"
WORLD_SIZE=8

# Model (OSS)
MODEL_NAME_OR_PATH="xxx"

# Training hyperparameters (aligned with llfree)
PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
NUM_GENERATIONS=$((WORLD_SIZE * 2))  # 16 completions per prompt
# Constraint: GENERATION_BATCH_SIZE >= batch * world * grad_accum
GENERATION_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * WORLD_SIZE * GRADIENT_ACCUMULATION_STEPS))  # 64
NUM_ITERATIONS=8
LEARNING_RATE=3e-6
MAX_PROMPT_LENGTH=400
MAX_COMPLETION_LENGTH=256
BETA_KL=1e-3
MAX_GRAD_NORM=0.2
WARMUP_RATIO=0.001
WEIGHT_DECAY=0.01
MAX_STEPS=2000

# ---- Checkpoint & evaluation (test at every save step) ----
SAVE_STEPS=200
EVAL_STEPS=${SAVE_STEPS}
# 0 = use full GSM8K test set (1319 samples); set >0 to cap
EVAL_DATASET_SIZE=0

# DiffGemma-specific parameters
DIFFUSION_STEPS=256
DIFFUSION_CANVAS_LENGTH=256
DIFFUSION_NUM_MC=4
DIFFUSION_NOISE_MIN=0.1
DIFFUSION_NOISE_MAX=1.0
PSI=1.0

# LoRA (via PEFT, since Unsloth disabled on ROCm)
LORA_R=64
LORA_ALPHA=128
LORA_TARGET="all-linear"

# user_params: accelerate config + entry script + training args
args="--config_file recipes/accelerate_configs/zero2.yaml \
diffgemma_rl_train.py \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--dataset_name ${DATASET} \
--rl_loss_type ${RL_LOSS_TYPE} \
--use_unsloth false \
--seed ${SEED} \
--per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
--gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
--num_generations ${NUM_GENERATIONS} \
--generation_batch_size ${GENERATION_BATCH_SIZE} \
--num_iterations ${NUM_ITERATIONS} \
--learning_rate ${LEARNING_RATE} \
--max_steps ${MAX_STEPS} \
--max_prompt_length ${MAX_PROMPT_LENGTH} \
--max_completion_length ${MAX_COMPLETION_LENGTH} \
--max_grad_norm ${MAX_GRAD_NORM} \
--warmup_ratio ${WARMUP_RATIO} \
--weight_decay ${WEIGHT_DECAY} \
--lr_scheduler_type constant_with_warmup \
--beta ${BETA_KL} \
--diffusion_steps ${DIFFUSION_STEPS} \
--diffusion_canvas_length ${DIFFUSION_CANVAS_LENGTH} \
--diffusion_num_mc ${DIFFUSION_NUM_MC} \
--diffusion_noise_min ${DIFFUSION_NOISE_MIN} \
--diffusion_noise_max ${DIFFUSION_NOISE_MAX} \
--psi ${PSI} \
--use_peft true \
--lora_r ${LORA_R} \
--lora_alpha ${LORA_ALPHA} \
--lora_target_modules ${LORA_TARGET} \
--bf16 true \
--gradient_checkpointing false \
--logging_steps 1 \
--save_steps ${SAVE_STEPS} \
--save_total_limit 5 \
--eval_strategy steps \
--eval_steps ${EVAL_STEPS} \
--eval_dataset_size ${EVAL_DATASET_SIZE} \
--output_dir ${OUTPUT_DIR} \
--run_name ${RUN_NAME} \
--report_to wandb \
--wandb_project diffgemma_rl"
