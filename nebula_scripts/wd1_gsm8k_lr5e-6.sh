
# ---- DiffGemma RL parameters ----
DATASET="gsm8k_gdsd"
RL_LOSS_TYPE="wd1"
WORLD_SIZE=8

# Model (OSS or HF mirror)
MODEL_NAME_OR_PATH="/data/oss_bucket_0/models/google/diffusiongemma-26B-A4B-it"

# Dataset path (Countdown CD3 full training set on OSS)
DATASET_PATH="/data/oss_bucket_0/datasets/gsm8k/gsm8k_train.jsonl"

# ---- Training config ----
PER_DEVICE_TRAIN_BATCH_SIZE=6
GRADIENT_ACCUMULATION_STEPS=2
NUM_GENERATIONS=8
# Constraint: GENERATION_BATCH_SIZE >= batch * world * grad_accum
GENERATION_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * WORLD_SIZE * GRADIENT_ACCUMULATION_STEPS))
NUM_ITERATIONS=1
LEARNING_RATE=5e-6
MAX_PROMPT_LENGTH=400
MAX_COMPLETION_LENGTH=256
MAX_STEPS=7700

# ---- Save config ----
SAVE_STEPS=500
SAVE_TOTAL_LIMIT=12

# ---- DiffGemma-specific parameters ----
DIFFUSION_STEPS=64
DIFFUSION_NUM_MC=1
DIFFUSION_SELF_CONDITIONING_PROB=0.0

# ---- WD1-specific parameters ----
WD1_TEMPERATURE=1.0
BETA=0.0
TEMPERATURE=1.0
TIE_ENCODER_DECODER_LORA=true

# ---- LoRA config ----
LORA_R=128
LORA_ALPHA=64
LORA_TARGET="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"

# Output
RUN_NAME="${DATASET}_diffgemma_${RL_LOSS_TYPE}_tielora_r${LORA_R}_alpha${LORA_ALPHA}_mc${DIFFUSION_NUM_MC}_lr${LEARNING_RATE}_seed${SEED}"
OUTPUT_DIR="/data/oss_bucket_0/projects/DiffGemma/rl_checkpoints/${RUN_NAME}"

# Logger
MY_WANDB_KEY="xxx"
ENV_VARS="WANDB_API_KEY=$MY_WANDB_KEY"
ENV_VARS+=",WANDB_ENTITY=espo"
ENV_VARS+=",HF_ENDPOINT=https://hf-mirror.com"
ENV_VARS+=",HF_HOME=/data/oss_bucket_0/hf_cache"
ENV_VARS+=",HF_DATASETS_CACHE=/tmp/hf_datasets"
ENV_VARS+=",PYTHONPATH=./DiffGemma:."
ENV_VARS+=",CUBLAS_WORKSPACE_CONFIG=:4096:8"

# user_params: accelerate config + entry script + training args
args="--config_file recipes/accelerate_configs/zero2.yaml \
diffgemma_rl_train.py \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--dataset_name ${DATASET} \
--dataset_path ${DATASET_PATH} \
--rl_loss_type ${RL_LOSS_TYPE} \
--wd1_temperature ${WD1_TEMPERATURE} \
--tie_encoder_decoder_lora ${TIE_ENCODER_DECODER_LORA} \
--use_unsloth false \
--use_vllm false \
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
--diffusion_steps ${DIFFUSION_STEPS} \
--diffusion_num_mc ${DIFFUSION_NUM_MC} \
--diffusion_self_conditioning_prob ${DIFFUSION_SELF_CONDITIONING_PROB} \
--beta ${BETA} \
--temperature ${TEMPERATURE} \
--max_grad_norm 0.2 \
--adam_beta2 0.99 \
--weight_decay 0.1 \
--use_peft true \
--lora_r ${LORA_R} \
--lora_alpha ${LORA_ALPHA} \
--lora_target_modules ${LORA_TARGET} \
--bf16 true \
--gradient_checkpointing false \
--logging_steps 1 \
--save_strategy steps \
--save_steps ${SAVE_STEPS} \
--save_total_limit ${SAVE_TOTAL_LIMIT} \
--output_dir ${OUTPUT_DIR} \
--run_name ${RUN_NAME} \
--report_to wandb \
--wandb_project diffgemma_rl"
