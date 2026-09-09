#!/bin/bash
# Debug-partition smoke test of the DiffusionGemma GRPO (PyTorch/TRL) path with
# the pluggable SFT probability term + optional encoder aux loss.
# Submit from the repo root:  sbatch cscs/grpo_smoke.sbatch
#
# Validates the container + venv end-to-end: loads the 26B model, generates a
# couple of completions per prompt on the committed 4x4 sudoku slice, scores
# them, and runs a few GRPO steps with --grpo_prob_term base-sft and a small
# encoder aux loss. Override knobs via --export, e.g.:
#   sbatch --export=ALL,PROB_TERM=reweighted-ce cscs/grpo_smoke.sbatch
#SBATCH --account=a0112
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=gdsdv2-grpo-smoke
#SBATCH --output=logs/grpo_smoke_%j.log

set -euo pipefail
cd "${SCRATCH}/projects/gdsdv2"
mkdir -p logs

MODEL="${MODEL:-google/diffusiongemma-26B-A4B-it}"
DATASET="${DATASET:-./dataset/4x4_test_sudoku.csv}"
PROB_TERM="${PROB_TERM:-base-sft}"          # corrupted-ce | base-sft | reweighted-ce | loo-ce
ENC_WEIGHT="${ENC_WEIGHT:-0.1}"             # 0 disables the encoder aux loss
ENC_TARGET="${ENC_TARGET:-completion}"      # completion | prompt | prompt_completion
WEIGHT_CLIP="${WEIGHT_CLIP:-}"             # set e.g. 20 for reweighted-ce
MAX_STEPS="${MAX_STEPS:-2}"
BATCH="${BATCH:-2}"                          # must be divisible by NUM_GEN
NUM_GEN="${NUM_GEN:-2}"
CANVAS="${CANVAS:-64}"
DIFF_STEPS="${DIFF_STEPS:-8}"               # denoising steps during generation (small = fast)
OUTDIR="${OUTDIR:-./xp_grpo_smoke}"

# Model is public + cached in HF_HOME; pass a token through if present.
TOKEN_FILE="${HF_TOKEN_FILE:-${SCRATCH}/.secrets/hf_token}"
if [[ -z "${HF_TOKEN:-}" && -f "${TOKEN_FILE}" ]]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "${TOKEN_FILE}")"
fi

CLIP_ARG=""
if [[ -n "${WEIGHT_CLIP}" ]]; then
  CLIP_ARG="--diffusion_weight_clip ${WEIGHT_CLIP}"
fi

LAUNCH="$(mktemp "${SCRATCH}/projects/gdsdv2/logs/grpo_smoke_launch.XXXXXX.sh")"
cat > "${LAUNCH}" <<EOF
set -euo pipefail
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${SCRATCH}/projects/gdsdv2" \
python -B -m DiffGemma.diffgemma_trl.train \
  --model_name_or_path ${MODEL} \
  --dataset_name sudoku --dataset_path ${DATASET} \
  --rl_loss_type grpo \
  --grpo_prob_term ${PROB_TERM} ${CLIP_ARG} \
  --encoder_loss_weight ${ENC_WEIGHT} --encoder_loss_target ${ENC_TARGET} \
  --diffusion_canvas_length ${CANVAS} --max_completion_length ${CANVAS} \
  --diffusion_steps ${DIFF_STEPS} \
  --diffusion_noise_min 0.0001 --diffusion_noise_max 0.9999 \
  --num_generations ${NUM_GEN} --beta 0.0 --num_iterations 1 \
  --use_unsloth false --use_peft true --lora_r 64 --lora_alpha 128 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --gradient_checkpointing false \
  --max_steps ${MAX_STEPS} --per_device_train_batch_size ${BATCH} --gradient_accumulation_steps 1 \
  --learning_rate 1.5e-4 --warmup_steps 0 --lr_scheduler_type constant \
  --max_grad_norm 1.0 --logging_steps 1 --save_strategy no \
  --report_to none --remove_unused_columns false --bf16 true --max_prompt_length 256 \
  --output_dir ${OUTDIR}
EOF

trap 'rm -f "${LAUNCH}"' EXIT
srun -ul --environment="${SCRATCH}/projects/gdsdv2/cscs/gdsdv2-pt.toml" bash "${LAUNCH}"
