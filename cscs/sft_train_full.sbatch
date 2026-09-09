#!/bin/bash
# Full DiffusionGemma SFT (PyTorch/TRL) run on CSCS Clariden — 4000 steps, LoRA,
# checkpoints every 1000 steps (-> checkpoint-1000/2000/3000/4000).
# Submit from the repo root:  sbatch cscs/sft_train_full.sbatch
#
# Config rationale (single GH200, see cscs/CSCS_SETUP.md "GH200-specific fixes"):
#   - eager MoE experts: grouped_mm asserts on GH200; batched_mm OOMs even at
#     batch 1 (replicates experts x tokens). eager fits and is ~8 s/step @ batch 1.
#   - batch 1: the 26B model is ~50GB and gradient checkpointing is unsupported, so
#     batch>1 risks OOM; eager @ batch 1 does 4000 steps in ~9h (< the 12h cap).
#
# Debug checkpoint-smoke (verifies intermediate saving in ~3 min) — overrides win:
#   sbatch -p debug -t 00:25:00 --export=ALL,MAX_STEPS=6,SAVE_STEPS=3,OUTDIR=./xp_sft_ckpt_smoke \
#     cscs/sft_train_full.sbatch
#SBATCH --account=a0112
#SBATCH --partition=normal
#SBATCH --time=11:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=gdsdv2-sft-full
#SBATCH --output=logs/sft_full_%j.log

set -euo pipefail
cd "${SCRATCH}/projects/gdsdv2"
mkdir -p logs

MODEL="${MODEL:-google/diffusiongemma-26B-A4B-it}"
DATASET="${DATASET:-./dataset/sudoku_train_full.jsonl}"
MAX_STEPS="${MAX_STEPS:-4000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
BATCH="${BATCH:-1}"
EXPERTS="${EXPERTS:-eager}"
OUTDIR="${OUTDIR:-./xp_sft_base_sft_trl}"

# Model is public + cached in HF_HOME; pass a token if present (read from $SCRATCH,
# since /users is not mounted in the container).
TOKEN_FILE="${HF_TOKEN_FILE:-${SCRATCH}/.secrets/hf_token}"
if [[ -z "${HF_TOKEN:-}" && -f "${TOKEN_FILE}" ]]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "${TOKEN_FILE}")"
fi

LAUNCH="$(mktemp "${SCRATCH}/projects/gdsdv2/logs/sft_full_launch.XXXXXX.sh")"
cat > "${LAUNCH}" <<EOF
set -euo pipefail
source .venv/bin/activate
export DIFFGEMMA_EXPERTS_IMPL=${EXPERTS}
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${SCRATCH}/projects/gdsdv2" \
python -m DiffGemma.diffgemma_trl.sft_train \
  --model_name_or_path ${MODEL} \
  --dataset_name ${DATASET} --dataset_path ${DATASET} \
  --sft_variant base-sft \
  --encoder_loss_weight 1.0 --decoder_loss_weight 1.0 \
  --diffusion_self_conditioning_prob 0.5 \
  --diffusion_noise_min 0.0001 --diffusion_noise_max 0.9999 \
  --use_unsloth false --use_peft true --lora_r 64 --lora_alpha 128 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --max_steps ${MAX_STEPS} --per_device_train_batch_size ${BATCH} --gradient_accumulation_steps 1 \
  --learning_rate 1.5e-4 --warmup_steps 100 --lr_scheduler_type cosine \
  --max_grad_norm 1.0 --logging_steps 20 --save_steps ${SAVE_STEPS} --save_strategy steps \
  --report_to none --remove_unused_columns false --bf16 true --max_prompt_length 256 \
  --output_dir ${OUTDIR}
EOF

trap 'rm -f "${LAUNCH}"' EXIT
echo "[full] $(date) host=$(hostname) model=${MODEL} steps=${MAX_STEPS} save_every=${SAVE_STEPS} batch=${BATCH} experts=${EXPERTS} out=${OUTDIR}"
srun -ul --environment="${SCRATCH}/projects/gdsdv2/cscs/gdsdv2-pt.toml" bash "${LAUNCH}"
echo "[full] done $(date); checkpoints:"; ls -d "${OUTDIR}"/checkpoint-* 2>/dev/null || true
