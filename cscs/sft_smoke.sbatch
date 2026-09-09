#!/bin/bash
# Debug-partition smoke test of the DiffusionGemma SFT (PyTorch/TRL) env.
# Submit from the repo root:  sbatch cscs/sft_smoke.sbatch
#
# Validates the container + venv end-to-end: it loads the 26B model and runs a
# few SFT steps on the committed 64-row slice (dataset/sudoku_sft.jsonl) instead
# of the 2.7 GB full set. Override MODEL / MAX_STEPS / DATASET / BATCH via
# --export, e.g.:  sbatch --export=ALL,BATCH=2 cscs/sft_smoke.sbatch
#SBATCH --account=a0112
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=gdsdv2-sft-smoke
#SBATCH --output=logs/sft_smoke_%j.log

set -euo pipefail
cd "${SCRATCH}/projects/gdsdv2"
mkdir -p logs

MODEL="${MODEL:-google/diffusiongemma-26B-A4B-it}"
DATASET="${DATASET:-./dataset/sudoku_sft.jsonl}"
# Batch 2: the 26B model (~50GB weights) + the two self-conditioning forward
# passes fill the 96GB GH200 at batch 4; the class sets
# _supports_gradient_checkpointing=False, so activations can't be traded for
# compute. ~11GB/sample, so batch 2 leaves headroom. (See CSCS_SETUP.md for how
# to run the upstream batch-4 command: unsloth, multi-GPU FSDP, or grad-accum.)
MAX_STEPS="${MAX_STEPS:-20}"
BATCH="${BATCH:-2}"
OUTDIR="${OUTDIR:-./xp_sft_smoke}"

# Model is public + already cached in HF_HOME, so a token isn't required; pass it
# through if present (the /users path is unreachable in-container, so read from
# $SCRATCH). sbatch propagates exported env to srun.
TOKEN_FILE="${HF_TOKEN_FILE:-${SCRATCH}/.secrets/hf_token}"
if [[ -z "${HF_TOKEN:-}" && -f "${TOKEN_FILE}" ]]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "${TOKEN_FILE}")"
fi

# Build the in-container launcher (heredoc keeps the JSON kwarg quoting sane).
LAUNCH="$(mktemp "${SCRATCH}/projects/gdsdv2/logs/sft_smoke_launch.XXXXXX.sh")"
cat > "${LAUNCH}" <<EOF
set -euo pipefail
source .venv/bin/activate
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
  --learning_rate 1.5e-4 --warmup_steps 5 --lr_scheduler_type cosine \
  --max_grad_norm 1.0 --logging_steps 5 --save_strategy no \
  --report_to none --remove_unused_columns false --bf16 true --max_prompt_length 256 \
  --output_dir ${OUTDIR}
EOF

trap 'rm -f "${LAUNCH}"' EXIT
srun -ul --environment="${SCRATCH}/projects/gdsdv2/cscs/gdsdv2-pt.toml" bash "${LAUNCH}"
