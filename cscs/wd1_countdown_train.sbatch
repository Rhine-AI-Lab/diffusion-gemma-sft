#!/bin/bash
# wd1 RL training on countdown for ONE prompt style (normal partition, 12h).
# Resumable + self-chaining: each link resumes from the latest checkpoint and submits
# a dependent successor (afterany) so the run completes across multiple 12h windows;
# the chain stops once checkpoint-${MAX_STEPS} exists (or CHAIN guard is hit).
#
# Submit one job per prompt style, e.g.:
#   sbatch --job-name=wd1-countdown          --export=ALL,PS=countdown          cscs/wd1_countdown_train.sbatch
#   sbatch --job-name=wd1-countdown_d2_train --export=ALL,PS=countdown_d2_train cscs/wd1_countdown_train.sbatch
#   sbatch --job-name=wd1-countdown_d2_eval  --export=ALL,PS=countdown_d2_eval  cscs/wd1_countdown_train.sbatch
#SBATCH --account=a0112
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=gdsdv2-wd1-countdown
#SBATCH --output=logs/wd1_%x_%j.log

set -euo pipefail
cd "${SCRATCH}/projects/gdsdv2"
mkdir -p logs

MODEL="${MODEL:-google/diffusiongemma-26B-A4B-it}"
PS="${PS:?set PS=countdown|countdown_d2_train|countdown_d2_eval}"
DATASET="${DATASET:-./dataset/countdown_cd3_train.jsonl}"
OUTDIR="${OUTDIR:-./xp_wd1_countdown_cd3_${PS}}"
MAX_STEPS="${MAX_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-100}"
NUM_GEN="${NUM_GEN:-8}"
BATCH="${BATCH:-8}"                  # per_device_train_batch_size; must be divisible by NUM_GEN
CANVAS="${CANVAS:-256}"
DIFF_STEPS="${DIFF_STEPS:-64}"
LR="${LR:-1e-4}"
CHAIN="${CHAIN:-1}"; MAX_CHAIN="${MAX_CHAIN:-5}"

TOKEN_FILE="${HF_TOKEN_FILE:-${SCRATCH}/.secrets/hf_token}"
if [[ -z "${HF_TOKEN:-}" && -f "${TOKEN_FILE}" ]]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "${TOKEN_FILE}")"
fi

# Already finished? then stop the chain.
if [[ -d "${OUTDIR}/checkpoint-${MAX_STEPS}" ]]; then
  echo "[done] ${OUTDIR}/checkpoint-${MAX_STEPS} exists; nothing to do."
  exit 0
fi

# Submit the successor NOW (runs after this job ends for ANY reason, incl. timeout),
# so a 12h wall-clock limit doesn't break the chain. Successor re-checks done-ness.
if [[ "${CHAIN}" -lt "${MAX_CHAIN}" ]]; then
  sbatch --dependency="afterany:${SLURM_JOB_ID}" --job-name="${SLURM_JOB_NAME}" \
    --export=ALL,PS="${PS}",OUTDIR="${OUTDIR}",MAX_STEPS="${MAX_STEPS}",SAVE_STEPS="${SAVE_STEPS}",NUM_GEN="${NUM_GEN}",BATCH="${BATCH}",CANVAS="${CANVAS}",DIFF_STEPS="${DIFF_STEPS}",LR="${LR}",CHAIN="$((CHAIN + 1))",MAX_CHAIN="${MAX_CHAIN}" \
    cscs/wd1_countdown_train.sbatch || echo "[warn] successor submit failed"
fi

# Resume from the latest checkpoint, if any.
RESUME_ARG=""
LATEST="$(ls -d "${OUTDIR}"/checkpoint-* 2>/dev/null | sed 's#.*/checkpoint-##' | sort -n | tail -1 || true)"
if [[ -n "${LATEST}" ]]; then
  RESUME_ARG="--resume_from_checkpoint ${OUTDIR}/checkpoint-${LATEST}"
  echo "[resume] from ${OUTDIR}/checkpoint-${LATEST}"
fi

LAUNCH="$(mktemp "${SCRATCH}/projects/gdsdv2/logs/wd1_launch.XXXXXX.sh")"
cat > "${LAUNCH}" <<EOF
set -euo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${SCRATCH}/projects/gdsdv2" \
python -B -u -m DiffGemma.diffgemma_trl.train \
  --model_name_or_path ${MODEL} \
  --dataset_name countdown --dataset_path ${DATASET} \
  --countdown_prompt_style ${PS} \
  --rl_loss_type wd1 --wd1_temperature 1.0 --beta 0.0 --num_iterations 1 \
  --num_generations ${NUM_GEN} \
  --diffusion_canvas_length ${CANVAS} --max_completion_length ${CANVAS} \
  --diffusion_steps ${DIFF_STEPS} \
  --diffusion_noise_min 0.0001 --diffusion_noise_max 0.9999 \
  --use_unsloth false --use_peft true --lora_r 64 --lora_alpha 128 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --gradient_checkpointing false \
  --max_steps ${MAX_STEPS} --per_device_train_batch_size ${BATCH} --gradient_accumulation_steps 1 \
  --learning_rate ${LR} --warmup_steps 0 --lr_scheduler_type constant --max_grad_norm 1.0 \
  --logging_steps 1 --save_strategy steps --save_steps ${SAVE_STEPS} \
  --report_to none --remove_unused_columns false --bf16 true --max_prompt_length 256 \
  --output_dir ${OUTDIR} ${RESUME_ARG}
EOF

trap 'rm -f "${LAUNCH}"' EXIT
srun -ul --environment="${SCRATCH}/projects/gdsdv2/cscs/gdsdv2-pt.toml" bash "${LAUNCH}"
