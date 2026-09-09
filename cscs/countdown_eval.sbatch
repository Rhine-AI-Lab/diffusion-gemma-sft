#!/bin/bash
# Countdown eval of ONE prompt style on the debug partition (local diffusion generate()).
# Base model by default (no adapter); pass ADAPTER=... to eval a checkpoint later.
# Submit one job per prompt style, e.g.:
#   sbatch --job-name=cdeval-countdown          --export=ALL,PS=countdown          cscs/countdown_eval.sbatch
#   sbatch --job-name=cdeval-countdown_d2_train --export=ALL,PS=countdown_d2_train cscs/countdown_eval.sbatch
#   sbatch --job-name=cdeval-countdown_d2_eval  --export=ALL,PS=countdown_d2_eval  cscs/countdown_eval.sbatch
#SBATCH --account=a0112
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=gdsdv2-cdeval
#SBATCH --output=logs/cdeval_%x_%j.log

set -euo pipefail
cd "${SCRATCH}/projects/gdsdv2"
mkdir -p logs eval_out

MODEL="${MODEL:-google/diffusiongemma-26B-A4B-it}"
PS="${PS:?set PS=countdown|countdown_d2_train|countdown_d2_eval}"
ADAPTER="${ADAPTER:-}"                  # empty => base model
TEST_FILE="${TEST_FILE:-./dataset/countdown_cd3_test.jsonl}"
BATCH="${BATCH:-16}"
LIMIT="${LIMIT:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"      # canvas 256
MAX_DENOISING_STEPS="${MAX_DENOISING_STEPS:-64}"

TOKEN_FILE="${HF_TOKEN_FILE:-${SCRATCH}/.secrets/hf_token}"
if [[ -z "${HF_TOKEN:-}" && -f "${TOKEN_FILE}" ]]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "${TOKEN_FILE}")"
fi

ADAPTER_ARG=""
if [[ -n "${ADAPTER}" ]]; then ADAPTER_ARG="--adapter_path ${ADAPTER}"; fi

LAUNCH="$(mktemp "${SCRATCH}/projects/gdsdv2/logs/cdeval_launch.XXXXXX.sh")"
cat > "${LAUNCH}" <<EOF
set -euo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "==== countdown eval prompt_style=${PS} (model=${ADAPTER:-BASE}) ===="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${SCRATCH}/projects/gdsdv2" \
python -u -m DiffGemma.diffgemma_trl.countdown_eval \
  --prompt_style ${PS} --model_path ${MODEL} ${ADAPTER_ARG} \
  --test_file ${TEST_FILE} \
  --max_new_tokens ${MAX_NEW_TOKENS} --max_denoising_steps ${MAX_DENOISING_STEPS} \
  --batch_size ${BATCH} --limit ${LIMIT} --seed 0 \
  2>&1 | tee "eval_out/countdown_base_${PS}.txt"
EOF

trap 'rm -f "${LAUNCH}"' EXIT
srun -ul --environment="${SCRATCH}/projects/gdsdv2/cscs/gdsdv2-pt.toml" bash "${LAUNCH}"
