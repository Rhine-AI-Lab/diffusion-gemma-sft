#!/bin/bash
# Full base-model eval of ONE dataset on the debug partition (local diffusion
# generate() backend, no adapter). Submit one job per dataset, e.g.:
#   sbatch --job-name=eval-math     --export=ALL,DS=math      cscs/eval_base.sbatch
#   sbatch --job-name=eval-mbpp     --export=ALL,DS=mbpp,BATCH=16 cscs/eval_base.sbatch
#SBATCH --account=a0112
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=gdsdv2-eval-base
#SBATCH --output=logs/eval_base_%x_%j.log

set -euo pipefail
cd "${SCRATCH}/projects/gdsdv2"
mkdir -p logs eval_out

MODEL="${MODEL:-google/diffusiongemma-26B-A4B-it}"
DS="${DS:?set DS=math|countdown|mbpp|humaneval}"
BATCH="${BATCH:-16}"
LIMIT="${LIMIT:-0}"                 # 0 => full dataset
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_DENOISING_STEPS="${MAX_DENOISING_STEPS:-64}"

TOKEN_FILE="${HF_TOKEN_FILE:-${SCRATCH}/.secrets/hf_token}"
if [[ -z "${HF_TOKEN:-}" && -f "${TOKEN_FILE}" ]]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "${TOKEN_FILE}")"
fi

LAUNCH="$(mktemp "${SCRATCH}/projects/gdsdv2/logs/eval_base_launch.XXXXXX.sh")"
cat > "${LAUNCH}" <<EOF
set -euo pipefail
source .venv/bin/activate
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "==== FULL eval ${DS} (base model, batch=${BATCH}, limit=${LIMIT}) ===="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${SCRATCH}/projects/gdsdv2" \
python -m DiffGemma.diffgemma_trl.sft_eval_${DS} \
  --backend local --model_path ${MODEL} \
  --max_new_tokens ${MAX_NEW_TOKENS} --max_denoising_steps ${MAX_DENOISING_STEPS} \
  --batch_size ${BATCH} --limit ${LIMIT} \
  2>&1 | tee "eval_out/${DS}_base.txt"
EOF

trap 'rm -f "${LAUNCH}"' EXIT
srun -ul --environment="${SCRATCH}/projects/gdsdv2/cscs/gdsdv2-pt.toml" bash "${LAUNCH}"
