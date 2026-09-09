#!/bin/bash
# Debug-partition smoke test of the new DiffusionGemma eval scripts (math /
# countdown / mbpp / humaneval), local diffusion generate() backend, BASE model
# only (no adapter), tiny --limit. Submit from the repo root:
#   sbatch cscs/eval_smoke.sbatch
# Override DATASETS / LIMIT / MODEL / ADAPTER / BACKEND via --export, e.g.:
#   sbatch --export=ALL,DATASETS="countdown math",LIMIT=4 cscs/eval_smoke.sbatch
#   sbatch --export=ALL,ADAPTER=./xp_sft_reweighted_ce/checkpoint-4000 cscs/eval_smoke.sbatch
#SBATCH --account=a0112
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=gdsdv2-eval-smoke
#SBATCH --output=logs/eval_smoke_%j.log

set -euo pipefail
cd "${SCRATCH}/projects/gdsdv2"
mkdir -p logs

MODEL="${MODEL:-google/diffusiongemma-26B-A4B-it}"
DATASETS="${DATASETS:-countdown math mbpp humaneval}"
LIMIT="${LIMIT:-4}"
BACKEND="${BACKEND:-local}"
ADAPTER="${ADAPTER:-}"               # empty => base model only
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_DENOISING_STEPS="${MAX_DENOISING_STEPS:-64}"

# Public model, usually cached in HF_HOME; pass a token through if present.
TOKEN_FILE="${HF_TOKEN_FILE:-${SCRATCH}/.secrets/hf_token}"
if [[ -z "${HF_TOKEN:-}" && -f "${TOKEN_FILE}" ]]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "${TOKEN_FILE}")"
fi

ADAPTER_ARG=""
if [[ -n "${ADAPTER}" ]]; then ADAPTER_ARG="--adapter_path ${ADAPTER}"; fi

LAUNCH="$(mktemp "${SCRATCH}/projects/gdsdv2/logs/eval_smoke_launch.XXXXXX.sh")"
cat > "${LAUNCH}" <<EOF
set -euo pipefail
source .venv/bin/activate
# Everything (26B model + all datasets) is pre-cached in HF_HOME; compute nodes
# have no internet, so force offline to avoid hub lookups hanging.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for ds in ${DATASETS}; do
  echo "==== eval \${ds} (backend=${BACKEND}, model=${ADAPTER:-BASE}) ===="
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${SCRATCH}/projects/gdsdv2" \
  python -m DiffGemma.diffgemma_trl.sft_eval_\${ds} \
    --backend ${BACKEND} --model_path ${MODEL} ${ADAPTER_ARG} \
    --max_new_tokens ${MAX_NEW_TOKENS} --max_denoising_steps ${MAX_DENOISING_STEPS} \
    --limit ${LIMIT}
done
EOF

trap 'rm -f "${LAUNCH}"' EXIT
srun -ul --environment="${SCRATCH}/projects/gdsdv2/cscs/gdsdv2-pt.toml" bash "${LAUNCH}"
