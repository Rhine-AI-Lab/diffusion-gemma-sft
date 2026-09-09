#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Convenience launcher: evaluate DiffusionGemma on math / countdown / mbpp /
# humaneval (and gsm8k) with a single backend setting.
#
# Base model only (no adapter), local diffusion generate():
#   bash DiffGemma/scripts/run_diffgemma_eval.sh
#
# A LoRA checkpoint, local backend:
#   ADAPTER_PATH=./xp_sft_reweighted_ce/checkpoint-4000 \
#     bash DiffGemma/scripts/run_diffgemma_eval.sh
#
# Against a running vLLM server (see serve_diffgemma_vllm.sh):
#   BACKEND=vllm VLLM_BASE_URL=http://localhost:8000/v1 \
#     VLLM_MODEL=google/diffusiongemma-26B-A4B-it \
#     bash DiffGemma/scripts/run_diffgemma_eval.sh
#
# On CSCS Clariden submit this inside `srun -p debug ... --environment=<edf.toml>`.
# =============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

DATASETS="${DATASETS:-math countdown mbpp humaneval}"
BACKEND="${BACKEND:-local}"
MODEL_PATH="${MODEL_PATH:-unsloth/diffusiongemma-26B-A4B-it}"
ADAPTER_PATH="${ADAPTER_PATH:-}"            # empty => base model only
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
VLLM_MODEL="${VLLM_MODEL:-google/diffusiongemma-26B-A4B-it}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_DENOISING_STEPS="${MAX_DENOISING_STEPS:-64}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LIMIT="${LIMIT:-0}"
SEED="${SEED:-0}"

for ds in ${DATASETS}; do
  echo "==== eval ${ds} (backend=${BACKEND}) ===="
  ARGS=(
    --backend "${BACKEND}"
    --model_path "${MODEL_PATH}"
    --vllm_base_url "${VLLM_BASE_URL}"
    --vllm_model "${VLLM_MODEL}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --batch_size "${BATCH_SIZE}"
    --limit "${LIMIT}"
    --seed "${SEED}"
  )
  if [[ -n "${ADAPTER_PATH}" ]]; then
    ARGS+=(--adapter_path "${ADAPTER_PATH}")
  fi
  if [[ "${BACKEND}" == "local" ]]; then
    ARGS+=(--max_denoising_steps "${MAX_DENOISING_STEPS}")
  fi
  PYTHONPATH="${REPO_ROOT}" python -m "DiffGemma.diffgemma_trl.sft_eval_${ds}" "${ARGS[@]}" \
    || { echo "[WARN] eval failed: ${ds}"; continue; }
done

echo "All evaluations completed!"
