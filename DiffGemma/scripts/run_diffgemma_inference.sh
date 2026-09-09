#!/usr/bin/env bash
set -euo pipefail

# Commented HF/Transformers inference wrapper for DiffusionGemma.
#
# Example:
#   PROMPT="Why is the sky blue?" bash scripts/run_diffgemma_inference.sh
#
# Multimodal:
#   IMAGE="https://..." PROMPT="Describe this image." bash scripts/run_diffgemma_inference.sh

MODEL_ID="${MODEL_ID:-${MODEL_PATH:-google/diffusiongemma-26B-A4B-it}}"
CACHE_DIR="${CACHE_DIR:-}"
OUTPUT_FILE="${OUTPUT_FILE:-}"

PROMPT="${PROMPT:-Why is the sky blue?}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
THINKING="${THINKING:-0}"
IMAGE="${IMAGE:-}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_DENOISING_STEPS="${MAX_DENOISING_STEPS:-48}"
ENTROPY_BOUND="${ENTROPY_BOUND:-0.1}"
T_MIN="${T_MIN:-0.4}"
T_MAX="${T_MAX:-0.8}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.005}"
STABILITY_THRESHOLD="${STABILITY_THRESHOLD:-1}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
DTYPE="${DTYPE:-auto}"
CACHE_IMPLEMENTATION="${CACHE_IMPLEMENTATION:-}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
SKIP_SPECIAL_TOKENS="${SKIP_SPECIAL_TOKENS:-0}"

ARGS=(
  --model_id "${MODEL_ID}"
  --prompt "${PROMPT}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --max_denoising_steps "${MAX_DENOISING_STEPS}"
  --entropy_bound "${ENTROPY_BOUND}"
  --t_min "${T_MIN}"
  --t_max "${T_MAX}"
  --confidence_threshold "${CONFIDENCE_THRESHOLD}"
  --stability_threshold "${STABILITY_THRESHOLD}"
  --device_map "${DEVICE_MAP}"
  --dtype "${DTYPE}"
)

if [[ -n "${CACHE_DIR}" ]]; then
  ARGS+=(--cache_dir "${CACHE_DIR}")
fi

if [[ -n "${OUTPUT_FILE}" ]]; then
  ARGS+=(--output_file "${OUTPUT_FILE}")
fi

if [[ -n "${SYSTEM_PROMPT}" ]]; then
  ARGS+=(--system_prompt "${SYSTEM_PROMPT}")
fi

if [[ "${THINKING}" == "1" ]]; then
  ARGS+=(--thinking)
fi

if [[ -n "${IMAGE}" ]]; then
  ARGS+=(--image "${IMAGE}")
fi

if [[ -n "${CACHE_IMPLEMENTATION}" ]]; then
  ARGS+=(--cache_implementation "${CACHE_IMPLEMENTATION}")
fi

if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  ARGS+=(--local_files_only)
fi

if [[ "${SKIP_SPECIAL_TOKENS}" == "1" ]]; then
  ARGS+=(--skip_special_tokens)
fi

python DiffGemma/examples/hf_inference.py "${ARGS[@]}"
