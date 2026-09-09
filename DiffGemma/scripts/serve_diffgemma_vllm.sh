#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Serve DiffusionGemma with vLLM (OpenAI-compatible endpoint) for evaluation.
#
# The diffusion sampler (entropy_bound), canvas length and denoising behaviour are
# fixed HERE at serve time via --hf-overrides / --diffusion-config; the eval client
# (sft_eval_*.py --backend vllm) only sets max_tokens / temperature / seed.
#
# Usage:
#   bash DiffGemma/scripts/serve_diffgemma_vllm.sh
#   MODEL_ID=/path/to/local/diffusiongemma bash DiffGemma/scripts/serve_diffgemma_vllm.sh
#   ADAPTER_NAME=my_ckpt ADAPTER_PATH=./xp_sft/checkpoint-4000 \
#       bash DiffGemma/scripts/serve_diffgemma_vllm.sh        # LoRA (if the fork supports it)
#
# On CSCS Clariden, run this INSIDE a `srun -p debug ... --environment=<edf.toml>`
# allocation (never the login node); serve + client share the node's localhost.
# =============================================================================

MODEL_ID="${MODEL_ID:-google/diffusiongemma-26B-A4B-it}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
CANVAS_LENGTH="${CANVAS_LENGTH:-256}"
ENTROPY_BOUND="${ENTROPY_BOUND:-0.1}"
# Optional: override the entropy-bound sampler's max denoising steps per canvas
# (generation_config.json default is 48). Empty => use the model default.
MAX_DENOISING_STEPS="${MAX_DENOISING_STEPS:-}"

if [[ -n "${MAX_DENOISING_STEPS}" ]]; then
  DIFFUSION_CONFIG="{\"canvas_length\": ${CANVAS_LENGTH}, \"max_denoising_steps\": ${MAX_DENOISING_STEPS}}"
else
  DIFFUSION_CONFIG="{\"canvas_length\": ${CANVAS_LENGTH}}"
fi

ARGS=(
  serve "${MODEL_ID}"
  --port "${PORT}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --attention-backend TRITON_ATTN
  --generation-config vllm
  --hf-overrides "{\"diffusion_sampler\": \"entropy_bound\", \"diffusion_entropy_bound\": ${ENTROPY_BOUND}}"
  --diffusion-config "${DIFFUSION_CONFIG}"
  --enable-chunked-prefill
)

# CUDA graphs are captured sized to the canvas, but vLLM caps
# max_cudagraph_capture_size at 512 -> canvas_length > 512 (e.g. 1024/2048) fails
# engine init ("No valid cudagraph sizes"). Set ENFORCE_EAGER=1 for those to disable
# CUDA graphs (a bit slower, but the only way to serve large canvases).
if [[ -n "${ENFORCE_EAGER:-}" ]]; then
  ARGS+=(--enforce-eager)
fi

# Optional: serve a LoRA adapter so it can be evaluated through vLLM
# (--backend vllm --vllm_model "${ADAPTER_NAME}"). Requires a diffusion vLLM
# build with LoRA support.
if [[ -n "${ADAPTER_PATH:-}" ]]; then
  ARGS+=(--enable-lora --lora-modules "${ADAPTER_NAME:-adapter}=${ADAPTER_PATH}")
fi

exec vllm "${ARGS[@]}"
