#!/usr/bin/env bash
# Plain vLLM OpenAI-compatible server for DiffusionGemma/Gemma4-style serving.
# This is useful for manual probing. The TRL GRPO trainer uses run_diffu_grpo_vllm_server.sh instead.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root (~/gdsdv2)

MODEL_PATH="${1:-google/diffusiongemma-26B-A4B-it}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
TP_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
# NOTE: intentionally NOT named VLLM_PORT -- vLLM itself reads that exact env
# var internally for distributed/engine-coordination port allocation
# (vllm.envs.VLLM_PORT), unrelated to the HTTP --port flag. Reusing that name
# here previously caused multiple concurrent instances to collide on vLLM's
# internal port instead of each getting their own HTTP port.
API_PORT="${VLLM_API_PORT:-8000}"
# The google/diffusiongemma-26B-A4B-it repo ships its own chat_template.jinja
# (with tool-call formatting matching --tool-call-parser gemma4), which vLLM
# auto-loads from the model dir/tokenizer by default. Only set
# VLLM_CHAT_TEMPLATE to override it with a different template file.
CHAT_TEMPLATE="${VLLM_CHAT_TEMPLATE:-}"
# Clean alias with no '/' for clients (e.g. Harbor/LiteLLM hosted_vllm/<name>)
# that require a bare served-model-name instead of the full HF repo id.
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-}"

if command -v uv >/dev/null 2>&1; then
  VLLM=(uv run vllm)
elif [ -x .venv/bin/vllm ]; then
  VLLM=(.venv/bin/vllm)
else
  echo "ERROR: no uv or .venv/bin/vllm found. Run from ~/gdsdv2." >&2
  exit 127
fi

SERVED_NAME_ARGS=()
if [ -n "$SERVED_MODEL_NAME" ]; then
  SERVED_NAME_ARGS=(--served-model-name "$SERVED_MODEL_NAME")
fi

CHAT_TEMPLATE_ARGS=()
if [ -n "$CHAT_TEMPLATE" ]; then
  CHAT_TEMPLATE_ARGS=(--chat-template "$CHAT_TEMPLATE")
fi

MAX_MODEL_LEN_ARGS=()
if [ -n "$MAX_MODEL_LEN" ]; then
  MAX_MODEL_LEN_ARGS=(--max-model-len "$MAX_MODEL_LEN")
fi

echo "=== OpenAI vLLM server | model=$MODEL_PATH | served_name=${SERVED_MODEL_NAME:-$MODEL_PATH} | GPUs=$CUDA_VISIBLE_DEVICES | port=$API_PORT | $(date) ==="
"${VLLM[@]}" serve "$MODEL_PATH"   --port "$API_PORT"   --max-num-seqs "$MAX_NUM_SEQS"   --tensor-parallel-size "$TP_SIZE"   --enable-auto-tool-choice   --tool-call-parser gemma4   --reasoning-parser gemma4   "${SERVED_NAME_ARGS[@]}"   "${CHAT_TEMPLATE_ARGS[@]}"   "${MAX_MODEL_LEN_ARGS[@]}"
