#!/usr/bin/env bash
# Run the OpenThoughts-TBLite terminal-agent benchmark against a locally
# served DiffusionGemma model.
#
# Prerequisites:
#   1. ./setup.sh has been run (installs `harbor`, clones tasks/).
#   2. A DiffusionGemma OpenAI-compatible vLLM server is already running,
#      e.g. from the repo root:
#        VLLM_SERVED_MODEL_NAME=diffgemma \
#          bash DiffGemma/diffgemma_trl/run_vllm_openai_gemma4.sh \
#          google/diffusiongemma-26B-A4B-it
#      `--served-model-name` must be a bare alias (letters/digits/._- only,
#      no '/') because Harbor's `hosted_vllm/<name>` routing requires exactly
#      one '/' in the model string.
#
# Harbor calls the model over LiteLLM's `hosted_vllm/` provider, which talks
# plain OpenAI-compatible chat completions to --api-base. `terminus-2` is
# Harbor/Terminal-Bench's own reference agent: a minimal "shell + model" loop
# with no extra coding-agent harness, which is what we want to benchmark the
# base model's own tool-use rather than e.g. Claude Code's scaffolding.
set -euo pipefail
cd "$(dirname "$0")"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-diffgemma}"
VLLM_API_BASE="${VLLM_API_BASE:-http://localhost:8000/v1}"
AGENT="${AGENT:-terminus-2}"
N_CONCURRENT="${N_CONCURRENT:-4}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-8192}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-2048}"
JOB_NAME="${JOB_NAME:-diffgemma_$(date +%Y%m%d_%H%M%S)}"

if ! command -v harbor >/dev/null 2>&1; then
  echo "ERROR: harbor CLI not found on PATH. Run ./setup.sh first (or 'uv tool update-shell')." >&2
  exit 127
fi

if [ ! -d tasks ]; then
  echo "ERROR: tasks/ not found. Run ./setup.sh first to clone OpenThoughts-TBLite." >&2
  exit 1
fi

MODEL_INFO=$(cat <<JSON
{"max_input_tokens": ${MAX_INPUT_TOKENS}, "max_output_tokens": ${MAX_OUTPUT_TOKENS}, "input_cost_per_token": 0, "output_cost_per_token": 0}
JSON
)

echo "=== TBLite eval | model=hosted_vllm/${SERVED_MODEL_NAME} | api_base=${VLLM_API_BASE} | agent=${AGENT} | n_concurrent=${N_CONCURRENT} ==="
harbor run \
  --path tasks \
  --agent "$AGENT" \
  --model "hosted_vllm/${SERVED_MODEL_NAME}" \
  --ak "api_base=${VLLM_API_BASE}" \
  --ak "model_info=${MODEL_INFO}" \
  --env docker \
  --n-concurrent "$N_CONCURRENT" \
  --job-name "$JOB_NAME" \
  --jobs-dir jobs \
  "$@"

echo "=== Done. Aggregate scores: jobs/${JOB_NAME}/result.json | Browse trajectories: harbor view jobs ==="
