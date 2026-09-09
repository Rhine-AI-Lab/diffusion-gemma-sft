#!/bin/bash
# Shared helpers to (re)serve DiffusionGemma with vLLM for the math-benchmark evals.
#
# The diffusion sampler (entropy_bound) is fixed at SERVE time; we vary canvas_length
# per server so that #canvases = ceil(max_new_tokens / canvas_length). Uses the
# ISOLATED .venv-vllm (vllm 0.24.0) so the training .venv (torch 2.12 / transformers
# 5.12.1) is never touched. NOTE: DiffusionGemma is single-GPU only in vLLM 0.24.0
# (TP/PP>1 crashes, issue #45719) -> one server per GPU.
#
# MEMORY NOTE: the diffusion sampler warms up a float32 buffer sized
#   max_num_seqs * canvas_length * vocab(262144) * 4 bytes. With canvas 2048 and a
#   large max_num_seqs this is tens of GiB and OOMs at high gpu_mem_util. Keep
#   MAX_NUM_SEQS small (the client is serial) and GPU_MEM_UTIL modest; scale
#   MAX_NUM_SEQS down as canvas grows.
#
# source this file, then: pid=$(vllm_start <gpu> <port> <canvas> <max_model_len> <log>)
#                         vllm_wait <port> [timeout_s]
#                         ... run client evals ...
#                         vllm_stop "$pid"

_VLLM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_VLLM_SERVE_SH="$_VLLM_REPO/DiffGemma/scripts/serve_diffgemma_vllm.sh"

vllm_start() {   # gpu port canvas max_model_len logfile
  local gpu="$1" port="$2" canvas="$3" mml="$4" logf="$5"
  CUDA_VISIBLE_DEVICES="$gpu" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PATH="$_VLLM_REPO/.venv-vllm/bin:$PATH" \
  MODEL_ID="${MODEL_ID:-google/diffusiongemma-26B-A4B-it}" \
  PORT="$port" CANVAS_LENGTH="$canvas" MAX_MODEL_LEN="$mml" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}" GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}" \
  ENTROPY_BOUND="${ENTROPY_BOUND:-0.1}" MAX_DENOISING_STEPS="${MAX_DENOISING_STEPS:-}" \
  ENFORCE_EAGER="${ENFORCE_EAGER:-}" \
  bash "$_VLLM_SERVE_SH" > "$logf" 2>&1 &
  echo $!
}

vllm_wait() {    # port pid [timeout_s]
  local port="$1" pid="$2" timeout="${3:-1800}" waited=0
  until curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; do
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "[error] vLLM server pid=$pid died before becoming healthy (check the log)" >&2
      return 1
    fi
    sleep 5; waited=$((waited + 5))
    if [[ "$waited" -ge "$timeout" ]]; then
      echo "[error] vLLM on :${port} not healthy after ${timeout}s" >&2; return 1
    fi
  done
  echo "[vllm] healthy on :${port} after ${waited}s"
}

vllm_stop() {    # pid
  local pid="${1:-}"
  [[ -n "$pid" ]] || return 0
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  echo "[vllm] stopped pid=$pid"
}
