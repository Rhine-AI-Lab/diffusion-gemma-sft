#!/usr/bin/env bash
set -euo pipefail

# Commented Kauldron/JAX fine-tune wrapper for the official Gemma package.
#
# The official configs import `gemma.diffusion...` and use data paths under
# `gemma/diffusion/hackable_diffusion_adapter/...`. Run this from, or point
# GEMMA_PARENT at, the parent directory that contains the official `gemma/`
# source tree.
#
# Examples:
#   GEMMA_PARENT=/work/google-deepmind TASK=sudoku ACTION=prepare-data bash scripts/run_diffgemma_finetune.sh
#   GEMMA_PARENT=/work/google-deepmind TASK=sudoku SFT_MODE=lora WORKDIR=/tmp/xp_sudoku bash scripts/run_diffgemma_finetune.sh
#   GEMMA_PARENT=/work/google-deepmind ACTION=eval TASK=sudoku WORKDIR=/tmp/xp_sudoku STEP=1000 bash scripts/run_diffgemma_finetune.sh

ACTION="${ACTION:-train}"          # train | eval | prepare-data | test
TASK="${TASK:-sudoku}"             # sudoku | pubmedqa
SFT_MODE="${SFT_MODE:-lora}"       # sudoku only: lora | full
GEMMA_PARENT="${GEMMA_PARENT:-$(pwd)}"
WORKDIR="${WORKDIR:-${GEMMA_PARENT}/xp_dir_${TASK}_${SFT_MODE}}"

STEP="${STEP:-}"                   # eval only; empty means latest checkpoint
EVAL_NAMES="${EVAL_NAMES:-sample_ar_steps64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
EVAL_NUM_BATCHES="${EVAL_NUM_BATCHES:-}" # optional; set to 1 or 2 for smoke tests
NUM_CANVASES="${NUM_CANVASES:-}"   # optional eval override
MAX_STEPS="${MAX_STEPS:-}"         # optional train smoke-test override

GEMMA_DIR="${GEMMA_PARENT}/gemma"
ADAPTER_DIR="gemma/diffusion/hackable_diffusion_adapter"

if [[ ! -d "${GEMMA_DIR}" ]]; then
  echo "Expected GEMMA_PARENT to contain a gemma/ directory: ${GEMMA_PARENT}" >&2
  echo "Set GEMMA_PARENT=/path/to/parent/of/gemma before running." >&2
  exit 1
fi

cd "${GEMMA_PARENT}"
export PYTHONPATH="${GEMMA_PARENT}:${PYTHONPATH:-}"

# These defaults mirror the official README recommendations for avoiding JAX
# compilation/NCCL hangs on multi-GPU runs.
export XLA_FLAGS="${XLA_FLAGS:---xla_disable_hlo_passes=constant_folding}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export NCCL_PROTO="${NCCL_PROTO:-LL128}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"

case "${TASK}:${SFT_MODE}" in
  sudoku:lora)
    CFG="${ADAPTER_DIR}/configs/sft_sudoku.py"
    ;;
  sudoku:full)
    CFG="${ADAPTER_DIR}/configs/sft_sudoku_full.py"
    ;;
  pubmedqa:*)
    CFG="${ADAPTER_DIR}/configs/sft_pubmedqa.py"
    ;;
  *)
    echo "Unsupported TASK/SFT_MODE: TASK=${TASK}, SFT_MODE=${SFT_MODE}" >&2
    exit 1
    ;;
esac

if [[ ! -f "${CFG}" ]]; then
  echo "Config not found: ${GEMMA_PARENT}/${CFG}" >&2
  echo "Make sure the DiffusionGemma adapter exists in the official gemma tree." >&2
  exit 1
fi

case "${ACTION}" in
  prepare-data)
    if [[ "${TASK}" == "sudoku" ]]; then
      bash "${ADAPTER_DIR}/data/sudoku/prepare_sudoku_dataset.sh"
    elif [[ "${TASK}" == "pubmedqa" ]]; then
      bash "${ADAPTER_DIR}/data/pubmedqa/prepare_pubmedqa_dataset.sh"
    else
      echo "Unknown TASK for prepare-data: ${TASK}" >&2
      exit 1
    fi
    ;;

  train)
    ARGS=(
      -m kauldron.main
      --cfg="${CFG}"
      --cfg.workdir="${WORKDIR}"
    )
    if [[ -n "${MAX_STEPS}" ]]; then
      ARGS+=(--cfg.num_train_steps="${MAX_STEPS}")
    fi
    if [[ -n "${EVAL_NUM_BATCHES}" ]]; then
      ARGS+=(--cfg.aux.eval_num_batches="${EVAL_NUM_BATCHES}")
    fi
    python3 "${ARGS[@]}"
    ;;

  eval)
    export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
    export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"

    ARGS=(
      -m gemma.diffusion.hackable_diffusion_adapter.eval_main
      --cfg="${CFG}"
      --task="${TASK}"
      --eval_names="${EVAL_NAMES}"
      --cfg.workdir="${WORKDIR}"
      --cfg.eval_ds.batch_size="${EVAL_BATCH_SIZE}"
    )
    if [[ -n "${EVAL_NUM_BATCHES}" ]]; then
      ARGS+=(--cfg.aux.eval_num_batches="${EVAL_NUM_BATCHES}")
    fi
    if [[ -n "${STEP}" ]]; then
      ARGS+=(--step="${STEP}")
    fi
    if [[ -n "${NUM_CANVASES}" ]]; then
      ARGS+=(--cfg.aux.num_canvases="${NUM_CANVASES}")
    fi
    python3 "${ARGS[@]}"
    ;;

  test)
    pytest "${ADAPTER_DIR}"
    ;;

  *)
    echo "Unknown ACTION: ${ACTION}. Use train, eval, prepare-data, or test." >&2
    exit 1
    ;;
esac
