#!/bin/bash
# Local (no SLURM) SFT sweep across the three DiffusionGemma SFT objectives on the
# OpenR1-Math-220k (<=4096-token) traces. One training run per objective, each 4k steps
# with checkpoints at 1k/2k/3k/4k (eval 1k/2k/4k). Data-parallel across the given GPUs
# via `accelerate launch`. Mirrors local/wd1_countdown_train.sh's launch pattern.
#
#   bash local/math_sft_sweep.sh                              # full sweep, GPUs 0,1,2,3
#   GPUS=1,2 VARIANTS="base-sft" bash local/math_sft_sweep.sh # one objective, 2 GPUs
#   SMOKE=1 GPUS=1,2 bash local/math_sft_sweep.sh             # fast pipeline smoke
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
mkdir -p logs out
ACCEL="${ACCEL:-$REPO/.venv/bin/accelerate}"   # venv accelerate (not on PATH by default)

MODEL="${MODEL:-google/diffusiongemma-26B-A4B-it}"
VARIANTS="${VARIANTS:-base-sft reweighted-ce loo-ce}"
GPUS="${GPUS:-0,1,2,3}"
NPROC="${NPROC:-$(awk -F, '{print NF}' <<<"$GPUS")}"
MAIN_PORT="${MAIN_PORT:-29500}"   # accelerate rendezvous port; set DISTINCT per concurrent job
                                  # (e.g. two parallel sweeps split by dataset over 4 GPUs)
# Block-structured (multi-canvas) SFT: BLOCK=1 trains one sampled canvas_length block per
# example with the clean prior blocks as encoder context (matches the block-AR eval). CANVAS
# is then the BLOCK length (256, the eval config); the completion is kept up to
# MAX_COMPLETION and split into blocks. DATA_CANVAS is only the token length-filter the jsonl
# was built with (used in the filename), independent of the training block length.
BLOCK="${BLOCK:-0}"
if [[ "$BLOCK" == "1" ]]; then
  # 256 block canvas. The decoder forward already encodes [prompt; prior blocks] and returns
  # that hidden state, which the encoder AR loss REUSES (one shared encoder pass) -> a single
  # combined backward both fits (~one encoder graph) and is DDP-clean (every param gets a grad
  # each step). So interleaved is OFF and unused in block mode (sft_trainer forces single bwd).
  CANVAS="${CANVAS:-256}"; INTERLEAVED="${INTERLEAVED:-false}"
else
  CANVAS="${CANVAS:-4096}"; INTERLEAVED="${INTERLEAVED:-true}"   # 4096 canvas -> interleaved bwd
fi
DATA_CANVAS="${DATA_CANVAS:-4096}"          # build-time length filter (dataset filename only)
MAX_COMPLETION="${MAX_COMPLETION:-4096}"    # block mode: cap on the full completion before blocking
MAX_PROMPT="${MAX_PROMPT:-512}"
MAX_STEPS="${MAX_STEPS:-4000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
BATCH="${BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1.5e-4}"
ENC_W="${ENC_W:-1.0}"           # encoder AR (next-token) loss weight over [prompt; x0].
                                # Now active on the HF+DDP path (sft_trainer unwraps DDP+PEFT
                                # to reach base.model.encoder + base.lm_head). Adds an extra
                                # encoder forward + lm_head pass per step -> slower (see smoke).
WEIGHT_CLIP="${WEIGHT_CLIP:-100}"  # reweighted-ce: bound on w(t)=1/(1-t)
GC="${GC:-false}"               # gradient checkpointing (enable if canvas 4096 OOMs)
RESPONSES="${RESPONSES:-solution r1cot}"   # SFT response objectives to sweep (dataset per mode)
DS_SUFFIX="${DS_SUFFIX:-}"      # dataset filename suffix (SMOKE sets "_smoke")
OUT_SUFFIX="${OUT_SUFFIX:-}"    # extra tag on OUTDIR + RUN_NAME (e.g. LR sweep: _lr1.5e-5)
OUT_ROOT="${OUT_ROOT:-./out}"
WANDB_PROJECT="${WANDB_PROJECT:-diffuGemma-pt}"
REPORT_TO="${REPORT_TO:-wandb}"

# SMOKE: a few steps, no checkpoints, on the 200-row per-mode subsets.
if [[ "${SMOKE:-0}" == "1" ]]; then
  MAX_STEPS="${MAX_STEPS_SMOKE:-4}"; SAVE_STEPS=0
  VARIANTS="${VARIANTS_SMOKE:-base-sft}"
  DS_SUFFIX="_smoke"
  REPORT_TO=none
  echo "[smoke] steps=${MAX_STEPS} variants='${VARIANTS}' responses='${RESPONSES}' gpus=${GPUS}"
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export DIFFGEMMA_EXPERTS_IMPL="${DIFFGEMMA_EXPERTS_IMPL:-eager}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="$GPUS"
export WANDB_PROJECT="$WANDB_PROJECT"

if [[ "$SAVE_STEPS" == "0" ]]; then SAVE_ARGS="--save_strategy no"
else SAVE_ARGS="--save_strategy steps --save_steps ${SAVE_STEPS}"; fi

ANSWER_CONTENT_FRAC="${ANSWER_CONTENT_FRAC:-0.0}"   # block mode: upsample answer-content canvases (0=uniform)
if [[ "$BLOCK" == "1" ]]; then
  BLOCK_ARGS="--block_diffusion true --max_completion_length ${MAX_COMPLETION} --interleaved_backward ${INTERLEAVED} --answer_content_frac ${ANSWER_CONTENT_FRAC}"
  RUN_TAG="${RUN_TAG:-_block}"   # separate out/ dirs + wandb runs from the single-canvas sweep
  echo "[block] block-diffusion SFT: block(canvas)=${CANVAS} max_completion=${MAX_COMPLETION} interleaved=${INTERLEAVED} answer_content_frac=${ANSWER_CONTENT_FRAC} tag=${RUN_TAG}"
else
  BLOCK_ARGS="--block_diffusion false --interleaved_backward ${INTERLEAVED}"
  RUN_TAG="${RUN_TAG:-}"
fi

for RESP in $RESPONSES; do
 DATASET="${DATASET_OVERRIDE:-./dataset/openr1_math_sft_${RESP}_c${DATA_CANVAS}${DS_SUFFIX}.jsonl}"
 for VARIANT in $VARIANTS; do
  OUTDIR="${OUT_ROOT}/openr1_math_sft_${RESP}_${VARIANT}${RUN_TAG}${OUT_SUFFIX}"
  RUN_NAME="openr1_math_sft_${RESP}_${VARIANT}_c${CANVAS}${RUN_TAG}${OUT_SUFFIX}"
  EXTRA=""
  [[ "$VARIANT" == "reweighted-ce" ]] && EXTRA="--diffusion_weight_clip ${WEIGHT_CLIP}"

  # resume from latest checkpoint if present
  RESUME_ARG=""
  LATEST="$(ls -d "${OUTDIR}"/checkpoint-* 2>/dev/null | sed 's#.*/checkpoint-##' | sort -n | tail -1 || true)"
  [[ -n "${LATEST}" ]] && RESUME_ARG="--resume_from_checkpoint ${OUTDIR}/checkpoint-${LATEST}" \
    && echo "[resume] ${RESP}/${VARIANT} from checkpoint-${LATEST}"

  echo "==== [sft] resp=${RESP} variant=${VARIANT} canvas=${CANVAS} steps=${MAX_STEPS} data=${DATASET} -> ${OUTDIR} ===="
  set -x
  "$ACCEL" launch --num_processes "${NPROC}" --num_machines 1 --main_process_port "${MAIN_PORT}" --mixed_precision bf16 --dynamo_backend no \
    -m DiffGemma.diffgemma_trl.sft_train \
    --model_name_or_path "${MODEL}" \
    --use_unsloth false --use_peft true --lora_r 64 --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --dataset_path "${DATASET}" --prompt_style math \
    --chat_route true --reasoning_prefill true --enable_thinking false \
    --prompt_column prompt --completion_column completion \
    --sft_variant "${VARIANT}" ${EXTRA} ${BLOCK_ARGS} \
    --encoder_loss_weight "${ENC_W}" \
    --diffusion_canvas_length "${CANVAS}" --max_prompt_length "${MAX_PROMPT}" \
    --output_dir "${OUTDIR}" ${RESUME_ARG} \
    --max_steps "${MAX_STEPS}" ${SAVE_ARGS} \
    --per_device_train_batch_size "${BATCH}" --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate "${LR}" --warmup_steps 100 --lr_scheduler_type cosine --max_grad_norm 1.0 \
    --gradient_checkpointing "${GC}" \
    --remove_unused_columns false --logging_steps 10 \
    --report_to "${REPORT_TO}" --run_name "${RUN_NAME}" \
    2>&1 | tee "logs/${RUN_NAME}.log"
  set +x
 done
done
echo "==================== [sft] sweep DONE (responses='${RESPONSES}' variants='${VARIANTS}') ===================="
