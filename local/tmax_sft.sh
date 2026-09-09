#!/bin/bash
# Tmax terminal-agent block-SFT for DiffusionGemma. Response-anchored canvas sampling on the
# converted trajectories (dataset/tmax_sft_only_success.jsonl): per step sample one assistant
# response, tile it into 256-token canvases, sample one canvas (denoiser target), and encode
# everything before it (whole-prefix encoder AR loss). LR 1.5e-5, 4k steps, checkpoint every 500.
#
#   GPUS=0,1 bash local/tmax_sft.sh
#   SMOKE=1 GPUS=1,2 bash local/tmax_sft.sh      # a few steps, no checkpoints
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs out
ACCEL="${ACCEL:-$REPO/.venv/bin/accelerate}"

MODEL="${MODEL:-google/diffusiongemma-26B-A4B-it}"
DATASET="${DATASET:-dataset/tmax_sft_only_success.jsonl}"
GPUS="${GPUS:-0,1}"; NPROC="${NPROC:-$(awk -F, '{print NF}' <<<"$GPUS")}"
MAIN_PORT="${MAIN_PORT:-29610}"
LR="${LR:-1.5e-5}"; MAX_STEPS="${MAX_STEPS:-4000}"; SAVE_STEPS="${SAVE_STEPS:-500}"
CANVAS=256; MAX_PROMPT="${MAX_PROMPT:-4096}"; MAX_COMPLETION="${MAX_COMPLETION:-4096}"
ENC_W="${ENC_W:-1.0}"; BATCH="${BATCH:-1}"; GRAD_ACCUM="${GRAD_ACCUM:-8}"
OUTDIR="${OUTDIR:-out/tmax_sft}"; REPORT_TO="${REPORT_TO:-wandb}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  MAX_STEPS="${MAX_STEPS_SMOKE:-6}"; SAVE_STEPS=0; REPORT_TO=none; OUTDIR="out/tmax_sft_smoke"
  echo "[smoke] steps=${MAX_STEPS} gpus=${GPUS}"
fi
RUN_NAME="$(basename "$OUTDIR")"

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export DIFFGEMMA_EXPERTS_IMPL=eager PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES="$GPUS"

RESUME_ARG=""
LATEST="$(ls -d "${OUTDIR}"/checkpoint-* 2>/dev/null | sed 's#.*/checkpoint-##' | sort -n | tail -1 || true)"
[[ -n "${LATEST:-}" ]] && RESUME_ARG="--resume_from_checkpoint ${OUTDIR}/checkpoint-${LATEST}" \
  && echo "[resume] from checkpoint-${LATEST}"
if [[ "$SAVE_STEPS" == "0" ]]; then SAVE_ARGS="--save_strategy no"
else SAVE_ARGS="--save_strategy steps --save_steps ${SAVE_STEPS}"; fi

echo "==== [tmax-sft] LR=${LR} steps=${MAX_STEPS} canvas=${CANVAS} data=${DATASET} -> ${OUTDIR} ===="
set -x
"$ACCEL" launch --num_processes "$NPROC" --num_machines 1 --main_process_port "$MAIN_PORT" --mixed_precision bf16 --dynamo_backend no \
  -m DiffGemma.diffgemma_trl.sft_train \
  --model_name_or_path "$MODEL" \
  --use_unsloth false --use_peft true --lora_r 64 --lora_alpha 128 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --dataset_path "$DATASET" --prompt_style math \
  --chat_route true --reasoning_prefill false --enable_thinking false \
  --prompt_column prompt --completion_column completion \
  --sft_variant base-sft \
  --block_diffusion true --tmax_response_sampling true --interleaved_backward false \
  --max_completion_length "$MAX_COMPLETION" \
  --encoder_loss_weight "$ENC_W" \
  --diffusion_canvas_length "$CANVAS" --max_prompt_length "$MAX_PROMPT" \
  --output_dir "$OUTDIR" ${RESUME_ARG} \
  --max_steps "$MAX_STEPS" ${SAVE_ARGS} \
  --per_device_train_batch_size "$BATCH" --gradient_accumulation_steps "$GRAD_ACCUM" \
  --learning_rate "$LR" --warmup_steps 100 --lr_scheduler_type cosine --max_grad_norm 1.0 \
  --gradient_checkpointing false --remove_unused_columns false --logging_steps 10 \
  --report_to "$REPORT_TO" --run_name "$RUN_NAME" \
  2>&1 | tee "logs/${RUN_NAME}.log"
set +x
echo "==================== [tmax-sft] DONE -> ${OUTDIR} ===================="
