#!/bin/bash
# Full DiffusionGemma math-benchmark sweep (runs AFTER the smoke test is approved).
#
# Best prompt format is chosen by the smoke test and passed via env:
#     THINK=0|1  REASONING=0|1   (from local/smoke_mathbench.sh's BEST FORMAT line)
#
# Grid: 6 datasets x generation lengths {256,512,1024,2048} x {1 canvas, 2 canvases}
# (256 = 1 canvas only) = 7 (gen,canvas) configs x 6 datasets = 42 runs.
#
# canvas_length is fixed at SERVE time, so we run one server per canvas_length and
# point client runs of the appropriate max_new_tokens at it (#canvases = gen/canvas):
#     canvas 256  -> gen 256 (1c), 512 (2c)      GPU 1
#     canvas 512  -> gen 512 (1c), 1024 (2c)     GPU 2
#     canvas 1024 -> gen 1024 (1c), 2048 (2c)    GPU 4
#     canvas 2048 -> gen 2048 (1c)               GPU 5
# The 4 canvas servers run concurrently (one GPU each). Non-AIME datasets are
# sharded across NSHARDS concurrent clients (server batches them); AIME datasets
# use avg@k (seeds 0..K-1, mean+/-std).
#
#   THINK=0 REASONING=1 bash local/run_mathbench.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
source local/_vllm_lib.sh
mkdir -p eval_out logs

STEPS="${STEPS:-}"                      # max denoising steps; empty => model default (48)
export MAX_DENOISING_STEPS="$STEPS"
DATASETS_ALL="${DATASETS:-math500 minerva olympiad amc aime24 aime25}"
AVGK_DATASETS="aime24 aime25"
NSHARDS="${NSHARDS:-4}"                 # concurrent client shards per non-AIME (ds,gen)
K="${K:-8}"                             # avg@k for AIME
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
VLLM_MODEL="${VLLM_MODEL:-google/diffusiongemma-26B-A4B-it}"
CANVASES="${CANVASES:-256 512 1024 2048}"
# server batches up to MAX_NUM_SEQS requests; match it to NSHARDS. Sampler warmup
# buffer ~ MAX_NUM_SEQS*canvas*262144*4 bytes, so keep this modest for canvas 2048.
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-$NSHARDS}" GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"
export PYTHONPATH="$REPO"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false

# Prompt route: DEFAULT = native apply_chat_template (thinking channel).
#   ENABLE_THINKING=1  -> apply_chat_template enable_thinking (dropped acc in testing).
#   RAW=1              -> hand-written <|turn> template with THINK/REASONING knobs.
FMT_FLAGS=""
if [[ "${RAW:-0}" == "1" ]]; then
  FMT_FLAGS="--raw_template"
  [[ "${THINK:-0}" == "1" ]] && FMT_FLAGS="$FMT_FLAGS --think"
  [[ "${REASONING:-0}" == "1" ]] && FMT_FLAGS="$FMT_FLAGS --reasoning_prefill"
  echo "==== [sweep] RAW format: think=${THINK:-0} reasoning_prefill=${REASONING:-0} | steps=${STEPS:-48} | datasets: ${DATASETS_ALL} ===="
else
  # Format 6 default: chat + reasoning_prefill (validated 60.4% on full MATH-500).
  FMT_FLAGS=""
  [[ "${ENABLE_THINKING:-0}" == "1" ]] && FMT_FLAGS="$FMT_FLAGS --enable_thinking"
  [[ "${REASONING:-1}" == "1" ]] && FMT_FLAGS="$FMT_FLAGS --reasoning_prefill"
  echo "==== [sweep] CHAT (apply_chat_template) enable_thinking=${ENABLE_THINKING:-0} reasoning_prefill=${REASONING:-1} | steps=${STEPS:-48} | datasets: ${DATASETS_ALL} ===="
fi

canvas_gpu()  { case "$1" in 256) echo 1;; 512) echo 2;; 1024) echo 4;; 2048) echo 5;; esac; }
canvas_port() { case "$1" in 256) echo 8010;; 512) echo 8011;; 1024) echo 8012;; 2048) echo 8013;; esac; }
canvas_gens() { case "$1" in 256) echo "256 512";; 512) echo "512 1024";; 1024) echo "1024 2048";; 2048) echo "2048";; esac; }
is_avgk() { [[ " $AVGK_DATASETS " == *" $1 "* ]]; }

# One canvas server + all its client evals, then stop. Meant to run in the background.
run_canvas_block() {   # canvas
  local canvas="$1" gpu port gens log pid gen ds tag
  gpu="$(canvas_gpu "$canvas")"; port="$(canvas_port "$canvas")"; gens="$(canvas_gens "$canvas")"
  log="logs/vllm_sweep_c${canvas}_gpu${gpu}.log"
  # canvas > 512 has no valid CUDA-graph size (capture cap is 512) -> enforce eager.
  local ee=""; [[ "$canvas" -gt 512 ]] && ee=1
  echo "[sweep] serve canvas=${canvas} on GPU ${gpu} (:${port})${ee:+ [enforce-eager]}"
  pid="$(ENFORCE_EAGER="$ee" vllm_start "$gpu" "$port" "$canvas" "$MAX_MODEL_LEN" "$log")"
  if ! vllm_wait "$port" "$pid" 1800; then
    echo "[error] canvas=${canvas} serve failed; tail ${log}:"; tail -40 "$log"; return 1
  fi

  for gen in $gens; do
    for ds in $DATASETS_ALL; do
      tag="mathbench_${ds}_g${gen}_c${canvas}"
      if is_avgk "$ds"; then
        echo "[sweep] ${tag}  (avg@${K})"
        .venv/bin/python -u -m DiffGemma.diffgemma_trl.sft_eval_mathbench \
          --backend vllm --vllm_base_url "http://localhost:${port}/v1" --vllm_model "$VLLM_MODEL" \
          --dataset "$ds" --k "$K" --max_new_tokens "$gen" $FMT_FLAGS \
          --result_file "eval_out/${tag}.json" > "logs/${tag}.log" 2>&1 \
          || echo "[warn] ${tag} failed (see logs/${tag}.log)"
      else
        echo "[sweep] ${tag}  (sharded x${NSHARDS})"
        local spids=() sid
        for sid in $(seq 0 $((NSHARDS - 1))); do
          rm -f "eval_out/${tag}.shard${sid}.json"
          .venv/bin/python -u -m DiffGemma.diffgemma_trl.sft_eval_mathbench \
            --backend vllm --vllm_base_url "http://localhost:${port}/v1" --vllm_model "$VLLM_MODEL" \
            --dataset "$ds" --max_new_tokens "$gen" $FMT_FLAGS --seed 0 \
            --num_shards "$NSHARDS" --shard_id "$sid" \
            --result_file "eval_out/${tag}.shard${sid}.json" > "logs/${tag}.shard${sid}.log" 2>&1 &
          spids+=($!)
        done
        local sfail=0; for p in "${spids[@]}"; do wait "$p" || sfail=1; done
        [[ "$sfail" -ne 0 ]] && echo "[warn] ${tag}: a shard failed (see logs/${tag}.shard*.log)"
        .venv/bin/python - "$tag" "$NSHARDS" <<'PY' || echo "[warn] ${tag}: aggregation failed"
import json, sys
tag, ns = sys.argv[1], int(sys.argv[2])
c = n = 0
for i in range(ns):
    d = json.load(open(f"eval_out/{tag}.shard{i}.json"))
    c += d["correct"]; n += d["n"]
json.dump({"label": tag, "correct": c, "n": n, "score": c / n}, open(f"eval_out/{tag}.json", "w"), indent=2)
print(f"[sweep] {tag}: {c / n * 100:.1f}%  ({c:.0f}/{n})")
PY
      fi
    done
  done
  vllm_stop "$pid"
  echo "[sweep] canvas=${canvas} block DONE"
}

# Launch the canvas blocks concurrently (one GPU each). Set SEQUENTIAL=1 to serialise.
bpids=()
for canvas in $CANVASES; do
  if [[ "${SEQUENTIAL:-0}" == "1" ]]; then run_canvas_block "$canvas"
  else run_canvas_block "$canvas" & bpids+=($!); fi
done
fail=0
for p in "${bpids[@]:-}"; do [[ -n "$p" ]] && { wait "$p" || fail=1; }; done

echo "==================== [sweep] ALL BLOCKS DONE (fail=${fail}) ===================="
.venv/bin/python local/collect_mathbench.py --md eval_out/RESULTS.md || true
