#!/bin/bash
# Single extra sweep point: gen 4096, canvas 256  => 16 canvases  (g4096/16c).
# Extends local/run_canvas_sweep.sh (which stops at g2048/8c) with a longer-generation
# column, using the SAME methodology: format 6 (chat + reasoning_prefill), 64 denoising
# steps/canvas, offline-pinned tokenizer template, sharded pass@1 for non-AIME and
# avg@k for AIME. To cut wall-clock, the 6 datasets are spread over the free GPUs
# (one vLLM server per GPU) instead of one server doing everything.
#
#   bash local/run_g4096_c256.sh
#   GPUS="0 1 2 3" bash local/run_g4096_c256.sh          # choose GPUs
#
# Result files: eval_out/mathbench_<ds>_g4096_c256.json  (auto-picked up by
# local/collect_mathbench.py -> a new g4096/16c column in RESULTS.md).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
source local/_vllm_lib.sh
mkdir -p eval_out logs

GEN="${GEN:-4096}"; CANVAS="${CANVAS:-256}"
STEPS="${STEPS:-64}"; export MAX_DENOISING_STEPS="$STEPS"
NSHARDS="${NSHARDS:-4}"; K="${K:-8}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
VLLM_MODEL="${VLLM_MODEL:-google/diffusiongemma-26B-A4B-it}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-$NSHARDS}" GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"
export PYTHONPATH="$REPO"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false  # PIN template
FMT_FLAGS="--reasoning_prefill"          # format 6 = chat (default) + reasoning prefill
AVGK_DATASETS="aime24 aime25"
is_avgk() { [[ " $AVGK_DATASETS " == *" $1 "* ]]; }

# GPU -> datasets. AIME sets are serial (avg@k, not sharded) so each gets its own GPU;
# math500 (largest sharded set) gets one; the mid-size sharded sets share the last.
GPUS="${GPUS:-0 1 2 3}"; read -r -a GPU_ARR <<< "$GPUS"
declare -A GPU_DS
GPU_DS[${GPU_ARR[0]}]="math500"
GPU_DS[${GPU_ARR[1]}]="minerva olympiad amc"
GPU_DS[${GPU_ARR[2]}]="aime24"
GPU_DS[${GPU_ARR[3]}]="aime25"
BASE_PORT="${BASE_PORT:-8030}"

run_gpu() {   # gpu port datasets...
  local gpu="$1" port="$2"; shift 2; local dss="$*"
  local log="logs/vllm_g${GEN}_c${CANVAS}_gpu${gpu}.log" pid ds tag
  echo "[g4096] serve canvas=${CANVAS} on GPU ${gpu} (:${port}) for: ${dss}  [$((GEN/CANVAS)) canvases]"
  pid="$(vllm_start "$gpu" "$port" "$CANVAS" "$MAX_MODEL_LEN" "$log")"
  if ! vllm_wait "$port" "$pid" 1800; then
    echo "[error] GPU ${gpu} serve failed; tail ${log}:"; tail -40 "$log"; return 1
  fi
  for ds in $dss; do
    tag="mathbench_${ds}_g${GEN}_c${CANVAS}"
    if is_avgk "$ds"; then
      echo "[g4096] ${tag}  (avg@${K}) on GPU ${gpu}"
      .venv/bin/python -u -m DiffGemma.diffgemma_trl.sft_eval_mathbench \
        --backend vllm --vllm_base_url "http://localhost:${port}/v1" --vllm_model "$VLLM_MODEL" \
        --dataset "$ds" --k "$K" --max_new_tokens "$GEN" $FMT_FLAGS \
        --result_file "eval_out/${tag}.json" > "logs/${tag}.log" 2>&1 \
        || echo "[warn] ${tag} failed (see logs/${tag}.log)"
    else
      echo "[g4096] ${tag}  (sharded x${NSHARDS}) on GPU ${gpu}"
      local spids=() sid
      for sid in $(seq 0 $((NSHARDS - 1))); do
        rm -f "eval_out/${tag}.shard${sid}.json"
        .venv/bin/python -u -m DiffGemma.diffgemma_trl.sft_eval_mathbench \
          --backend vllm --vllm_base_url "http://localhost:${port}/v1" --vllm_model "$VLLM_MODEL" \
          --dataset "$ds" --max_new_tokens "$GEN" $FMT_FLAGS --seed 0 \
          --num_shards "$NSHARDS" --shard_id "$sid" \
          --result_file "eval_out/${tag}.shard${sid}.json" > "logs/${tag}.shard${sid}.log" 2>&1 &
        spids+=($!)
      done
      local sfail=0 p; for p in "${spids[@]}"; do wait "$p" || sfail=1; done
      [[ "$sfail" -ne 0 ]] && echo "[warn] ${tag}: a shard failed"
      .venv/bin/python - "$tag" "$NSHARDS" <<'PY' || echo "[warn] ${tag}: aggregation failed"
import json, sys
tag, ns = sys.argv[1], int(sys.argv[2]); c = n = 0
for i in range(ns):
    d = json.load(open(f"eval_out/{tag}.shard{i}.json")); c += d["correct"]; n += d["n"]
json.dump({"label": tag, "correct": c, "n": n, "score": c / n}, open(f"eval_out/{tag}.json", "w"), indent=2)
print(f"[g4096] {tag}: {c / n * 100:.1f}%  ({c:.0f}/{n})")
PY
    fi
  done
  vllm_stop "$pid"
  echo "[g4096] GPU ${gpu} block DONE"
}

bpids=(); i=0
for gpu in "${GPU_ARR[@]}"; do
  ds="${GPU_DS[$gpu]:-}"; [[ -z "$ds" ]] && continue
  run_gpu "$gpu" "$((BASE_PORT + i))" $ds & bpids+=($!)
  i=$((i + 1))
done
fail=0; for p in "${bpids[@]}"; do wait "$p" || fail=1; done
echo "==================== [g4096] ALL BLOCKS DONE (fail=${fail}) ===================="
.venv/bin/python local/collect_mathbench.py --md eval_out/RESULTS.md || true
