#!/bin/bash
# Canvas-COUNT sweep at long generation lengths (extends the main sweep past 2 canvases).
#   gen 2048, 4 canvases  (canvas 512)   -> g2048/4c   GPU 2
#   gen 2048, 8 canvases  (canvas 256)   -> g2048/8c   GPU 1
#   gen 1024, 4 canvases  (canvas 256)   -> g1024/4c   GPU 4
# #canvases = gen / canvas_length. All canvas <= 512 -> CUDA graphs OK (no enforce-eager).
# Format 6 (chat + reasoning_prefill), 64 denoising steps, corrected scorer.
# Runs fully OFFLINE to PIN the currently-cached tokenizer template (avoid a mid-run
# Hub re-download silently changing the prompt, as happened before). One block per
# (gen,canvas) config, run concurrently, then aggregate into eval_out/RESULTS.md.
#
#   bash local/run_canvas_sweep.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
source local/_vllm_lib.sh
mkdir -p eval_out logs

STEPS="${STEPS:-64}"; export MAX_DENOISING_STEPS="$STEPS"
DATASETS_ALL="${DATASETS:-math500 minerva olympiad amc aime24 aime25}"
AVGK_DATASETS="aime24 aime25"
NSHARDS="${NSHARDS:-4}"; K="${K:-8}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
VLLM_MODEL="${VLLM_MODEL:-google/diffusiongemma-26B-A4B-it}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-$NSHARDS}" GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"
export PYTHONPATH="$REPO"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false  # PIN template
FMT_FLAGS="--reasoning_prefill"   # format 6 = chat (default) + reasoning prefill

# "canvas:gen:gpu:port"
CONFIGS=("512:2048:2:8020" "256:2048:1:8021" "256:1024:4:8022")
is_avgk() { [[ " $AVGK_DATASETS " == *" $1 "* ]]; }

run_block() {   # canvas gen gpu port
  local canvas="$1" gen="$2" gpu="$3" port="$4" log pid ds tag
  log="logs/vllm_csweep_c${canvas}_g${gen}_gpu${gpu}.log"
  local ee=""; [[ "$canvas" -gt 512 ]] && ee=1
  echo "[csweep] serve canvas=${canvas} for gen=${gen} on GPU ${gpu} (:${port}) [$((gen/canvas)) canvases]"
  pid="$(ENFORCE_EAGER="$ee" vllm_start "$gpu" "$port" "$canvas" "$MAX_MODEL_LEN" "$log")"
  if ! vllm_wait "$port" "$pid" 1800; then
    echo "[error] canvas=${canvas} gen=${gen} serve failed; tail ${log}:"; tail -40 "$log"; return 1
  fi
  for ds in $DATASETS_ALL; do
    tag="mathbench_${ds}_g${gen}_c${canvas}"
    if is_avgk "$ds"; then
      echo "[csweep] ${tag}  (avg@${K})"
      .venv/bin/python -u -m DiffGemma.diffgemma_trl.sft_eval_mathbench \
        --backend vllm --vllm_base_url "http://localhost:${port}/v1" --vllm_model "$VLLM_MODEL" \
        --dataset "$ds" --k "$K" --max_new_tokens "$gen" $FMT_FLAGS \
        --result_file "eval_out/${tag}.json" > "logs/${tag}.log" 2>&1 \
        || echo "[warn] ${tag} failed (see logs/${tag}.log)"
    else
      echo "[csweep] ${tag}  (sharded x${NSHARDS})"
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
      [[ "$sfail" -ne 0 ]] && echo "[warn] ${tag}: a shard failed"
      .venv/bin/python - "$tag" "$NSHARDS" <<'PY' || echo "[warn] ${tag}: aggregation failed"
import json, sys
tag, ns = sys.argv[1], int(sys.argv[2]); c = n = 0
for i in range(ns):
    d = json.load(open(f"eval_out/{tag}.shard{i}.json")); c += d["correct"]; n += d["n"]
json.dump({"label": tag, "correct": c, "n": n, "score": c / n}, open(f"eval_out/{tag}.json", "w"), indent=2)
print(f"[csweep] {tag}: {c / n * 100:.1f}%  ({c:.0f}/{n})")
PY
    fi
  done
  vllm_stop "$pid"
  echo "[csweep] canvas=${canvas} gen=${gen} block DONE"
}

bpids=()
for cfg in "${CONFIGS[@]}"; do
  IFS=: read -r canvas gen gpu port <<< "$cfg"
  run_block "$canvas" "$gen" "$gpu" "$port" & bpids+=($!)
done
fail=0; for p in "${bpids[@]}"; do wait "$p" || fail=1; done
echo "==================== [csweep] ALL BLOCKS DONE (fail=${fail}) ===================="
.venv/bin/python local/collect_mathbench.py --md eval_out/RESULTS.md || true
