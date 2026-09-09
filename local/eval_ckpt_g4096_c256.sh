#!/bin/bash
# Eval merged SFT checkpoints on the math-benchmark suite at gen 4096 / canvas 256
# (16 canvases), format 6 (chat + reasoning_prefill), 64 denoising steps/canvas -- the
# same methodology as local/run_g4096_c256.sh, but for LoRA checkpoints MERGED into full
# models (the diffusion vLLM fork does not support LoRA adapters directly; see local/merge_lora.py).
#
# Each checkpoint is served on the given GPU pair (one vLLM server per GPU), with the 6
# datasets split across the two servers; checkpoints run sequentially. Results:
#   eval_out/mathbench_<tag>_<ds>_g4096_c256.json
#
#   CKPTS="solution_ck1000:out/merged_solution_ck1000 r1cot_ck1000:out/merged_r1cot_ck1000" \
#     GPUS="2 3" bash local/eval_ckpt_g4096_c256.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
source local/_vllm_lib.sh
mkdir -p eval_out logs

GEN="${GEN:-4096}"; CANVAS="${CANVAS:-256}"
STEPS="${STEPS:-64}"; export MAX_DENOISING_STEPS="$STEPS"
NSHARDS="${NSHARDS:-4}"; K="${K:-8}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-$NSHARDS}" GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"
export PYTHONPATH="$REPO"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false  # PIN template
FMT_FLAGS="--reasoning_prefill"          # format 6 = chat (default) + reasoning prefill
AVGK_DATASETS="aime24 aime25"
is_avgk() { [[ " $AVGK_DATASETS " == *" $1 "* ]]; }

CKPTS="${CKPTS:-solution_ck1000:out/merged_solution_ck1000 r1cot_ck1000:out/merged_r1cot_ck1000}"
GPUS="${GPUS:-2 3}"; read -r -a GPU_ARR <<< "$GPUS"
BASE_PORT="${BASE_PORT:-8050}"
DATASETS="${DATASETS:-math500 minerva olympiad amc aime24 aime25}"  # subset override (e.g. "math500")
read -r -a DS_ARR <<< "$DATASETS"; NGPU=${#GPU_ARR[@]}

# one server on <gpu> serving <model>, then run the given datasets against it
serve_and_eval() {  # tag model gpu port datasets...
  local tag="$1" model="$2" gpu="$3" port="$4"; shift 4; local dss="$*"
  local abs; abs="$(cd "$model" && pwd)"      # absolute path: served name == client model name
  local log="logs/vllm_${tag}_gpu${gpu}.log" pid ds rtag
  echo "[eval] serve ${tag} canvas=${CANVAS} on GPU ${gpu} (:${port})  [$((GEN/CANVAS)) canvases]  for: ${dss}"
  pid="$(MODEL_ID="$abs" vllm_start "$gpu" "$port" "$CANVAS" "$MAX_MODEL_LEN" "$log")"
  if ! vllm_wait "$port" "$pid" 1800; then
    echo "[error] GPU ${gpu} serve failed; tail ${log}:"; tail -40 "$log"; return 1
  fi
  for ds in $dss; do
    rtag="mathbench_${tag}_${ds}_g${GEN}_c${CANVAS}"
    if is_avgk "$ds"; then
      echo "[eval] ${rtag}  (avg@${K}) on GPU ${gpu}"
      .venv/bin/python -u -m DiffGemma.diffgemma_trl.sft_eval_mathbench \
        --backend vllm --vllm_base_url "http://localhost:${port}/v1" --vllm_model "$abs" \
        --dataset "$ds" --k "$K" --max_new_tokens "$GEN" $FMT_FLAGS \
        --result_file "eval_out/${rtag}.json" > "logs/${rtag}.log" 2>&1 \
        || echo "[warn] ${rtag} failed (see logs/${rtag}.log)"
    else
      echo "[eval] ${rtag}  (sharded x${NSHARDS}) on GPU ${gpu}"
      local spids=() sid
      for sid in $(seq 0 $((NSHARDS - 1))); do
        rm -f "eval_out/${rtag}.shard${sid}.json"
        .venv/bin/python -u -m DiffGemma.diffgemma_trl.sft_eval_mathbench \
          --backend vllm --vllm_base_url "http://localhost:${port}/v1" --vllm_model "$abs" \
          --dataset "$ds" --max_new_tokens "$GEN" $FMT_FLAGS --seed 0 \
          --num_shards "$NSHARDS" --shard_id "$sid" \
          --result_file "eval_out/${rtag}.shard${sid}.json" > "logs/${rtag}.shard${sid}.log" 2>&1 &
        spids+=($!)
      done
      local sfail=0 p; for p in "${spids[@]}"; do wait "$p" || sfail=1; done
      [[ "$sfail" -ne 0 ]] && echo "[warn] ${rtag}: a shard failed"
      .venv/bin/python - "$rtag" "$NSHARDS" <<'PY' || echo "[warn] ${rtag}: aggregation failed"
import json, sys
tag, ns = sys.argv[1], int(sys.argv[2]); c = n = 0
for i in range(ns):
    d = json.load(open(f"eval_out/{tag}.shard{i}.json")); c += d["correct"]; n += d["n"]
json.dump({"label": tag, "correct": c, "n": n, "score": c / n}, open(f"eval_out/{tag}.json", "w"), indent=2)
print(f"[eval] {tag}: {c / n * 100:.1f}%  ({c:.0f}/{n})")
PY
    fi
  done
  vllm_stop "$pid"
  echo "[eval] ${tag} GPU ${gpu} block DONE"
}

for entry in $CKPTS; do
  tag="${entry%%:*}"; model="${entry#*:}"
  if [[ ! -d "$model" ]]; then echo "[skip] $tag: $model not found"; continue; fi
  echo "==================== [eval] ${tag}  ($model)  datasets='${DATASETS}' on GPUs ${GPUS} ===================="
  # round-robin the datasets across the given GPUs (one vLLM server per GPU)
  declare -a G2DS; for i in "${!GPU_ARR[@]}"; do G2DS[$i]=""; done
  di=0; for ds in "${DS_ARR[@]}"; do gi=$((di % NGPU)); G2DS[$gi]="${G2DS[$gi]} $ds"; di=$((di+1)); done
  pids=()
  for i in "${!GPU_ARR[@]}"; do
    [[ -z "${G2DS[$i]// }" ]] && continue
    serve_and_eval "$tag" "$model" "${GPU_ARR[$i]}" "$((BASE_PORT + i))" ${G2DS[$i]} & pids+=($!)
  done
  f=0; for p in "${pids[@]}"; do wait "$p" || f=1; done
  echo "==================== [eval] ${tag} DONE (fail=${f}) ===================="
done
echo "==================== [eval] ALL CHECKPOINTS DONE ===================="
for entry in $CKPTS; do
  tag="${entry%%:*}"
  for ds in $DATASETS; do
    f="eval_out/mathbench_${tag}_${ds}_g${GEN}_c${CANVAS}.json"
    [[ -f "$f" ]] && .venv/bin/python -c "import json;d=json.load(open('$f'));print(f\"  {'$tag':22s} {'$ds':10s} {d['score']*100:5.1f}%  ({d['correct']:.0f}/{d['n']})\")"
  done
done