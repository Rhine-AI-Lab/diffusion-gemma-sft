#!/bin/bash
# Prompt-format ablation (corrected math_verify scorer) on ONE dataset at n=LIMIT.
# Runs all 7 formats against a single vLLM server (canvas 512, STEPS denoising steps):
#   raw route  : t0_r0, t1_r0, t0_r1, t1_r1   (<|think|> x <reasoning> prefill)
#   chat route : et0_rp0 (5), et0_rp1 (6, default), et1_rp0 (7)
#
#   GPU=1 STEPS=64 LIMIT=100 bash local/ablation_mathbench.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
source local/_vllm_lib.sh
mkdir -p eval_out logs

GPU="${GPU:-1}"; PORT="${PORT:-8001}"; CANVAS="${CANVAS:-512}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
STEPS="${STEPS:-64}"; DATASET="${DATASET:-math500}"; LIMIT="${LIMIT:-100}"; MNT="${MNT:-500}"
VLLM_MODEL="${VLLM_MODEL:-google/diffusiongemma-26B-A4B-it}"
export PYTHONPATH="$REPO"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false

LOG="logs/vllm_ablation_c${CANVAS}_s${STEPS}_gpu${GPU}.log"
echo "==== [ablation] serve canvas=${CANVAS} steps=${STEPS} on GPU ${GPU} (:${PORT}) ===="
PID="$(HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MAX_DENOISING_STEPS="$STEPS" \
  vllm_start "$GPU" "$PORT" "$CANVAS" "$MAX_MODEL_LEN" "$LOG")"
trap 'vllm_stop "$PID"' EXIT
vllm_wait "$PORT" "$PID" 1800 || { echo "[error] serve failed; tail $LOG:"; tail -40 "$LOG"; exit 1; }
grep -o "'max_denoising_steps': [0-9]*" "$LOG" | head -1

run() {  # label  flag...
  local label="$1"; shift
  local out="eval_out/ablation_${DATASET}_${label}_s${STEPS}.json"
  echo "-------- [ablation] ${label}  (flags: $*) --------"
  .venv/bin/python -u -m DiffGemma.diffgemma_trl.sft_eval_mathbench \
    --backend vllm --vllm_base_url "http://localhost:${PORT}/v1" --vllm_model "$VLLM_MODEL" \
    --dataset "$DATASET" --limit "$LIMIT" --max_new_tokens "$MNT" --seed 0 "$@" \
    --result_file "$out" 2>&1 | tee "logs/ablation_${DATASET}_${label}.log"
}

# raw <|turn> route
run raw_t0_r0 --raw_template
run raw_t1_r0 --raw_template --think
run raw_t0_r1 --raw_template --reasoning_prefill
run raw_t1_r1 --raw_template --think --reasoning_prefill
# native apply_chat_template route
run chat_et0_rp0
run chat_et0_rp1 --reasoning_prefill
run chat_et1_rp0 --enable_thinking

echo; echo "==================== ABLATION SUMMARY (${DATASET}, n<=${LIMIT}, gen=${MNT}, steps=${STEPS}, corrected scorer) ===================="
.venv/bin/python - "$DATASET" "$STEPS" <<'PY'
import json, sys, glob, os
ds, steps = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(glob.glob(f"eval_out/ablation_{ds}_*_s{steps}.json")):
    d = json.load(open(f))
    label = os.path.basename(f).replace(f"ablation_{ds}_","").replace(f"_s{steps}.json","")
    rows.append((label, d.get("score", d.get("mean"))))
rows.sort(key=lambda r: -(r[1] or 0))
print(f"{'format':14} {'score':>7}")
for label, sc in rows:
    print(f"{label:14} {sc*100:>6.1f}%")
if rows:
    print(f"\nBEST -> {rows[0][0]}  ({rows[0][1]*100:.1f}%)")
PY
echo "==== [ablation] done. ===="
