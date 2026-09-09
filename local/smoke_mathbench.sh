#!/bin/bash
# Smoke test: pick the best prompt format for DiffusionGemma math evals.
#
# On ONE dataset (default MATH-500, first LIMIT problems) run the 2x2 grid of the
# two format ablations at generation length 500:
#     think        in {off, on}   (<|think|> token prepended)
#     reasoning    in {off, on}   (<reasoning> block prefilled into the model turn)
# via a vLLM server with canvas_length=512 (fits 500 tokens in a single canvas).
#
#   GPU=1 bash local/smoke_mathbench.sh
#
# >>> HARD STOP after this: review the 4 accuracies + the winning format, then get
# >>> user go-ahead before local/run_mathbench.sh (the full 42-run sweep).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
source local/_vllm_lib.sh
mkdir -p eval_out logs

GPU="${GPU:-1}"                       # one of the allowed GPUs 1,2,4,5
PORT="${PORT:-8001}"
CANVAS="${CANVAS:-512}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
DATASET="${DATASET:-math500}"
LIMIT="${LIMIT:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-500}"
VLLM_MODEL="${VLLM_MODEL:-google/diffusiongemma-26B-A4B-it}"
export PYTHONPATH="$REPO"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"  # client needs datasets
export TOKENIZERS_PARALLELISM=false

LOG="logs/vllm_smoke_c${CANVAS}_gpu${GPU}.log"
echo "==== [smoke] serving DiffusionGemma canvas=${CANVAS} on GPU ${GPU} (:${PORT}) ===="
SERVER_PID="$(HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 vllm_start "$GPU" "$PORT" "$CANVAS" "$MAX_MODEL_LEN" "$LOG")"
trap 'vllm_stop "$SERVER_PID"' EXIT
vllm_wait "$PORT" "$SERVER_PID" 1800 || { echo "[error] serve failed; tail $LOG:"; tail -40 "$LOG"; exit 1; }

run_fmt() {  # think reasoning
  local think="$1" reason="$2"
  local tag="t${think}_r${reason}" flags=""
  [[ "$think"  == "1" ]] && flags="$flags --think"
  [[ "$reason" == "1" ]] && flags="$flags --reasoning_prefill"
  local out="eval_out/smoke_${DATASET}_${tag}.json"
  echo "-------- [smoke] format ${tag} (think=${think} reasoning_prefill=${reason}) --------"
  .venv/bin/python -u -m DiffGemma.diffgemma_trl.sft_eval_mathbench \
    --backend vllm --vllm_base_url "http://localhost:${PORT}/v1" --vllm_model "$VLLM_MODEL" \
    --dataset "$DATASET" --limit "$LIMIT" --max_new_tokens "$MAX_NEW_TOKENS" \
    --seed 0 --raw_template $flags --result_file "$out" \
    2>&1 | tee "logs/smoke_${DATASET}_${tag}.log"
}

run_fmt 0 0
run_fmt 1 0
run_fmt 0 1
run_fmt 1 1

echo; echo "==================== SMOKE SUMMARY (${DATASET}, n<=${LIMIT}, gen=${MAX_NEW_TOKENS}) ===================="
.venv/bin/python - "$DATASET" <<'PY'
import json, sys, glob, os
ds = sys.argv[1]
rows = []
for f in sorted(glob.glob(f"eval_out/smoke_{ds}_t*_r*.json")):
    d = json.load(open(f))
    name = os.path.basename(f)
    think = "_t1_" in name; reason = "_r1." in name
    score = d.get("score", d.get("mean"))
    rows.append((think, reason, score, name))
rows.sort(key=lambda r: -(r[2] or 0))
print(f"{'think':>6} {'reasoning':>10} {'score':>8}")
for think, reason, score, _ in rows:
    print(f"{int(think):>6} {int(reason):>10} {score*100:>7.1f}%")
if rows:
    b = rows[0]
    print(f"\nBEST FORMAT -> think={int(b[0])} reasoning_prefill={int(b[1])}  ({b[2]*100:.1f}%)")
    print(f"For the full sweep: THINK={int(b[0])} REASONING={int(b[1])} bash local/run_mathbench.sh")
PY
echo "==== [smoke] done. STOP here for review before the full sweep. ===="
