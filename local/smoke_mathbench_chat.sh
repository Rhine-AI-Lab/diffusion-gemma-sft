#!/bin/bash
# Smoke test using the model's NATIVE apply_chat_template (thinking channel) instead
# of the raw <|turn> MATH_PROMPT. Two cells on MATH-500 (n<=LIMIT, gen 500):
#     enable_thinking=0  -> default native mode: model turn primed with
#                           <|channel>thought<channel|> (channel reasoning)
#     enable_thinking=1  -> <|think|> system-turn mode (no channel priming)
# Compare against the raw-format smoke (local/smoke_mathbench.sh).
#
#   GPU=2 PORT=8002 STEPS=64 bash local/smoke_mathbench_chat.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
source local/_vllm_lib.sh
mkdir -p eval_out logs

GPU="${GPU:-2}"                       # one of the allowed GPUs 1,2,4,5
PORT="${PORT:-8002}"
CANVAS="${CANVAS:-512}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
STEPS="${STEPS:-64}"                  # max denoising steps (serve-time override)
DATASET="${DATASET:-math500}"
LIMIT="${LIMIT:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-500}"
VLLM_MODEL="${VLLM_MODEL:-google/diffusiongemma-26B-A4B-it}"
export PYTHONPATH="$REPO"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false

LOG="logs/vllm_smokechat_c${CANVAS}_s${STEPS}_gpu${GPU}.log"
echo "==== [chat-smoke] serve canvas=${CANVAS} steps=${STEPS} on GPU ${GPU} (:${PORT}) ===="
SERVER_PID="$(HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MAX_DENOISING_STEPS="$STEPS" \
  vllm_start "$GPU" "$PORT" "$CANVAS" "$MAX_MODEL_LEN" "$LOG")"
trap 'vllm_stop "$SERVER_PID"' EXIT
vllm_wait "$PORT" "$SERVER_PID" 1800 || { echo "[error] serve failed; tail $LOG:"; tail -40 "$LOG"; exit 1; }
grep -o "'max_denoising_steps': [0-9]*" "$LOG" | head -1

run_chat() {  # enable_thinking(0|1)
  local et="$1"
  local tag="chat_et${et}_s${STEPS}" flags="--chat_template"
  [[ "$et" == "1" ]] && flags="$flags --enable_thinking"
  local out="eval_out/smoke_${DATASET}_${tag}.json"
  echo "-------- [chat-smoke] enable_thinking=${et} (steps=${STEPS}) --------"
  .venv/bin/python -u -m DiffGemma.diffgemma_trl.sft_eval_mathbench \
    --backend vllm --vllm_base_url "http://localhost:${PORT}/v1" --vllm_model "$VLLM_MODEL" \
    --dataset "$DATASET" --limit "$LIMIT" --max_new_tokens "$MAX_NEW_TOKENS" \
    --seed 0 $flags --result_file "$out" \
    2>&1 | tee "logs/smoke_${DATASET}_${tag}.log"
}

run_chat 0
run_chat 1

echo; echo "==================== CHAT-SMOKE SUMMARY (${DATASET}, n<=${LIMIT}, gen=${MAX_NEW_TOKENS}, steps=${STEPS}) ===================="
.venv/bin/python - "$DATASET" "$STEPS" <<'PY'
import json, sys, glob, os
ds, steps = sys.argv[1], sys.argv[2]
print(f"{'enable_thinking':>15} {'score':>8}")
best = None
for f in sorted(glob.glob(f"eval_out/smoke_{ds}_chat_et*_s{steps}.json")):
    d = json.load(open(f)); et = "1" if "_et1_" in os.path.basename(f) else "0"
    sc = d.get("score", d.get("mean"))
    print(f"{et:>15} {sc*100:>7.1f}%")
    if best is None or sc > best[1]: best = (et, sc)
if best:
    print(f"\nBEST (chat) -> enable_thinking={best[0]}  ({best[1]*100:.1f}%)")
print("Compare vs raw-format smoke: eval_out/smoke_math500_steps48_*.json (48 steps) and smoke_math500_t*_r*.json")
PY
echo "==== [chat-smoke] done. ===="
