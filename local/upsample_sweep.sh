#!/bin/bash
# Answer-content upsampling sweep. Trains r1cot base-sft (block) at LR 1.5e-5 for each
# answer_content_frac in {0.2,0.3,0.4,0.5} (500 steps, ckpt every 250), then merges
# checkpoint-500 and evals MATH-500 with analyze_transfer.py (has_boxed rate + output length +
# accuracy). Reports target x {measured answer-content%, has_boxed%, avg_len, MATH-500 acc,
# acc|boxed} with base + natural-LR-1.5e-5 baselines. Runs in waves across the free GPUs.
#
#   bash local/upsample_sweep.sh
#   TRAIN_GPUS="0 1 3" bash local/upsample_sweep.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
source local/_vllm_lib.sh
mkdir -p logs out eval_out
export PYTHONPATH="$REPO" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}" GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}" MAX_DENOISING_STEPS=64

LR="${LR:-1.5e-5}"; RESP=r1cot; STEPS="${STEPS:-500}"; SAVE_EVERY="${SAVE_EVERY:-250}"
read -r -a FRACS <<< "${FRACS:-0.2 0.3 0.4 0.5}"   # override: FRACS="0.6 0.7 0.8"
read -r -a GPUS <<< "${TRAIN_GPUS:-0 1 3}"; NG=${#GPUS[@]}
BASE="google/diffusiongemma-26B-A4B-it"
GEN="${GEN:-4096}"; CANVAS=256; MML=8192; CONC="${CONC:-8}"
pct() { awk "BEGIN{printf \"%d\", $1*100}"; }

# ---------------- Phase 1: train in waves ----------------
echo "==== [upsweep] Phase 1: train ${#FRACS[@]} upsample runs (LR ${LR}, ${STEPS} steps) on GPUs ${GPUS[*]} ===="
train_one() {  # frac gpu
  local p="$1" gpu="$2" pc; pc="$(pct "$p")"
  LR="$LR" MAX_STEPS="$STEPS" SAVE_STEPS="$SAVE_EVERY" BLOCK=1 RESPONSES="$RESP" VARIANTS=base-sft \
    GPUS="$gpu" MAIN_PORT="$((29600 + gpu))" OUT_SUFFIX="_lr${LR}_up${pc}" ANSWER_CONTENT_FRAC="$p" \
    REPORT_TO=none bash local/math_sft_sweep.sh > "logs/upsweep_train_up${pc}.log" 2>&1
}
idx=0
while [ "$idx" -lt "${#FRACS[@]}" ]; do
  pids=()
  for ((j=0; j<NG && idx<${#FRACS[@]}; j++, idx++)); do
    echo "  up$(pct "${FRACS[$idx]}")% -> GPU ${GPUS[$j]}"
    train_one "${FRACS[$idx]}" "${GPUS[$j]}" & pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
done
echo "==== [upsweep] Phase 1 done ===="

# ---------------- Phase 2: merge + analyze (MATH-500) in waves ----------------
echo "==== [upsweep] Phase 2: merge + analyze_transfer (MATH-500) ===="
# eval an UPSAMPLED run: merge ck-500 -> serve -> analyze -> delete merged (disk)
eval_up() {  # frac gpu
  local p="$1" gpu="$2" pc; pc="$(pct "$p")"
  local ck="out/openr1_math_sft_${RESP}_base-sft_block_lr${LR}_up${pc}/checkpoint-${STEPS}"
  local merged="out/merged_upsweep_up${pc}" port="$((8080 + gpu))"
  [ -d "$ck" ] || { echo "[upsweep] MISSING $ck"; return 1; }
  CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python local/merge_lora.py "$BASE" "$ck" "$merged" \
    > "logs/upsweep_merge_up${pc}.log" 2>&1 || { echo "[upsweep] merge up${pc} failed"; return 1; }
  local abs pid; abs="$(cd "$merged" && pwd)"
  pid="$(MODEL_ID="$abs" vllm_start "$gpu" "$port" "$CANVAS" "$MML" "logs/upsweep_serve_up${pc}.log")"
  vllm_wait "$port" "$pid" 1800 || { echo "[upsweep] serve up${pc} failed"; tail -20 "logs/upsweep_serve_up${pc}.log"; }
  .venv/bin/python local/analyze_transfer.py --vllm_base_url "http://localhost:${port}/v1" \
    --vllm_model "$abs" --dataset math500 --max_new_tokens "$GEN" --concurrency "$CONC" \
    --label "up${pc}" --out "eval_out/upsweep_up${pc}.json" > "logs/upsweep_client_up${pc}.log" 2>&1 \
    || echo "[upsweep] analyze up${pc} failed"
  vllm_stop "$pid"; rm -rf "$merged"   # checkpoint remains; merged model is regenerable
}
# natural LR-1.5e-5 baseline: analyze the already-merged model (no merge, no delete)
eval_natural() {  # gpu
  local gpu="$1" merged="out/merged_lrsweep_r1cot_lr1.5e-5" port="$((8090 + gpu))"
  [ -d "$merged" ] || { echo "[upsweep] natural baseline model missing"; return 0; }
  local abs pid; abs="$(cd "$merged" && pwd)"
  pid="$(MODEL_ID="$abs" vllm_start "$gpu" "$port" "$CANVAS" "$MML" "logs/upsweep_serve_natural.log")"
  vllm_wait "$port" "$pid" 1800 || { echo "[upsweep] serve natural failed"; }
  .venv/bin/python local/analyze_transfer.py --vllm_base_url "http://localhost:${port}/v1" \
    --vllm_model "$abs" --dataset math500 --max_new_tokens "$GEN" --concurrency "$CONC" \
    --label "natural" --out "eval_out/upsweep_natural.json" > "logs/upsweep_client_natural.log" 2>&1 \
    || echo "[upsweep] analyze natural failed"
  vllm_stop "$pid"
}
# eval set: the 4 upsampled runs + the natural baseline, in waves of NG GPUs
EVALS=(); for _p in "${FRACS[@]}"; do EVALS+=("up:${_p}"); done
[ "${INCLUDE_NATURAL:-0}" = "1" ] && EVALS+=("natural:")
idx=0
while [ "$idx" -lt "${#EVALS[@]}" ]; do
  pids=()
  for ((j=0; j<NG && idx<${#EVALS[@]}; j++, idx++)); do
    kind="${EVALS[$idx]%%:*}"; arg="${EVALS[$idx]#*:}"
    if [ "$kind" = "up" ]; then eval_up "$arg" "${GPUS[$j]}" & else eval_natural "${GPUS[$j]}" & fi
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
done
echo "==== [upsweep] Phase 2 done ===="

# ---------------- Phase 3: report ----------------
echo
echo "============= ANSWER-CONTENT UPSAMPLING SWEEP (r1cot, LR ${LR}, ${STEPS} steps, MATH-500) ============="
PCTS=""; for _p in "${FRACS[@]}"; do PCTS+="$(pct "$_p") "; done
.venv/bin/python - "$LR" "$STEPS" "$PCTS" <<'PY'
import json, os, sys
LR, STEPS, PCTS = sys.argv[1], sys.argv[2], sys.argv[3].split()
def load(p): return json.load(open(p)) if os.path.exists(p) else None
def ac_frac(statsp):  # measured answer-content = transition + answer_only
    s = load(statsp)
    return (s["fracs"]["transition"] + s["fracs"]["answer_only"]) * 100 if s else None
def line(name, target_ac, meas_ac, m):
    if m is None: print(f"  {name:<12} {str(target_ac):>7}  {'-':>9}  {'(no result)':>10}"); return
    ta = f"{target_ac}" if target_ac is not None else "-"
    ma = f"{meas_ac:.1f}%" if meas_ac is not None else "-"
    print(f"  {name:<12} {ta:>7}  {ma:>9}  {m['has_boxed_rate']*100:>8.1f}%  {m['avg_len']:>7.0f}  "
          f"{m['acc_overall']*100:>7.1f}%  {m['acc_given_boxed']*100:>8.1f}%")
print(f"  {'run':<12} {'target':>7}  {'meas.ac%':>9}  {'has_box':>9}  {'avglen':>7}  {'MATH500':>8}  {'acc|box':>9}")
print(f"  {'-'*12} {'-'*7}  {'-'*9}  {'-'*9}  {'-'*7}  {'-'*8}  {'-'*9}")
line("base", None, None, load("eval_out/analyze_base.json"))
nat = load("eval_out/upsweep_natural.json")
if nat:
    line("natural", "12.8%",
         ac_frac("out/openr1_math_sft_r1cot_base-sft_block_lr1.5e-5/canvas_bin_stats.json"), nat)
for p in PCTS:
    line(f"up{p}", f"{p}%",
         ac_frac(f"out/openr1_math_sft_r1cot_base-sft_block_lr{LR}_up{p}/canvas_bin_stats.json"),
         load(f"eval_out/upsweep_up{p}.json"))
PY
echo "======================================================================================================"
