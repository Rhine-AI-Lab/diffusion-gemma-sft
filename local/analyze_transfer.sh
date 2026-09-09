#!/bin/bash
# Transfer analysis: base model vs the LR-1.5e-4 r1cot checkpoint on MATH-500 (g4096/c256,
# format 6). Reports has_boxed rate, avg output length (vs the SFT dataset target length), and
# correctness conditional on has_boxed. Serves both models in parallel (base=GPU0, ckpt=GPU1).
#
#   bash local/analyze_transfer.sh            # full MATH-500
#   LIMIT=5 bash local/analyze_transfer.sh    # smoke
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
source local/_vllm_lib.sh
mkdir -p logs eval_out
export PYTHONPATH="$REPO" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}" GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}" MAX_DENOISING_STEPS=64

CANVAS=256; GEN="${GEN:-4096}"; MML="${MML:-8192}"; LIMIT="${LIMIT:-0}"; CONC="${CONC:-8}"
BASE_MODEL="google/diffusiongemma-26B-A4B-it"
CKPT_MODEL="$(cd out/merged_lrsweep_r1cot_lr1.5e-4 && pwd)"
BASE_GPU="${BASE_GPU:-0}"; CKPT_GPU="${CKPT_GPU:-1}"

# --- SFT dataset target length (read-only, no GPU) ---
echo "[analyze] computing SFT r1cot dataset completion length ..."
.venv/bin/python - <<'PY'
import json, statistics as st
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("unsloth/diffusiongemma-26B-A4B-it")
lens = [len(tok.encode(json.loads(l)["completion"], add_special_tokens=False))
        for l in open("dataset/openr1_math_sft_r1cot_c4096.jsonl") if l.strip()]
s = sorted(lens)
out = {"n": len(lens), "mean": st.mean(lens), "median": st.median(lens),
       "p90": s[int(0.9*len(s))], "max": s[-1]}
json.dump(out, open("eval_out/analyze_sft_len.json", "w"), indent=2)
print("  SFT r1cot completion tokens:", out)
PY

# --- serve a model, run the client, stop the server ---
run_one() {  # label model gpu port out
  local label="$1" model="$2" gpu="$3" port="$4" out="$5"
  local log="logs/analyze_serve_${label}.log"
  local pid; pid="$(MODEL_ID="$model" vllm_start "$gpu" "$port" "$CANVAS" "$MML" "$log")"
  if ! vllm_wait "$port" "$pid" 1800; then echo "[error] serve ${label} failed"; tail -30 "$log"; return 1; fi
  echo "[analyze] generating: ${label} (GPU ${gpu}, MATH-500, gen ${GEN})"
  .venv/bin/python local/analyze_transfer.py --vllm_base_url "http://localhost:${port}/v1" \
    --vllm_model "$model" --dataset math500 --max_new_tokens "$GEN" --limit "$LIMIT" \
    --concurrency "$CONC" --label "$label" --out "$out" > "logs/analyze_client_${label}.log" 2>&1 \
    || { echo "[error] client ${label} failed"; tail -20 "logs/analyze_client_${label}.log"; }
  vllm_stop "$pid"
  echo "[analyze] ${label} done"
}

run_one base     "$BASE_MODEL" "$BASE_GPU" 8080 eval_out/analyze_base.json     & p1=$!
run_one lr1.5e-4 "$CKPT_MODEL" "$CKPT_GPU" 8081 eval_out/analyze_lr1p5e-4.json & p2=$!
wait "$p1"; wait "$p2"

# --- report ---
echo
echo "================= R1-CoT -> DiffGemma transfer (MATH-500, g${GEN}/c${CANVAS}) ================="
.venv/bin/python - <<'PY'
import json, os
b = json.load(open("eval_out/analyze_base.json"))
c = json.load(open("eval_out/analyze_lr1p5e-4.json"))
sft = json.load(open("eval_out/analyze_sft_len.json"))
def row(name, kb, kc, pct=False, suf=""):
    fb, fc = b[kb], c[kc]
    if pct: print(f"  {name:<26} {fb*100:>8.1f}%   {fc*100:>8.1f}%")
    else:   print(f"  {name:<26} {fb:>9.0f}{suf}   {fc:>9.0f}{suf}")
print(f"  {'metric':<26} {'base':>10}   {'lr1.5e-4':>10}")
print(f"  {'-'*26} {'-'*10}   {'-'*10}")
row("has_boxed rate",        "has_boxed_rate", "has_boxed_rate", pct=True)
row("avg output length (tok)","avg_len", "avg_len")
row("  ..of boxed gens",     "avg_len_boxed", "avg_len_boxed")
row("  ..of unboxed gens",   "avg_len_unboxed", "avg_len_unboxed")
row("acc | has_boxed",       "acc_given_boxed", "acc_given_boxed", pct=True)
row("acc | not boxed",       "acc_given_unboxed", "acc_given_unboxed", pct=True)
row("acc overall (sanity)",  "acc_overall", "acc_overall", pct=True)
print(f"\n  SFT r1cot dataset target length: mean={sft['mean']:.0f}  median={sft['median']:.0f}  "
      f"p90={sft['p90']:.0f}  (tokens; n={sft['n']})")
print(f"  base gen boxed at {b['n_boxed']}/{b['n']}; lr1.5e-4 boxed at {c['n_boxed']}/{c['n']}")
PY
echo "================================================================================================"
