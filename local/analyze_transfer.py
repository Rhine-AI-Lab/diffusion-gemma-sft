"""Per-example transfer analysis of a served DiffusionGemma model on a math dataset.

Records, per generation: has_boxed, output token length, and correctness (reusing the
math eval scorer). Aggregates has_boxed rate, avg length (overall / boxed / unboxed) and
correctness conditional on has_boxed. Writes a metrics JSON.

  python local/analyze_transfer.py --vllm_base_url http://localhost:8080/v1 \
      --vllm_model google/diffusiongemma-26B-A4B-it --label base --out eval_out/analyze_base.json
"""
import argparse
import concurrent.futures as cf
import json

from openai import OpenAI
from transformers import AutoTokenizer

from DiffGemma.diffgemma_trl.sft_data_utils import MATH_INSTRUCTION
from DiffGemma.diffgemma_trl.sft_eval_math import _score_math
from DiffGemma.diffgemma_trl.sft_eval_mathbench import _apply_chat, _load_rows

ap = argparse.ArgumentParser()
ap.add_argument("--vllm_base_url", required=True)
ap.add_argument("--vllm_model", required=True)
ap.add_argument("--dataset", default="math500")
ap.add_argument("--max_new_tokens", type=int, default=4096)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--concurrency", type=int, default=8)
ap.add_argument("--tokenizer", default="unsloth/diffusiongemma-26B-A4B-it")
ap.add_argument("--label", default="model")
ap.add_argument("--out", required=True)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.tokenizer)
rows = _load_rows(args.dataset)
if args.limit:
    rows = rows[: args.limit]
# format 6: chat + reasoning_prefill, no <think> (same as SFT + all prior evals)
_apply_chat(rows, tok, enable_thinking=False, instruction=MATH_INSTRUCTION, reasoning_prefill=True)

client = OpenAI(base_url=args.vllm_base_url, api_key="EMPTY")


def _gen(idx_row):
    i, row = idx_row
    resp = client.completions.create(
        model=args.vllm_model, prompt=row["_prompt"], max_tokens=args.max_new_tokens,
        temperature=0.0, seed=0, extra_body={"add_special_tokens": False})
    return i, resp.choices[0].text


texts = [""] * len(rows)
done = 0
with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
    for i, t in ex.map(_gen, list(enumerate(rows))):
        texts[i] = t
        done += 1
        if done % 25 == 0:
            print(f"  [{args.label}] {done}/{len(rows)}", flush=True)

recs = []
for row, t in zip(rows, texts):
    recs.append({
        "has_boxed": ("\\boxed{" in t),
        "n_tokens": len(tok.encode(t, add_special_tokens=False)),
        "correct": float(_score_math(t, row)),
    })

boxed = [r for r in recs if r["has_boxed"]]
unboxed = [r for r in recs if not r["has_boxed"]]
mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
metrics = {
    "label": args.label, "dataset": args.dataset, "n": len(recs),
    "has_boxed_rate": mean([r["has_boxed"] for r in recs]),
    "avg_len": mean([r["n_tokens"] for r in recs]),
    "avg_len_boxed": mean([r["n_tokens"] for r in boxed]),
    "avg_len_unboxed": mean([r["n_tokens"] for r in unboxed]),
    "acc_overall": mean([r["correct"] for r in recs]),
    "acc_given_boxed": mean([r["correct"] for r in boxed]),
    "acc_given_unboxed": mean([r["correct"] for r in unboxed]),
    "n_boxed": len(boxed), "n_unboxed": len(unboxed),
}
json.dump(metrics, open(args.out, "w"), indent=2)
print(json.dumps(metrics, indent=2))
