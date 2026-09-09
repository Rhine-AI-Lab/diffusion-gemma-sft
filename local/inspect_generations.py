"""Generate a few raw samples from a merged checkpoint to eyeball coherence.
  python local/inspect_generations.py <merged_model_dir> <dataset> <n>
"""
import sys

from DiffGemma.diffgemma_trl.sft_eval_common import LocalBackend, load_local_model
from DiffGemma.diffgemma_trl.sft_eval_mathbench import _apply_chat, _load_rows
from DiffGemma.diffgemma_trl.sft_data_utils import MATH_INSTRUCTION

model_dir, ds_name, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
print(f"[inspect] model={model_dir} dataset={ds_name} n={n}", flush=True)

model, tok = load_local_model(model_dir)
be = LocalBackend(model, tok, max_denoising_steps=64, batch_size=n, add_special_tokens=False)

rows = _load_rows(ds_name)[:n]
_apply_chat(rows, tok, enable_thinking=False, instruction=MATH_INSTRUCTION, reasoning_prefill=True)
outs = be.generate([r["_prompt"] for r in rows], max_new_tokens=4096)

for i, (r, o) in enumerate(zip(rows, outs)):
    has_box = "\\boxed{" in o
    print(f"\n{'='*90}\nPROBLEM {i}  gold_answer={r['answer']!r}  | out_len={len(o)} chars | has_boxed={has_box}\n{'-'*90}")
    if len(o) <= 3200:
        print(o)
    else:
        print(o[:2200])
        print(f"\n   ...[middle {len(o) - 3000} chars elided]...\n")
        print(o[-800:])
print("\n[inspect] DONE", flush=True)
