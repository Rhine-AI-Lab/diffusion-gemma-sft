"""Build a train/eval split of BigCodeBench (bigcode/bigcodebench, v0.1.4),
filtered to problems whose `libs` are all importable in this venv (so tests
can actually run), and emit SFT training pairs in the flat prompt/completion
format (DiffGemma/diffgemma_trl/sft_data_utils.py, prompt_style=bugfix -- a
bare user-turn wrapper, reused as-is since it's generic).

Eval indices are saved separately and MUST stay disjoint from training --
eval_bcb.py loads them directly, not a fresh random sample, so the same
held-out set is used consistently across checkpoints.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import random

import datasets

_PROMPT_TEMPLATE = "{instruction}\n\nComplete this function:\n```python\n{code_prompt}\n```"


def _satisfiable_indices(ds) -> list[int]:
    cache: dict[str, bool] = {}

    def importable(lib: str) -> bool:
        top = lib.split(".")[0]
        if top not in cache:
            try:
                importlib.import_module(top)
                cache[top] = True
            except Exception:
                cache[top] = False
        return cache[top]

    ok = []
    for i, ex in enumerate(ds):
        libs = ast.literal_eval(ex["libs"])
        if all(importable(l) for l in libs):
            ok.append(i)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_eval", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_train", default="codefix_eval/bcb_sft.jsonl")
    ap.add_argument("--out_eval_idx", default="codefix_eval/bcb_eval_idx.json")
    args = ap.parse_args()

    ds = datasets.load_dataset("bigcode/bigcodebench", split="v0.1.4")
    ok_idxs = _satisfiable_indices(ds)
    print(f"{len(ok_idxs)}/{len(ds)} problems have satisfiable deps")

    random.seed(args.seed)
    shuffled = ok_idxs[:]
    random.shuffle(shuffled)
    eval_idxs = sorted(shuffled[: args.n_eval])
    train_idxs = sorted(shuffled[args.n_eval :])
    assert set(eval_idxs).isdisjoint(train_idxs)
    print(f"train: {len(train_idxs)}  eval (held out): {len(eval_idxs)}")

    with open(args.out_eval_idx, "w") as f:
        json.dump(eval_idxs, f)

    n = 0
    with open(args.out_train, "w") as f:
        for i in train_idxs:
            ex = ds[i]
            prompt = _PROMPT_TEMPLATE.format(
                instruction=ex["instruct_prompt"], code_prompt=ex["code_prompt"]
            )
            completion = f"```python\n{ex['code_prompt']}{ex['canonical_solution']}\n```"
            f.write(
                json.dumps(
                    {"prompt": prompt, "completion": completion, "task_id": ex["task_id"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    print(f"wrote {n} training pairs -> {args.out_train}")
    print(f"wrote {len(eval_idxs)} held-out eval indices -> {args.out_eval_idx}")


if __name__ == "__main__":
    main()
