"""Eval a served DiffusionGemma checkpoint on the held-out BigCodeBench split
(indices from codefix_eval/bcb_eval_idx.json -- MUST stay disjoint from
codefix_eval/bcb_sft.jsonl's training indices, see build_bcb_data.py)."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import datasets

_TIMEOUT_S = 15

_PROMPT_TEMPLATE = "{instruction}\n\nComplete this function:\n```python\n{code_prompt}\n```"


def _extract_code(text: str) -> str:
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    return text


def _run_test(code: str, test: str) -> bool:
    script = code + "\n" + test + "\nimport unittest\nunittest.main()\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        r = subprocess.run(
            [sys.executable, path], capture_output=True, timeout=_TIMEOUT_S, text=True
        )
        return r.returncode == 0
    except Exception:
        return False
    finally:
        Path(path).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_base", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="diffgemma")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=800)
    ap.add_argument("--eval_idx", default="codefix_eval/bcb_eval_idx.json")
    ap.add_argument("--out", default="codefix_eval/bcb_results.jsonl")
    args = ap.parse_args()

    eval_idxs = json.load(open(args.eval_idx))
    ds = datasets.load_dataset("bigcode/bigcodebench", split="v0.1.4")

    n_pass = 0
    results = []
    for j, i in enumerate(eval_idxs):
        ex = ds[i]
        prompt = _PROMPT_TEMPLATE.format(
            instruction=ex["instruct_prompt"], code_prompt=ex["code_prompt"]
        )
        body = json.dumps(
            {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }
        ).encode()
        try:
            out = json.load(
                urllib.request.urlopen(
                    urllib.request.Request(
                        f"{args.api_base}/chat/completions",
                        body,
                        {"Content-Type": "application/json"},
                    ),
                    timeout=120,
                )
            )
            msg = out["choices"][0]["message"]
            # DiffusionGemma's reasoning-parser sometimes routes a full, valid
            # completion into `reasoning` instead of `content` (observed on a
            # less-trained checkpoint that hadn't yet learned whatever signal
            # separates the two channels) -- fall back rather than silently
            # score it as empty/failed.
            raw = msg.get("content") or msg.get("reasoning") or ""
        except Exception as e:
            raw = ""
            print(f"{ex['task_id']}: REQUEST ERROR {e}")

        code = _extract_code(raw)
        passed = _run_test(code, ex["test"])
        n_pass += passed
        results.append({"task_id": ex["task_id"], "passed": passed})
        print(f"[{j+1}/{len(eval_idxs)}] {ex['task_id']:20s} passed={passed}", flush=True)

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n=== {n_pass}/{len(eval_idxs)} passed ({100*n_pass/len(eval_idxs):.1f}%) ===")


if __name__ == "__main__":
    main()
