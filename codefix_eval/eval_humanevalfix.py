"""Eval a served DiffusionGemma checkpoint on HumanEvalFix (bigcode/humanevalpack,
python config): generate one fix per buggy problem, extract the code, run the
problem's real test suite, report pass rate.

Held out from training -- codefix_eval/build_bugfix_data.py sources from MBPP,
entirely disjoint from HumanEval.
"""

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

_TIMEOUT_S = 10

_PROMPT_TEMPLATE = """Fix the bug in the following Python function.

Task: {text}

Buggy code:
```python
{buggy_code}
```

Write the complete corrected function."""


def _extract_code(text: str) -> str:
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    return text


def _run_test(code: str, test: str, test_setup: str, entry_point: str) -> bool:
    script = (
        (test_setup or "")
        + "\n"
        + code
        + "\n"
        + test
        + f"\ncheck({entry_point})\n"
    )
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
    ap.add_argument("--max_tokens", type=int, default=600)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="codefix_eval/humanevalfix_results.jsonl")
    args = ap.parse_args()

    ds = datasets.load_dataset("bigcode/humanevalpack", "python", split="test")
    if args.limit:
        ds = ds.select(range(args.limit))

    n_pass = 0
    results = []
    for ex in ds:
        buggy = ex["declaration"] + ex["buggy_solution"]
        text = ex["docstring"].strip() if ex.get("docstring") else ex["instruction"]
        prompt = _PROMPT_TEMPLATE.format(text=text, buggy_code=buggy)
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
            # see codefix_eval/eval_bcb.py: the reasoning-parser can route a
            # full valid completion into `reasoning` instead of `content`.
            raw = msg.get("content") or msg.get("reasoning") or ""
        except Exception as e:
            raw = ""
            print(f"{ex['task_id']}: REQUEST ERROR {e}")

        code = _extract_code(raw)
        passed = _run_test(code, ex["test"], ex.get("test_setup", ""), ex["entry_point"])
        n_pass += passed
        results.append({"task_id": ex["task_id"], "bug_type": ex["bug_type"], "passed": passed, "raw": raw})
        print(f"{ex['task_id']:20s} bug_type={ex['bug_type']:16s} passed={passed}")

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n=== {n_pass}/{len(ds)} passed ({100*n_pass/len(ds):.1f}%) ===")
    from collections import Counter, defaultdict
    by_type = defaultdict(lambda: [0, 0])
    for r in results:
        by_type[r["bug_type"]][1] += 1
        by_type[r["bug_type"]][0] += r["passed"]
    for bt, (p, n) in sorted(by_type.items()):
        print(f"  {bt:16s}: {p}/{n}")


if __name__ == "__main__":
    main()
