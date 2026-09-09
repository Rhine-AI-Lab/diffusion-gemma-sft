"""Build execution-tracing SFT data (predict exact output of f(args)) from
MBPP's verified-correct, single-function solutions -- disjoint from
CRUXEval (whose problems are LLM-generated/filtered, not sourced from MBPP),
which stays untouched as the held-out eval set.

For each MBPP problem: keep only single-function, no-class, deterministic
solutions; rename the function to `f` (matching CRUXEval's convention);
extract each test-case call expression; actually EXECUTE it to get the real,
unambiguous output (repr'd, matching CRUXEval's output-field format).
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import datasets

_TIMEOUT_S = 5
_NONDETERMINISTIC = ("random", "time", "datetime", "uuid", "os.urandom")

_PROMPT_TEMPLATE = """Here is a Python function and a call to it:

```python
{code}
```

f{args}

What is the exact output of this call? Respond with ONLY the output value, no explanation."""


def _is_clean_single_function(code: str) -> tuple[bool, str | None]:
    """Return (ok, function_name) if code is exactly one deterministic,
    class-free function definition."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, None
    top = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    fns = [n for n in top if isinstance(n, ast.FunctionDef)]
    others = [n for n in top if not isinstance(n, ast.FunctionDef)]
    if len(fns) != 1 or others:
        return False, None
    if any(nd for nd in ast.walk(tree) if isinstance(nd, ast.ClassDef)):
        return False, None
    if any(mod in code for mod in _NONDETERMINISTIC):
        return False, None
    return True, fns[0].name


def _extract_calls(test_list: list[str], fn_name: str) -> list[str]:
    """Pull the `fn_name(args)` call expression out of each assert line."""
    calls = []
    pat = re.compile(rf"\b{re.escape(fn_name)}\s*\(.*?\)")
    for t in test_list:
        m = pat.search(t)
        if m:
            calls.append(m.group(0))
    return calls


def _rename_fn(code: str, call: str, fn_name: str) -> tuple[str, str]:
    pat = re.compile(rf"\b{re.escape(fn_name)}\b")
    return pat.sub("f", code), pat.sub("f", call)


def _execute(code: str, call: str) -> str | None:
    """Run code + `repr(<call>)`, return the repr string, or None on error."""
    script = code + f"\nprint(repr({call}))\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(script)
        path = fh.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=_TIMEOUT_S, text=True)
        if r.returncode != 0:
            return None
        return r.stdout.strip()
    except Exception:
        return None
    finally:
        Path(path).unlink(missing_ok=True)


def main():
    mbpp = datasets.load_dataset("google-research-datasets/mbpp", "full")
    problems = list(mbpp["train"]) + list(mbpp["test"]) + list(mbpp["validation"])

    out = []
    n_problems_used = 0
    for ex in problems:
        code = ex["code"]
        ok, fn_name = _is_clean_single_function(code)
        if not ok:
            continue
        calls = _extract_calls(ex["test_list"], fn_name)
        if not calls:
            continue
        renamed_code, _ = _rename_fn(code, calls[0], fn_name)
        used_this_problem = 0
        for call in calls:
            _, renamed_call = _rename_fn(code, call, fn_name)
            result = _execute(renamed_code, renamed_call)
            if result is None:
                continue
            # args-only portion for the prompt, e.g. "(nums, n)" from "f(nums, n)"
            args = renamed_call[len("f") :]
            prompt = _PROMPT_TEMPLATE.format(code=renamed_code, args=args)
            out.append({"prompt": prompt, "completion": result, "task_id": ex["task_id"]})
            used_this_problem += 1
        n_problems_used += used_this_problem > 0

    with open("codefix_eval/cruxeval_sft.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{n_problems_used} MBPP problems used -> {len(out)} training examples")
    print("wrote codefix_eval/cruxeval_sft.jsonl")


if __name__ == "__main__":
    main()
