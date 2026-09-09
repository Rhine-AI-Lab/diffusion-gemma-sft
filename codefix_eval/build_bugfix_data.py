"""Build a single-shot bug-fix SFT dataset from MBPP (correct, verified solutions),
by injecting small, plausible bugs -- AST-level mutations analogous to
HumanEvalFix's bug categories (operator misuse, value misuse, variable misuse,
missing logic) -- and keeping only mutations verified to actually break at
least one of the problem's own tests (i.e. real, observable bugs, not
semantically-equivalent rewrites).

Source (train) and eval (HumanEvalFix, bigcode/humanevalpack) are disjoint
benchmarks -- no contamination.

Output schema matches the plain flat prompt/completion SFT path
(DiffGemma/diffgemma_trl/sft_data_utils.py), not the multi-turn agent one --
this is a single-shot task.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import datasets

_TIMEOUT_S = 5

_PROMPT_TEMPLATE = """Fix the bug in the following Python function.

Task: {text}

Buggy code:
```python
{buggy_code}
```

Write the complete corrected function."""


# ---- AST mutations -------------------------------------------------------

_CMP_FLIPS = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt,
    ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
}
_BOOL_FLIPS = {ast.And: ast.Or, ast.Or: ast.And}
_ARITH_FLIPS = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv}


class _MutationSite:
    """Records candidate (node, apply_fn) pairs found while walking the tree."""

    def __init__(self):
        self.sites: list[tuple[str, callable]] = []


def _collect_sites(tree: ast.AST) -> _MutationSite:
    ms = _MutationSite()

    for node in ast.walk(tree):
        # operator misuse: comparisons
        if isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                if type(op) in _CMP_FLIPS:
                    def _mk(n=node, i=i):
                        def apply():
                            n.ops[i] = _CMP_FLIPS[type(n.ops[i])]()
                        return apply
                    ms.sites.append(("operator misuse", _mk()))
        # operator misuse: boolean
        if isinstance(node, ast.BoolOp) and type(node.op) in _BOOL_FLIPS:
            def _mk(n=node):
                def apply():
                    n.op = _BOOL_FLIPS[type(n.op)]()
                return apply
            ms.sites.append(("operator misuse", _mk()))
        # operator misuse: arithmetic
        if isinstance(node, ast.BinOp) and type(node.op) in _ARITH_FLIPS:
            def _mk(n=node):
                def apply():
                    n.op = _ARITH_FLIPS[type(n.op)]()
                return apply
            ms.sites.append(("operator misuse", _mk()))
        # value misuse: integer literal off-by-one
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            def _mk(n=node):
                def apply():
                    n.value = n.value + random.choice([-1, 1])
                return apply
            ms.sites.append(("value misuse", _mk()))
        # missing logic: drop a return's negation / an "if" guard's body effect
        # (safe subset: flip a Return's boolean constant)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool):
            def _mk(n=node):
                def apply():
                    n.value.value = not n.value.value
                return apply
            ms.sites.append(("value misuse", _mk()))

    return ms


def _swap_variable_names(tree: ast.FunctionDef) -> list[tuple[str, callable]]:
    """variable misuse: swap two same-kind local Name loads within the function."""
    names = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)]
    by_id: dict[str, list[ast.Name]] = {}
    for n in names:
        by_id.setdefault(n.id, []).append(n)
    ids = [i for i in by_id if not i.startswith("_")]
    sites = []
    if len(ids) >= 2:
        a, b = random.sample(ids, 2)

        def apply(a=a, b=b, by_id=by_id):
            for n in by_id[a]:
                n.id = b
            for n in by_id[b]:
                n.id = a
        sites.append(("variable misuse", apply))
    return sites


def corrupt_once(source_code: str) -> tuple[str, str] | None:
    """Return (bug_type, mutated_source) or None if no mutation applied cleanly."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return None
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if not funcs:
        return None

    ms = _collect_sites(tree)
    for fn in funcs:
        ms.sites.extend(_swap_variable_names(fn))
    if not ms.sites:
        return None

    random.shuffle(ms.sites)
    for bug_type, apply in ms.sites:
        # ms.sites closures were bound to nodes in `tree`; apply the mutation
        # directly to that tree, then unparse it.
        try:
            apply()
            mutated = ast.unparse(tree)
        except Exception:
            continue
        if mutated.strip() != source_code.strip():
            return bug_type, mutated
    return None


def _run_tests(code: str, test_list: list[str], test_setup: str = "") -> bool:
    """Return True if code passes ALL asserts in test_list."""
    script = (test_setup or "") + "\n" + code + "\n" + "\n".join(test_list)
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


def build(n_attempts_per_problem: int, seed: int) -> list[dict]:
    random.seed(seed)
    mbpp = datasets.load_dataset("google-research-datasets/mbpp", "full")
    problems = list(mbpp["train"]) + list(mbpp["test"]) + list(mbpp["validation"])

    out = []
    for ex in problems:
        code = ex["code"]
        tests = ex["test_list"]
        setup = ex.get("test_setup_code", "") or ""
        if not _run_tests(code, tests, setup):
            continue  # skip problems whose own reference solution doesn't pass (rare data noise)

        found = None
        for _ in range(n_attempts_per_problem):
            res = corrupt_once(code)
            if res is None:
                continue
            bug_type, mutated = res
            if _run_tests(mutated, tests, setup):
                continue  # mutation didn't actually break anything observable -- discard
            found = (bug_type, mutated)
            break

        if found is None:
            continue
        bug_type, buggy_code = found
        out.append(
            {
                "task_id": ex["task_id"],
                "text": ex["text"],
                "canonical_solution": code,
                "buggy_solution": buggy_code,
                "bug_type": bug_type,
                "test_list": tests,
                "test_setup_code": setup,
            }
        )
    return out


def to_sft_pairs(records: list[dict]) -> list[dict]:
    pairs = []
    for r in records:
        prompt = _PROMPT_TEMPLATE.format(text=r["text"], buggy_code=r["buggy_solution"])
        completion = f"```python\n{r['canonical_solution']}\n```"
        pairs.append({"prompt": prompt, "completion": completion, "task_id": r["task_id"], "bug_type": r["bug_type"]})
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_raw", default="codefix_eval/bugfix_raw.jsonl")
    ap.add_argument("--out_sft", default="codefix_eval/bugfix_sft.jsonl")
    ap.add_argument("--attempts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    records = build(args.attempts, args.seed)
    with open(args.out_raw, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pairs = to_sft_pairs(records)
    with open(args.out_sft, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"MBPP problems -> {len(records)} verified bug-fix pairs (of {len(records)} attempted)")
    print("bug_type distribution:", Counter(r["bug_type"] for r in records))
    print(f"wrote {args.out_raw} and {args.out_sft}")


if __name__ == "__main__":
    main()
