"""Evaluate a DiffusionGemma checkpoint (or the base model) on MBPP (pass@1).

Generates a Python solution, extracts the fenced code block, and runs it against
the task's assert-style ``test_list`` using the repo's local code sandbox
(``gdsd/utils/local_sandbox.py``). pass@1 = fraction of tasks whose generated
code passes ALL of its tests.

NOTE: the sandbox is best-effort (reliability_guard), NOT a true security
sandbox -- run inside the project venv only.

Examples:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=~/gdsdv2 python -m DiffGemma.diffgemma_trl.sft_eval_mbpp \
        --backend local --limit 50
    ... python -m DiffGemma.diffgemma_trl.sft_eval_mbpp --backend vllm
"""

from __future__ import annotations

from datasets import load_dataset

from gdsd.utils.local_sandbox import get_successful_tests_slow

from .sft_data_utils import MBPP_PROMPT
from .sft_eval_common import build_argparser, build_backend, extract_code, run_eval


def _score_mbpp(completion: str, row: dict) -> float:
    program = extract_code(completion)
    if row.get("setup"):
        program = row["setup"] + "\n" + program
    tests = row["test_list"]
    if not tests:
        return 0.0
    results = get_successful_tests_slow(program, tests)
    return 1.0 if results and all(results) else 0.0


def main():
    ap = build_argparser("DiffusionGemma MBPP eval (pass@1)")
    ap.add_argument("--dataset_name", default="google-research-datasets/mbpp")
    ap.add_argument("--dataset_config", default="full",
                    help="MBPP config: 'full' or 'sanitized'.")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    ds = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    rows = []
    for r in ds:
        tests = list(r["test_list"])
        # Show the asserts so the model knows the required function name/signature.
        task = r["text"] + "\nYour code must pass these tests:\n" + "\n".join(tests)
        rows.append({
            "_prompt": task,
            "test_list": tests,
            "setup": r.get("test_setup_code", "") or "",
        })

    backend = build_backend(args)
    run_eval(rows, MBPP_PROMPT, _score_mbpp, backend, args, label="MBPP")


if __name__ == "__main__":
    main()
