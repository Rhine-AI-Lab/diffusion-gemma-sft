"""Evaluate a DiffusionGemma checkpoint (or the base model) on HumanEval (pass@1).

Generates a function completion, extracts the fenced code block, combines it with
the problem's signature/docstring if needed, and runs the official check against
the task's ``test`` using the repo's local code sandbox
(``gdsd/utils/local_sandbox.py``).

NOTE: the sandbox is best-effort (reliability_guard), NOT a true security
sandbox -- run inside the project venv only.

Examples:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=~/gdsdv2 python -m DiffGemma.diffgemma_trl.sft_eval_humaneval \
        --backend local --limit 50
    ... python -m DiffGemma.diffgemma_trl.sft_eval_humaneval --backend vllm
"""

from __future__ import annotations

from datasets import load_dataset

from gdsd.utils.local_sandbox import get_successful_tests_slow

from .sft_data_utils import HUMANEVAL_PROMPT
from .sft_eval_common import build_argparser, build_backend, extract_code, run_eval


def _score_humaneval(completion: str, row: dict) -> float:
    program = extract_code(completion)
    # The model is asked for the full function; if it returned only a body (no
    # signature), prepend the prompt so `entry_point` is defined.
    if f"def {row['entry_point']}" not in program:
        program = row["prompt"] + "\n" + program
    test = row["test"] + f"\n\ncheck({row['entry_point']})\n"
    results = get_successful_tests_slow(program, [test])
    return 1.0 if results and all(results) else 0.0


def main():
    ap = build_argparser("DiffusionGemma HumanEval eval (pass@1)")
    ap.add_argument("--dataset_name", default="openai/openai_humaneval")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    ds = load_dataset(args.dataset_name, split=args.split)
    rows = [
        {
            "_prompt": r["prompt"],
            "prompt": r["prompt"],
            "test": r["test"],
            "entry_point": r["entry_point"],
        }
        for r in ds
    ]

    backend = build_backend(args)
    run_eval(rows, HUMANEVAL_PROMPT, _score_humaneval, backend, args, label="HumanEval")


if __name__ == "__main__":
    main()
