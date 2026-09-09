"""Evaluate a DiffusionGemma checkpoint (or the base model) on MATH-500.

Generates a step-by-step solution with a \\boxed{} answer and scores it against
the gold answer with ``math_verify`` (boxed-string fallback if unavailable).

Examples:
    # base model only (no adapter), local backend:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=~/gdsdv2 python -m DiffGemma.diffgemma_trl.sft_eval_math \
        --backend local --limit 50
    # LoRA checkpoint via the local diffusion generate():
    ... python -m DiffGemma.diffgemma_trl.sft_eval_math --backend local \
        --adapter_path ./xp_sft_math/checkpoint-4000
    # via a running vLLM server:
    ... python -m DiffGemma.diffgemma_trl.sft_eval_math --backend vllm \
        --vllm_model google/diffusiongemma-26B-A4B-it
"""

from __future__ import annotations

import re

from datasets import load_dataset

from .sft_data_utils import MATH_PROMPT
from .sft_eval_common import build_argparser, build_backend, run_eval


def _last_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i, depth, start = idx, 0, None
    while i < len(text):
        if text[i] == "{":
            if start is None:
                start = i + 1
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i] if start is not None else None
        i += 1
    return None


def _score_math(completion: str, row: dict) -> float:
    gold = str(row["answer"])
    try:
        from math_verify import parse, verify

        # MATH-500/Minerva/Olympiad golds are BARE answers (no delimiters). Wrapping
        # in \boxed{} makes math_verify parse them as LaTeX answers -- otherwise a
        # coordinate tuple "\left(3,\frac{\pi}{2}\right)" parses to just "3" and a bare
        # symbolic answer "p - q" parses to nothing, both -> false negatives that
        # under-count accuracy. verify(gold, pred) is the documented arg order.
        gold_parsed = parse(gold if "\\boxed" in gold else "\\boxed{" + gold + "}")
        return 1.0 if verify(gold_parsed, parse(completion)) else 0.0
    except ImportError:
        pass
    # Fallback: compare normalized boxed answer.
    pred = _last_boxed(completion)
    if pred is None:
        m = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        pred = m.group(1) if m else completion
    norm = lambda s: re.sub(r"\s|\\left|\\right|\\,|\$", "", s or "").rstrip(".")
    return 1.0 if norm(pred) == norm(gold) else 0.0


def main():
    ap = build_argparser("DiffusionGemma MATH-500 eval")
    ap.add_argument("--dataset_name", default="HuggingFaceH4/MATH-500")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    ds = load_dataset(args.dataset_name, split=args.split)
    rows = [{"_prompt": r["problem"], "answer": r["answer"]} for r in ds]

    backend = build_backend(args)
    run_eval(rows, MATH_PROMPT, _score_math, backend, args, label="MATH-500")


if __name__ == "__main__":
    main()
