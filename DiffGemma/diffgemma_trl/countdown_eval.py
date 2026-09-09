"""Evaluate DiffusionGemma (base model or checkpoint) on Countdown.

The prompt style is selectable so evaluation can exactly match the prompt used
for training. Generation uses the shared local-HF/vLLM backend, and scoring
reuses the training reward helpers.
"""

from __future__ import annotations

import json
import os

from .rewards import evaluate_equation, extract_solution, validate_equation
from .sft_data_utils import PROMPT_TEMPLATES
from .sft_eval_common import build_argparser, build_backend, run_eval


_DEFAULT_TEST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "dataset",
    "countdown_cd3_test.jsonl",
)


def _normalize_row(row: dict) -> tuple[list[int], int]:
    """Return numbers and target for either supported Countdown JSONL schema."""
    if "nums" in row:
        return [int(value) for value in row["nums"]], int(row["target"])
    numbers = [
        int(value)
        for value in str(row["input"]).replace(" ", "").split(",")
        if value
    ]
    return numbers, int(row["output"])


def _score_countdown(completion: str, row: dict) -> float:
    equation = extract_solution(completion)
    if equation is None or not validate_equation(equation, row["numbers"]):
        return 0.0
    result = evaluate_equation(equation)
    return float(result is not None and abs(result - row["target"]) < 1e-5)


def main() -> None:
    parser = build_argparser("DiffusionGemma Countdown evaluation")
    parser.add_argument("--test_file", default=_DEFAULT_TEST)
    parser.add_argument(
        "--prompt_style",
        default="countdown",
        choices=sorted(
            name for name in PROMPT_TEMPLATES if name.startswith("countdown")
        ),
        help="Countdown prompt template; use the same style as training.",
    )
    args = parser.parse_args()

    rows = []
    with open(args.test_file, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            numbers, target = _normalize_row(json.loads(line))
            rows.append(
                {
                    "_fmt": {
                        "text": f"Numbers: {numbers}\nTarget: {target}",
                        "numbers": numbers,
                        "target": target,
                    },
                    "numbers": numbers,
                    "target": target,
                }
            )

    backend = build_backend(args)
    run_eval(
        rows,
        PROMPT_TEMPLATES[args.prompt_style],
        _score_countdown,
        backend,
        args,
        label=f"Countdown[{args.prompt_style}]",
    )


if __name__ == "__main__":
    main()
