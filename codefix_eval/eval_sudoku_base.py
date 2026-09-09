"""Eval a served DiffusionGemma checkpoint on the stratified sudoku eval sets
(dataset/sudoku_eval_{easy,medium,hard}.jsonl), using the exact native
DIFFUGEMMA_SUDOKU_PROMPT template (raw text completion, not chat)."""

from __future__ import annotations

import argparse
import json
import re
import urllib.request

DIFFUGEMMA_SUDOKU_PROMPT = (
    "<|turn>system Solve the following Sudoku puzzle. Empty cells are"
    " represented by 0. Output ONLY the solved puzzle immediately as"
    " a 9x9 grid of numbers separated by spaces. Do not include ####,"
    " explanations, or any other text.<turn|>\n<|turn>user"
    " {text}<turn|>\n<|turn>model\n"
)


def _extract_grid(text: str, prefer_first: bool = False) -> list[int] | None:
    # The base model sometimes rambles / re-attempts ("Wait, let me re-solve
    # carefully...") before its final answer -- take the LAST 81 digits, not
    # the first, so a genuine final answer isn't scored against an abandoned
    # earlier attempt. Fine-tuned checkpoints show the opposite pattern:
    # they answer immediately, then keep generating extra content after
    # (DiffusionGemma's block-diffusion decoding fills the requested canvas
    # length rather than stopping at EOS the way autoregressive models do)
    # -- pass prefer_first=True for those so the real answer isn't skipped.
    nums = re.findall(r"[0-9]", text)
    if len(nums) < 81:
        return None
    digits = nums[:81] if prefer_first else nums[-81:]
    return [int(n) for n in digits]


def eval_file(path: str, api_base: str, model: str, max_tokens: int, prefer_first: bool = False) -> tuple[int, int]:
    n_correct = 0
    n_total = 0
    for line in open(path):
        ex = json.loads(line)
        prompt = DIFFUGEMMA_SUDOKU_PROMPT.format(text=ex["puzzle"])
        body = json.dumps(
            {"model": model, "prompt": prompt, "temperature": 0.0, "max_tokens": max_tokens}
        ).encode()
        try:
            out = json.load(
                urllib.request.urlopen(
                    urllib.request.Request(
                        f"{api_base}/completions", body, {"Content-Type": "application/json"}
                    ),
                    timeout=120,
                )
            )
            raw = out["choices"][0]["text"] or ""
        except Exception as e:
            raw = ""
            print(f"  REQUEST ERROR: {e}")
        pred = _extract_grid(raw, prefer_first=prefer_first)
        target = [int(x) for x in ex["solution"].split()]
        correct = pred == target
        n_correct += correct
        n_total += 1
    return n_correct, n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_base", default="http://localhost:8015/v1")
    ap.add_argument("--model", default="diffgemma")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--prefer_first", action="store_true",
                     help="Score the first 81 digits instead of the last 81 -- use for "
                          "fine-tuned checkpoints, which answer immediately then keep "
                          "generating filler (unlike the rambling-then-answering base model).")
    args = ap.parse_args()

    for tier in ["easy", "medium", "hard"]:
        path = f"dataset/sudoku_eval_{tier}.jsonl"
        c, n = eval_file(path, args.api_base, args.model, args.max_tokens, prefer_first=args.prefer_first)
        print(f"{tier:8s}: {c}/{n} = {100*c/n:.1f}%")


if __name__ == "__main__":
    main()
