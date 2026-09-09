"""Evaluate a DiffusionGemma SFT LoRA checkpoint on 9x9 Sudoku difficulty bands.

Loads base model + LoRA adapter, generates solutions with the diffusion
``generate()`` loop, and scores exact-solve + cell accuracy per band
(easy/medium/hard), on the *same* official-band puzzles as the JAX eval
(easy>=40, medium 30-39, hard<30 clues; 100 each) so the numbers are comparable.

Example:
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=~/gdsdv2 python -m DiffGemma.diffgemma_trl.sft_eval_bands \
        --adapter_path ./xp_sft_loo_ce_trl/checkpoint-4000 \
        --bands_dir ./dataset --max_denoising_steps 64
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion

# Shared helpers live in sft_eval_common (kept importable here for back-compat with
# sft_eval_gsm8k, which imports them from this module).
from .sft_eval_common import _gen_config, _set_seed, _unwrap_clippable_linears

REASONING_PREFILL = "<reasoning>"
THINK_TOKEN = "<|think|>"

SUDOKU_9X9_SYSTEM_PROMPT = """Solve the following 9x9 Sudoku puzzle. Empty cells are represented by 0.

Rules:
- Fill empty cells with digits 1-9
- Each row must contain digits 1-9 exactly once
- Each column must contain digits 1-9 exactly once
- Each 3x3 box must contain digits 1-9 exactly once

Important: Your solution must be a COMPLETE 81-digit string or 9x9 grid with only the digits 1-9, representing your final solved grid.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[complete solved Sudoku grid]
</answer>"""


def _digits(s: str) -> str:
    return "".join(c for c in s if c.isdigit())


def _extract(text: str) -> str:
    start = text.rfind("<answer>")
    end = text.rfind("</answer>")
    if 0 <= start < end:
        return _digits(text[start + len("<answer>") : end])
    return _digits(text)


def _build_prompt(tokenizer, puzzle: str, prompt_template: str) -> tuple[str, bool]:
    if prompt_template == "legacy_sft":
        from DiffGemma.diffgemma_trl.sft_data_utils import DIFFUGEMMA_SUDOKU_PROMPT

        return DIFFUGEMMA_SUDOKU_PROMPT.format(text=puzzle), True

    question = f"Solve the following Sudoku puzzle:\n{puzzle}"
    user_content = SUDOKU_9X9_SYSTEM_PROMPT + "\n\n" + question
    if prompt_template == "official_think":
        return (
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": THINK_TOKEN},
                    {"role": "user", "content": user_content},
                ],
                add_generation_prompt=True,
                tokenize=False,
            ),
            False,
        )
    if prompt_template == "gdsd_prefill":
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            add_generation_prompt=True,
            tokenize=False,
        )
        return text + REASONING_PREFILL, False
    raise ValueError(f"Unknown prompt_template={prompt_template!r}")


def _score(pred: str, sol: str, puzzle: str) -> tuple[float, float]:
    size = len(sol)
    # Anchor on the FIRST `size` digits: the prompt asks for the grid
    # "immediately", so the answer is the leading 81 digits. This is robust to
    # trailing degeneration (the observed failure mode); taking the last `size`
    # would instead capture trailing junk. Short/malformed preds are zero-padded
    # and therefore (correctly) fail the exact-solve check.
    p = (pred + "0" * size)[:size]
    full = 1.0 if p == sol else 0.0
    empty = [i for i, v in enumerate(puzzle) if v in "0."]
    cell = (sum(1 for i in empty if p[i] == sol[i]) / len(empty)) if empty else 0.0
    return full, cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="google/diffusiongemma-26B-A4B-it")
    ap.add_argument("--adapter_path", default=None, help="Optional LoRA adapter/checkpoint to evaluate.")
    ap.add_argument(
        "--base_adapter_path",
        default=None,
        help="Optional LoRA adapter to apply before --adapter_path, e.g. the SFT adapter merged before RL.",
    )
    ap.add_argument(
        "--base_adapter_merge",
        action="store_true",
        help="Merge --base_adapter_path into the base before loading --adapter_path.",
    )
    ap.add_argument("--bands_dir", default="./dataset")
    ap.add_argument("--bands", default="easy,medium,hard")
    ap.add_argument("--max_denoising_steps", type=int, default=64)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument(
        "--prompt_template",
        choices=["legacy_sft", "gdsd_prefill", "official_think"],
        default="legacy_sft",
        help="'gdsd_prefill'=apply_chat_template + literal '<reasoning>'; "
        "'official_think'=system '<|think|>'; 'legacy_sft'=original bare-grid SFT prompt.",
    )
    ap.add_argument("--limit", type=int, default=0, help="eval only first N per band (0=all)")
    ap.add_argument("--save_generations", default=None, help="Optional JSONL path for per-example generations.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for stochastic generation.")
    args = ap.parse_args()

    _set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    _unwrap_clippable_linears(model)  # match training so PEFT can attach the adapter
    if args.base_adapter_path:
        model = PeftModel.from_pretrained(model, args.base_adapter_path)
        if args.base_adapter_merge:
            model = model.merge_and_unload()
            _unwrap_clippable_linears(model)
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    gc = _gen_config(args.max_denoising_steps, args.max_new_tokens)

    base_adapter = args.base_adapter_path or "none"
    print(
        f"base_adapter={base_adapter} | base_merge={args.base_adapter_merge} | "
        f"adapter={args.adapter_path or 'none'} | steps={args.max_denoising_steps} | "
        f"max_new_tokens={args.max_new_tokens} | prompt_template={args.prompt_template} | seed={args.seed}"
    )
    gen_f = None
    if args.save_generations:
        os.makedirs(os.path.dirname(args.save_generations) or ".", exist_ok=True)
        gen_f = open(args.save_generations, "w")

    try:
        for band in args.bands.split(","):
            path = os.path.join(args.bands_dir, f"sudoku_eval_{band}.jsonl")
            rows = [json.loads(l) for l in open(path)]
            if args.limit:
                rows = rows[: args.limit]
            nfull = ncell = 0.0
            for idx, r in enumerate(rows):
                sol, puz = _digits(r["solution"]), _digits(r["puzzle"])
                prompt_text, add_special_tokens = _build_prompt(tok, r["puzzle"], args.prompt_template)
                enc = tok(prompt_text, add_special_tokens=add_special_tokens, return_tensors="pt")
                inp = {k: v.to(model.device) for k, v in enc.items()}
                with torch.inference_mode():
                    out = model.generate(**inp, generation_config=gc)
                seq = out.sequences if hasattr(out, "sequences") else out
                plen = inp["input_ids"].shape[-1]
                text = tok.decode(seq[0, plen:], skip_special_tokens=True)
                extracted = _extract(text)
                full, cell = _score(extracted, sol, puz)
                nfull += full
                ncell += cell
                if gen_f is not None:
                    gen_f.write(json.dumps({
                        "band": band,
                        "idx": idx,
                        "prompt_template": args.prompt_template,
                        "max_new_tokens": args.max_new_tokens,
                        "max_denoising_steps": args.max_denoising_steps,
                        "seed": args.seed,
                        "puzzle": r["puzzle"],
                        "solution": r["solution"],
                        "prompt": prompt_text,
                        "generation": text,
                        "extracted_digits": extracted,
                        "scored_digits": (extracted + "0" * len(sol))[: len(sol)],
                        "full_solve": bool(full),
                        "cell_acc": cell,
                    }, ensure_ascii=False) + "\n")
                    gen_f.flush()
            n = len(rows)
            print(f"{band}: n={n}  full_solve={nfull / n * 100:.1f}%  cell_acc={ncell / n * 100:.1f}%", flush=True)
    finally:
        if gen_f is not None:
            gen_f.close()


if __name__ == "__main__":
    main()
