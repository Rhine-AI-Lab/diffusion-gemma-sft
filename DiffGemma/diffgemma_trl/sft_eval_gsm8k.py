"""Evaluate a DiffusionGemma SFT LoRA checkpoint on GSM8K (exact-answer match).

Generates a CoT solution with the diffusion generate() loop, extracts the final
number (reusing rewards.extract_answer), and compares to the gold answer.

Example:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=~/gdsdv2 python -m DiffGemma.diffgemma_trl.sft_eval_gsm8k \
        --adapter_path ./xp_sft_gsm8k_base/checkpoint-4000 --test_file ./dataset/gsm8k_test.jsonl
"""

from __future__ import annotations

import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion

from .rewards import extract_answer
from .sft_data_utils import GSM8K_PROMPT
from .sft_eval_common import _gen_config, _set_seed, _unwrap_clippable_linears


def _norm(s: str) -> str:
    return (s or "").strip().replace(",", "").rstrip(".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="unsloth/diffusiongemma-26B-A4B-it")
    ap.add_argument("--adapter_path", default=None,
                    help="LoRA checkpoint dir; omit to evaluate the base model.")
    ap.add_argument("--test_file", default="./dataset/gsm8k_test.jsonl")
    ap.add_argument("--max_denoising_steps", type=int, default=64)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    if args.adapter_path:
        _unwrap_clippable_linears(model)  # only needed to attach the LoRA adapter
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    gc = _gen_config(args.max_denoising_steps, args.max_new_tokens)

    rows = [json.loads(l) for l in open(args.test_file)]
    if args.limit:
        rows = rows[: args.limit]
    n = len(rows)
    correct = 0
    for i, r in enumerate(rows, 1):
        enc = tok(GSM8K_PROMPT.format(text=r["prompt"]), add_special_tokens=True, return_tensors="pt")
        inp = {k: v.to(model.device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model.generate(**inp, generation_config=gc)
        seq = out.sequences if hasattr(out, "sequences") else out
        text = tok.decode(seq[0, inp["input_ids"].shape[-1]:], skip_special_tokens=True)
        correct += int(_norm(extract_answer(text)) == _norm(r["answer"]))
        if i % 25 == 0 or i == n:
            print(f"  [{i}/{n}] running acc = {correct/i*100:.1f}%", flush=True)
    print(f"adapter={args.adapter_path or 'BASE (no LoRA)'} | seed={args.seed} | "
          f"GSM8K acc = {correct/n*100:.1f}%  ({correct}/{n})")


if __name__ == "__main__":
    main()
