"""Evaluate a DiffusionGemma GSM8K LoRA checkpoint on the GSM8K test set.

Mirrors sft_eval_bands.py (same DiffusionGemma HF generate path + entropy-bound
sampler), but for GSM8K: rebuilds the SFT GSM8K_PROMPT, generates, extracts the
<answer> tag, and scores exact-match against the ground-truth number.
"""
from __future__ import annotations

import argparse
import json
import re

import torch
from peft import PeftModel
from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion

from DiffGemma.diffgemma_trl.sft_eval_bands import _gen_config, _unwrap_clippable_linears, _set_seed
from DiffGemma.diffgemma_trl.sft_data_utils import GSM8K_PROMPT


def _norm(s: str) -> str:
    """Normalize a numeric answer for exact match (drop commas/$/spaces)."""
    if s is None:
        return ""
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return m.group(0) if m else s


def _extract_xml_answer(text: str) -> str:
    return text.split("<answer>")[-1].split("</answer>")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="google/diffusiongemma-26B-A4B-it")
    ap.add_argument("--adapter_path", required=True)
    ap.add_argument("--base_adapter_path", default=None)
    ap.add_argument("--base_adapter_merge", action="store_true")
    ap.add_argument("--test_path", default="./dataset/gsm8k_test.jsonl")
    ap.add_argument("--limit", type=int, default=200, help="eval first N (0=all 1319)")
    ap.add_argument("--max_denoising_steps", type=int, default=64)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress_every", type=int, default=20, help="print running accuracy every N problems")
    args = ap.parse_args()

    _set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    _unwrap_clippable_linears(model)
    if args.base_adapter_path:
        model = PeftModel.from_pretrained(model, args.base_adapter_path)
        if args.base_adapter_merge:
            model = model.merge_and_unload()
            _unwrap_clippable_linears(model)
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    gc = _gen_config(args.max_denoising_steps, args.max_new_tokens)

    rows = [json.loads(l) for l in open(args.test_path)]
    if args.limit:
        rows = rows[: args.limit]
    print(f"adapter={args.adapter_path} | n={len(rows)} | steps={args.max_denoising_steps} | seed={args.seed}", flush=True)

    ncorrect = 0
    for i, r in enumerate(rows):
        full = GSM8K_PROMPT.format(text=r["prompt"].strip())
        enc = tok(full, add_special_tokens=True, return_tensors="pt")
        inp = {k: v.to(model.device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model.generate(**inp, generation_config=gc)
        seq = out.sequences if hasattr(out, "sequences") else out
        plen = inp["input_ids"].shape[-1]
        text = tok.decode(seq[0, plen:], skip_special_tokens=True)
        pred = _norm(_extract_xml_answer(text))
        gt = _norm(r["answer"])
        ncorrect += int(pred == gt and pred != "")
        if i < 4:
            print(f"  [{i}] gt={gt!r} pred={pred!r} {'OK' if pred==gt else 'x'}", flush=True)
        done = i + 1
        if args.progress_every and done % args.progress_every == 0:
            print(f"  progress {done}/{len(rows)}  running_acc={100*ncorrect/done:.1f}%  ({ncorrect}/{done})", flush=True)
    n = len(rows)
    print(f"\nGSM8K accuracy: {100*ncorrect/n:.1f}%  ({ncorrect}/{n})", flush=True)


if __name__ == "__main__":
    main()
