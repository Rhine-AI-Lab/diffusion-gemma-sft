"""Build GSM8K train/test JSONL for DiffusionGemma SFT.

Train rows:  {"prompt": <question>, "completion": <CoT answer incl. '#### <n>'>}
Test rows:   {"prompt": <question>, "answer": <gold number>}

The completion's final answer is at the END ('#### <n>'), and the single-canvas
SFT collator truncates completions from the right -- so we FILTER train rows
whose tokenized completion exceeds the canvas (default 256), keeping only those
whose answer survives. (Longer chains need multi-canvas SFT, not yet supported.)

Run (gdsd .venv, has datasets+transformers):
  python dataset/make_gsm8k_jsonl.py --canvas_length 256 --out_dir dataset
"""

import argparse
import json
import os
import re

import datasets
from transformers import AutoTokenizer


def gold(answer: str) -> str:
    # GSM8K gold is the text after '####'.
    return answer.split("####", 1)[1].strip().replace(",", "") if "####" in answer else ""


def to_xml(answer: str) -> str:
    """Reformat GSM8K '{cot} #### {N}' into the repo's <reasoning>/<answer> format
    (matches gdsd/data_utils.GSM8K_XML_SYSTEM_PROMPT, so SFT output == RL prompt)."""
    cot, _, tail = answer.partition("####")
    return f"<reasoning>\n{cot.strip()}\n</reasoning>\n<answer>\n{tail.strip()}\n</answer>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_id", default="openai/gsm8k")
    ap.add_argument("--config", default="main")
    ap.add_argument("--tokenizer", default="unsloth/diffusiongemma-26B-A4B-it")
    ap.add_argument("--canvas_length", type=int, default=256)
    ap.add_argument("--out_dir", default="dataset")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = datasets.load_dataset(args.hf_id, args.config)

    # train: filter to completions that fit the canvas (answer at the end).
    n_keep = n_drop = 0
    with open(os.path.join(args.out_dir, "gsm8k_train.jsonl"), "w") as f:
        for r in ds["train"]:
            ans = to_xml(r["answer"])  # <reasoning>...</reasoning><answer>N</answer>
            if len(tok.encode(ans, add_special_tokens=False)) + 1 > args.canvas_length:
                n_drop += 1
                continue
            f.write(json.dumps({"prompt": r["question"], "completion": ans}) + "\n")
            n_keep += 1
    print(f"train: kept {n_keep}, dropped {n_drop} (> {args.canvas_length} tok)")

    with open(os.path.join(args.out_dir, "gsm8k_test.jsonl"), "w") as f:
        for r in ds["test"]:
            f.write(json.dumps({"prompt": r["question"], "answer": gold(r["answer"])}) + "\n")
    print(f"test: {len(ds['test'])} -> gsm8k_test.jsonl")


if __name__ == "__main__":
    main()
