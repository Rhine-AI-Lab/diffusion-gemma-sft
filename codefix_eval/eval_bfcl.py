"""BFCL (Berkeley Function-Calling Leaderboard) v3 eval harness -- held-out
only, never used for training. Uses BFCL_v3_simple.json (400 single-turn,
single-tool examples): given one tool schema + a user request, does the
model emit the correct {"tool": ..., "arguments": {...}} call?

Loaded via HfApi/hf_hub_download (list_repo_files), bypassing the
datasets.load_dataset auto-loader which fails on this repo's per-file
schema layout, same workaround used for API-Bank.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request

from huggingface_hub import hf_hub_download

sys.path.insert(0, "codefix_eval")
from build_apibank_data import _PROMPT_TEMPLATE  # noqa: E402


def _format_tool(fn: dict) -> str:
    props = fn.get("parameters", {}).get("properties", {})
    required = set(fn.get("parameters", {}).get("required", []))
    param_desc = ", ".join(
        f"{k}: {v.get('type', 'string')}" + (" (required)" if k in required else "")
        for k, v in props.items()
    )
    return f"- {fn['name']}({param_desc}): {fn.get('description', '')}"


def _extract_call(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    if "tool" not in obj:
        return None
    args = obj.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            pass
    return obj.get("tool"), args if isinstance(args, dict) else {}


def _score(pred_name, pred_args, ground_truth: list[dict]) -> bool:
    gt = ground_truth[0]
    gt_name = next(iter(gt.keys()))
    if pred_name != gt_name:
        return False
    gt_args = gt[gt_name]
    for k, acceptable in gt_args.items():
        pv = pred_args.get(k, None)
        matched = any(
            str(pv).strip().lower() == str(av).strip().lower()
            or (pv is None and av == "")
            for av in acceptable
        )
        if not matched:
            return False
    return True


def load_examples(n=None, seed=0):
    qpath = hf_hub_download(
        "gorilla-llm/Berkeley-Function-Calling-Leaderboard", "BFCL_v3_simple.json", repo_type="dataset"
    )
    apath = hf_hub_download(
        "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        "possible_answer/BFCL_v3_simple.json",
        repo_type="dataset",
    )
    questions = {}
    for line in open(qpath):
        ex = json.loads(line)
        questions[ex["id"]] = ex
    answers = {}
    for line in open(apath):
        ex = json.loads(line)
        answers[ex["id"]] = ex["ground_truth"]

    ids = sorted(questions.keys(), key=lambda i: int(i.split("_")[1]))
    if n is not None:
        import random

        random.seed(seed)
        ids = random.sample(ids, n)

    out = []
    for i in ids:
        q = questions[i]
        user_turn = q["question"][0][0]["content"]
        fn = q["function"][0]
        prompt = _PROMPT_TEMPLATE.format(tools=_format_tool(fn), user=user_turn)
        out.append({"id": i, "prompt": prompt, "ground_truth": answers[i]})
    return out


def main(api_base="http://localhost:8015/v1", model="diffgemma", n=100, max_tokens=200, seed=0):
    examples = load_examples(n=n, seed=seed)
    n_correct = 0
    n_name_only = 0
    for j, ex in enumerate(examples):
        body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": ex["prompt"]}],
                "temperature": 0.0,
                "max_tokens": max_tokens,
            }
        ).encode()
        try:
            out = json.load(
                urllib.request.urlopen(
                    urllib.request.Request(
                        f"{api_base}/chat/completions", body, {"Content-Type": "application/json"}
                    ),
                    timeout=120,
                )
            )
            msg = out["choices"][0]["message"]
            raw = msg.get("content") or msg.get("reasoning") or ""
        except Exception as e:
            raw = ""
            print(f"[{j}] REQUEST ERROR {e}")

        parsed = _extract_call(raw)
        gt_name = next(iter(ex["ground_truth"][0].keys()))
        if parsed is None:
            print(f"[{j+1}/{len(examples)}] {ex['id']} target={gt_name:28s} PARSE FAIL raw={raw[:80]!r}")
            continue
        pred_name, pred_args = parsed
        name_ok = pred_name == gt_name
        full_ok = _score(pred_name, pred_args, ex["ground_truth"])
        n_correct += full_ok
        n_name_only += name_ok
        print(f"[{j+1}/{len(examples)}] {ex['id']} target={gt_name:28s} name_ok={name_ok} full_ok={full_ok}")

    print(f"\n=== name correct: {n_name_only}/{len(examples)} ({100*n_name_only/len(examples):.1f}%) ===")
    print(f"=== name+args exact: {n_correct}/{len(examples)} ({100*n_correct/len(examples):.1f}%) ===")


if __name__ == "__main__":
    main()
