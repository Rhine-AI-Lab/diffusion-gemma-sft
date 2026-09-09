"""Build OpenR1-Math-220k SFT training files for DiffusionGemma (canvas 4096).

Two toggleable response objectives (``--response_mode``); both wrap the reasoning +
the GROUND-TRUTH answer (from the ``answer`` column) into the repo's math SFT format
that the MATH eval scorer expects:

    <reasoning>
    {reasoning}
    </reasoning>
    <answer>
    \\boxed{answer}      <- always the dataset's `answer` column (ground truth)
    </answer>

  * ``solution``  -- reasoning = the Numina reference ``solution`` column (concise,
                     natural, reference-correct; ~266 median tok -> fits any canvas).
  * ``r1_cot``    -- reasoning = the R1 <think> chain-of-thought ONLY (the post-</think>
                     restated summary is stripped); answer appended from `answer`.

Using the `answer` column for the box (rather than re-extracting the last \\boxed in the
trace) fixes the partial/wrong-answer problem on multi-answer rows.

Rows are kept only if the tokenized completion + EOS fits ``--canvas_length``.
``--strict_mathverify`` (r1_cot only) further keeps only rows whose trained generation
passed the strict rule-based math_verify check (~70%; the rest are Llama-judge-only).

Run (gdsd .venv):
  python dataset/make_openr1_math_jsonl.py --response_mode solution \
      --out dataset/openr1_math_sft_solution_c4096.jsonl
  python dataset/make_openr1_math_jsonl.py --response_mode r1_cot \
      --out dataset/openr1_math_sft_r1cot_c4096.jsonl
"""

import argparse
import glob
import importlib.util
import json
import os
import re

from transformers import AutoTokenizer

# reuse the exact loaders / formatting used by the length analysis
_ana_path = os.path.join(os.path.dirname(__file__), "analyze_math_sft_lengths.py")
_spec = importlib.util.spec_from_file_location("ana_lengths", _ana_path)
ana = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ana)


def think_inner(text: str) -> str:
    """The R1 chain-of-thought: content between <think>...</think>, dropping the
    post-</think> restated summary. Falls back to the whole trace (tags removed)."""
    if "<think>" in text and "</think>" in text:
        return text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    return ana.strip_think(text)


def r1_trace(row) -> str:
    msgs = row.get("messages") or []
    return msgs[-1]["content"] if msgs else (row.get("generations") or [row.get("solution") or ""])[0]


def clean_answer(ans: str) -> str:
    """The ground-truth answer to place inside \\boxed{}. Un-double-box, then strip the
    mangled empty unit markup the source leaves on some rows (``\\mathrm{~}``, ``\\mathrm{}``
    and the dangling ``/`` they leave behind, e.g. ``500\\mathrm{~}`` -> ``500`` and
    ``v_{R}=4\\mathrm{~}/\\mathrm{},v_{B}=10\\mathrm{~}/\\mathrm{}`` -> ``v_{R}=4,v_{B}=10``).
    Content-bearing markup (``\\mathrm{Ah}``, ``\\mathrm{E}``) is left untouched."""
    ans = (ans or "").strip()
    inner = ana.last_boxed(ans)
    if inner is not None:
        ans = inner.strip()
    ans = ans.replace("\\mathrm{~}", "").replace("\\mathrm{}", "")
    ans = re.sub(r"/(?=,|$)", "", ans)      # drop slashes left dangling by removed units
    return ans.strip().rstrip("/").strip()


def messages_is_mathverify_correct(row) -> bool:
    """True iff the generation used for `messages` passed strict math_verify."""
    trace = r1_trace(row)
    gens = row.get("generations") or []
    cmv = row.get("correctness_math_verify") or []
    for i, g in enumerate(gens):
        if g == trace:
            return bool(i < len(cmv) and cmv[i] is True)
    return False


def build_reasoning(row, mode: str) -> str:
    if mode == "solution":
        return (row.get("solution") or "").strip()
    if mode == "r1_cot":
        return think_inner(r1_trace(row))
    raise ValueError(f"unknown response_mode {mode!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--response_mode", choices=["solution", "r1_cot"], required=True)
    ap.add_argument("--tokenizer", default="unsloth/diffusiongemma-26B-A4B-it")
    ap.add_argument("--canvas_length", type=int, default=4096)
    ap.add_argument("--strict_mathverify", action="store_true",
                    help="r1_cot: keep only rows whose trained generation is math_verify-correct.")
    ap.add_argument("--restrict_prompts", default=None,
                    help="Path to a jsonl with a 'prompt' field; only emit rows whose problem text "
                         "is in it (use to subsample one mode to exactly another mode's rows). "
                         "NB: `problem` is the unique row key -- OpenR1 uuids are NOT unique.")
    ap.add_argument("--max_rows", type=int, default=0, help="Stop after N kept rows (0=all).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    restrict = None
    if args.restrict_prompts:
        restrict = {json.loads(l)["prompt"] for l in open(args.restrict_prompts) if l.strip()}
        print(f"restrict_prompts: {len(restrict)} prompts from {args.restrict_prompts}")

    import datasets
    datasets.disable_progress_bars()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    shards = sorted(glob.glob(os.path.join(ana.OPENR1_CACHE, "*.arrow")))
    ds = datasets.concatenate_datasets([datasets.Dataset.from_file(s) for s in shards])
    print(f"OpenR1 rows: {len(ds)} -> {args.out}  (mode={args.response_mode}, canvas={args.canvas_length}, "
          f"strict_mathverify={args.strict_mathverify})")

    n_keep = n_long = n_empty = n_wrong = n_skip = 0
    with open(args.out, "w") as f:
        for r in ds:
            if restrict is not None and (r.get("problem") or "") not in restrict:
                n_skip += 1
                continue
            if args.strict_mathverify and args.response_mode == "r1_cot" \
                    and not messages_is_mathverify_correct(r):
                n_wrong += 1
                continue
            reasoning = build_reasoning(r, args.response_mode)
            final = clean_answer(r.get("answer"))
            if not reasoning or not final:
                n_empty += 1
                continue
            completion = ana.build_completion(reasoning, final)
            if len(tok.encode(completion, add_special_tokens=False)) + 1 > args.canvas_length:
                n_long += 1
                continue
            f.write(json.dumps({"uuid": r.get("uuid"), "prompt": r.get("problem") or "",
                                "completion": completion}) + "\n")
            n_keep += 1
            if args.max_rows and n_keep >= args.max_rows:
                break
    print(f"kept {n_keep} | dropped {n_long} (>canvas) | {n_empty} (empty reasoning/answer) | "
          f"{n_wrong} (not math_verify-correct) | {n_skip} (not in restrict set) -> {args.out}")


if __name__ == "__main__":
    main()
