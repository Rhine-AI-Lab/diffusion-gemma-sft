"""Evaluate DiffusionGemma on the standard math-reasoning benchmark suite.

Covers MATH-500, Minerva-Math, OlympiadBench, AMC, AIME 2024 and AIME 2025 with a
single ``--dataset`` selector, reusing the shared eval backbone
(``build_argparser``/``build_backend``/``run_eval``) and the ``math_verify`` scorer
(``sft_eval_math._score_math``, boxed-string fallback). All benchmarks use the same
``MATH_PROMPT`` template, optionally transformed by the two prompt-format ablation
knobs (``--think`` / ``--reasoning_prefill``, see ``sft_data_utils.apply_prompt_format``).

Metric protocol (DiffusionGemma has no user-facing temperature -- the entropy-bound
diffusion sampler is fixed at *serve* time):
  * MATH-500 / Minerva / OlympiadBench / AMC -> pass@1 (single run, ``--seed``).
  * AIME 2024 / 2025 -> avg@k over seeds ``0..k-1`` (``--k``), reported as mean +/- std.
    (Diversity comes from the seed; if the served sampler is deterministic at
    temperature 0, pass ``--temperature > 0`` so the k seeds actually diverge.)

Examples:
    # single dataset via a running vLLM server (see serve_diffgemma_vllm.sh):
    python -m DiffGemma.diffgemma_trl.sft_eval_mathbench --backend vllm \
        --dataset math500 --max_new_tokens 512 --reasoning_prefill
    # AIME avg@8:
    python -m DiffGemma.diffgemma_trl.sft_eval_mathbench --backend vllm \
        --dataset aime24 --k 8 --max_new_tokens 2048 --result_file eval_out/aime24.json
"""

from __future__ import annotations

import json
import statistics

from datasets import load_dataset

from .sft_data_utils import MATH_INSTRUCTION, MATH_PROMPT, apply_prompt_format
from .sft_eval_common import build_argparser, build_backend, run_eval
from .sft_eval_math import _score_math

# name -> (hf_id, split, question_field, answer_field, [text_only])
# Field names differ per dataset (verified by inspecting each dataset's schema):
#   AIME_2024 uses capitalised Problem/Answer; OlympiadBench's answer is a list
#   (final_answer) and its rows include multimodal problems we drop (Text-only only).
DATASETS: dict[str, dict] = {
    "math500":  dict(hf_id="HuggingFaceH4/MATH-500", split="test",  q="problem",  a="answer"),
    "minerva":  dict(hf_id="math-ai/minervamath",    split="test",  q="question", a="answer"),
    "olympiad": dict(hf_id="math-ai/olympiadbench",  split="test",  q="question", a="final_answer",
                     text_only=True),
    "amc":      dict(hf_id="math-ai/amc23",          split="test",  q="question", a="answer"),
    "aime24":   dict(hf_id="Maxwell-Jia/AIME_2024",  split="train", q="Problem",  a="Answer"),
    "aime25":   dict(hf_id="math-ai/aime25",         split="test",  q="problem",  a="answer"),
}
# Small, high-variance sets -> report avg@k (mean +/- std) instead of a single pass@1.
AVGK_DATASETS = {"aime24", "aime25"}

# The chat-route instruction (placed in the USER turn) is defined once in
# sft_data_utils.MATH_INSTRUCTION so SFT training and eval share it byte-for-byte.


def _apply_chat(rows, tokenizer, *, enable_thinking, instruction, reasoning_prefill):
    """Replace each row['_prompt'] with the apply_chat_template-rendered prompt.

    The instruction + output-format block goes in the USER turn (blank line, then the
    question), matching the raw MATH_PROMPT. apply_chat_template supplies <bos>, the
    native role separators and (enable_thinking=False) the <|channel>thought<channel|>
    priming; reasoning_prefill then appends <reasoning> so the model opens its
    chain-of-thought immediately. The rendered string already carries <bos>, so the
    vLLM backend must send add_special_tokens=False.
    """
    for r in rows:
        content = f"{instruction}\n\n{r['_prompt']}" if instruction else r["_prompt"]
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
        if reasoning_prefill:
            rendered += "<reasoning>\n"
        r["_prompt"] = rendered
    return rows


def _load_rows(name: str) -> list[dict]:
    """Load a benchmark and normalise every row to ``{_prompt, answer}``."""
    spec = DATASETS[name]
    ds = load_dataset(spec["hf_id"], split=spec["split"])
    if spec.get("text_only"):
        ds = ds.filter(lambda r: r.get("modality") == "Text-only")
    rows: list[dict] = []
    for r in ds:
        ans = r[spec["a"]]
        if isinstance(ans, (list, tuple)):  # OlympiadBench final_answer is a list
            ans = ", ".join(str(x) for x in ans)
        rows.append({"_prompt": r[spec["q"]], "answer": str(ans)})
    return rows


def _eval_avgk(rows, template, backend, args, name: str, k: int) -> dict:
    """Run the eval k times over seeds 0..k-1; report mean +/- std of accuracy.

    ``run_eval``'s own per-run result_file write is suppressed during the loop; the
    aggregate (mean/std/per-seed) is written to ``--result_file`` here instead.
    """
    result_file = args.result_file
    args.result_file = None  # suppress per-seed writes; we write the aggregate below
    scores: list[float] = []
    for s in range(k):
        args.seed = s
        if hasattr(backend, "seed"):  # vLLM backend: vary the per-request seed too
            backend.seed = s
        sc = run_eval(rows, template, _score_math, backend, args, label=f"{name}[seed {s}]")
        scores.append(sc)
    args.result_file = result_file
    mean = statistics.mean(scores)
    std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    per_seed = [round(x * 100, 1) for x in scores]
    print(f"{name} | avg@{k} = {mean * 100:.1f}% +/- {std * 100:.1f}  (per-seed: {per_seed})")
    res = {"label": name, "avg_at_k": k, "mean": mean, "std": std,
           "scores": scores, "n": len(rows)}
    if result_file:
        res["format"] = {"think": args.think, "reasoning_prefill": args.reasoning_prefill}
        with open(result_file, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"[mathbench] wrote {result_file}")
    return res


def main():
    ap = build_argparser(
        "DiffusionGemma math-benchmark eval (MATH-500/Minerva/OlympiadBench/AMC/AIME)"
    )
    ap.add_argument("--dataset", required=True,
                    help="One benchmark name, a comma-separated list, or 'all'. "
                         f"Choices: {', '.join(DATASETS)}.")
    ap.add_argument("--think", action="store_true",
                    help="Prepend the <|think|> token to the prompt (format ablation).")
    ap.add_argument("--reasoning_prefill", action="store_true",
                    help="Open a <reasoning> block at the start of the model turn (format ablation).")
    ap.add_argument("--k", type=int, default=8,
                    help="avg@k samples for AIME datasets (seeds 0..k-1). Ignored elsewhere.")
    # DEFAULT route is the model's native apply_chat_template (thinking channel).
    # --chat_template is kept as an explicit (redundant) opt-in; --raw_template opts
    # out to the hand-written <|turn> MATH_PROMPT (+ --think/--reasoning_prefill).
    ap.add_argument("--chat_template", dest="chat_template", action="store_true", default=True,
                    help="(default) build prompts with the native apply_chat_template.")
    ap.add_argument("--raw_template", dest="chat_template", action="store_false",
                    help="Use the raw <|turn> MATH_PROMPT instead (enables --think/--reasoning_prefill).")
    ap.add_argument("--enable_thinking", action="store_true",
                    help="apply_chat_template enable_thinking flag (chat route only; "
                         "enable_thinking=1 dropped MATH-500 to 16%% in testing, so default off).")
    ap.add_argument("--system_prompt", default=MATH_INSTRUCTION,
                    help="Instruction placed in the user turn for the chat route "
                         "(includes the <reasoning>/<answer> output-format block).")
    args = ap.parse_args()

    if args.dataset == "all":
        names = list(DATASETS)
    else:
        names = [n.strip() for n in args.dataset.split(",") if n.strip()]
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        raise SystemExit(f"unknown dataset(s): {unknown}; choices: {list(DATASETS)}")

    backend = build_backend(args)

    tokenizer = None
    if args.chat_template:
        from transformers import AutoTokenizer
        tok_src = args.vllm_model if args.backend == "vllm" else args.model_path
        tokenizer = AutoTokenizer.from_pretrained(tok_src)
        template = "{text}"  # each row already carries the fully-rendered prompt
        if hasattr(backend, "add_special_tokens"):
            backend.add_special_tokens = False  # apply_chat_template already adds <bos>
        fmt_tag = (f"chat_template enable_thinking={int(args.enable_thinking)} "
                   f"reasoning_prefill={int(args.reasoning_prefill)}")
    else:
        template = apply_prompt_format(
            MATH_PROMPT, think=args.think, reasoning_prefill=args.reasoning_prefill)
        fmt_tag = f"raw think={int(args.think)} reasoning_prefill={int(args.reasoning_prefill)}"

    print(f"[mathbench] datasets={names} | format: {fmt_tag} | backend={args.backend} "
          f"| max_new_tokens={args.max_new_tokens}")

    if len(names) > 1 and args.result_file:
        print("[warn] --result_file with multiple datasets: only the LAST dataset's "
              "result is kept. Run one dataset per --result_file for aggregation.")

    for name in names:
        rows = _load_rows(name)
        if args.limit:
            rows = rows[: args.limit]
        if tokenizer is not None:
            _apply_chat(rows, tokenizer, enable_thinking=args.enable_thinking,
                        instruction=args.system_prompt,
                        reasoning_prefill=args.reasoning_prefill)
        # avg@k datasets (AIME): _eval_avgk writes the aggregate to --result_file.
        # Everything else: run_eval writes its own {label,correct,n,score} JSON
        # (the schema the shard aggregator in local/*.sh expects).
        if name in AVGK_DATASETS and args.k > 1:
            _eval_avgk(rows, template, backend, args, name, args.k)
        else:
            run_eval(rows, template, _score_math, backend, args, label=name)


if __name__ == "__main__":
    main()
