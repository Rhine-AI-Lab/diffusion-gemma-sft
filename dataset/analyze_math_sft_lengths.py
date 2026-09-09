"""Analyse reasoning-trace token-length distributions for candidate math SFT datasets.

For each dataset we build the completion in the EXACT format the DiffusionGemma SFT
collator + math eval expect (matches `MATH_PROMPT` / `sft_eval_math._score_math`):

    <reasoning>
    {reasoning trace}
    </reasoning>
    <answer>
    \\boxed{final}
    </answer>

then tokenize it with the DiffusionGemma tokenizer (add_special_tokens=False, +1 for the
EOS the collator appends) and report percentiles and the fraction of rows whose completion
fits each candidate canvas {256, 512, 1024, 2048}. This drives the length-scaling / subset
selection for the SFT run (single-canvas SFT right-truncates the completion, so anything
over the canvas loses its \\boxed answer -- see dataset/make_gsm8k_jsonl.py).

Only OpenR1-Math-220k, s1K, and OlympiadBench carry genuine chain-of-thought traces;
CollegeMATH and ORZ57K are question + short-answer only (empty reasoning) and are reported
as such.

Run (gdsd .venv):
  python dataset/analyze_math_sft_lengths.py --sample 4000 --out_dir eval_out
"""

import argparse
import glob
import json
import os
import random

import numpy as np
from transformers import AutoTokenizer

OPENR1_CACHE = (
    "/users/staff/dmi-dmi/bankes0000/mdpo-private-repo/cache/"
    "open-r1___open_r1-math-220k/default/0.0.0/"
    "e4e141ec9dea9f8326f4d347be56105859b2bd68"
)

CANVAS_TARGETS = [256, 512, 1024, 2048]
# histogram bin edges (token counts) shared across datasets for the overlay chart
HIST_EDGES = [0, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 1_000_000]


# ----------------------------------------------------------------------------- helpers
def last_boxed(s: str):
    """Return the payload inside the last \\boxed{...} in s, else None."""
    if not s:
        return None
    idx = s.rfind("\\boxed")
    if idx < 0:
        return None
    i = s.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1 : j]
    return None


def final_boxed(text: str, fallback: str) -> str:
    """Best-effort final answer: last \\boxed in text, else the fallback string."""
    b = last_boxed(text)
    if b is not None:
        return b.strip()
    fb = last_boxed(fallback)
    if fb is not None:
        return fb.strip()
    return (fallback or "").strip()


def strip_think(text: str) -> str:
    """Drop literal <think>/</think> tags (OpenR1 R1 traces) but keep the reasoning body."""
    return text.replace("<think>", "").replace("</think>", "").strip()


def build_completion(reasoning: str, final: str) -> str:
    reasoning = (reasoning or "").strip()
    final = (final or "").strip()
    return f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n\\boxed{{{final}}}\n</answer>"


# ----------------------------------------------------------------------------- loaders
# Each loader yields dicts: {"reasoning": str, "final": str, "question": str, "has_cot": bool}


def load_openr1(limit):
    import datasets

    datasets.disable_progress_bars()
    shards = sorted(glob.glob(os.path.join(OPENR1_CACHE, "*.arrow")))
    ds = datasets.concatenate_datasets([datasets.Dataset.from_file(s) for s in shards])
    n = len(ds)
    idxs = _sample_idxs(n, limit)
    for i in idxs:
        r = ds[int(i)]
        msgs = r.get("messages") or []
        trace = msgs[-1]["content"] if msgs else (r.get("generations") or [r.get("solution") or ""])[0]
        trace = strip_think(trace)
        final = final_boxed(trace, r.get("answer") or "")
        yield {"reasoning": trace, "final": final, "question": r.get("problem") or "", "has_cot": True}


def load_s1k(limit):
    import datasets

    datasets.disable_progress_bars()
    ds = datasets.load_dataset("simplescaling/s1K", split="train")
    for i in _sample_idxs(len(ds), limit):
        r = ds[int(i)]
        tj = r.get("thinking_trajectories") or []
        reasoning = tj[0] if tj else ""
        final = final_boxed(r.get("attempt") or "", r.get("solution") or "")
        yield {"reasoning": reasoning, "final": final, "question": r.get("question") or "", "has_cot": True}


def load_olympiad(limit):
    import datasets

    datasets.disable_progress_bars()
    ds = datasets.load_dataset("knoveleng/OlympiadBench", split="train")
    for i in _sample_idxs(len(ds), limit):
        r = ds[int(i)]
        sol = r.get("solution") or []
        reasoning = sol[0] if isinstance(sol, list) and sol else (sol if isinstance(sol, str) else "")
        fa = r.get("final_answer") or []
        fa = fa[0] if isinstance(fa, list) and fa else (fa if isinstance(fa, str) else "")
        final = final_boxed(r.get("answer") or "", fa)
        yield {"reasoning": reasoning, "final": final, "question": r.get("question") or "", "has_cot": True}


def load_collegemath(limit):
    import datasets

    datasets.disable_progress_bars()
    ds = datasets.load_dataset("realtreetune/college_math", split="test")
    for i in _sample_idxs(len(ds), limit):
        r = ds[int(i)]
        final = final_boxed(r.get("solution") or "", r.get("answer") or "")
        # answer-only benchmark: no chain-of-thought available
        yield {"reasoning": "", "final": final, "question": r.get("problem") or "", "has_cot": False}


def load_orz(limit):
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        "Open-Reasoner-Zero/orz_math_57k_collection",
        "orz_math_57k_collection.json",
        repo_type="dataset",
    )
    with open(p) as f:
        data = json.load(f)
    for i in _sample_idxs(len(data), limit):
        pair = data[int(i)]
        q = pair[0].get("value", "")
        gt = pair[1].get("ground_truth", {}).get("value", "") if len(pair) > 1 else ""
        yield {"reasoning": "", "final": gt, "question": q, "has_cot": False}


DATASETS = {
    "OpenR1-Math-220k": ("open-r1/OpenR1-Math-220k (local cache, R1 <think> traces)", load_openr1),
    "s1K": ("simplescaling/s1K (thinking_trajectories)", load_s1k),
    "OlympiadBench": ("knoveleng/OlympiadBench train (en math text, worked solution)", load_olympiad),
    "CollegeMATH": ("realtreetune/college_math test (answer-only, no CoT)", load_collegemath),
    "ORZ57K": ("Open-Reasoner-Zero/orz_math_57k (answer-only, no CoT)", load_orz),
}

_RNG = random.Random(0)


def _sample_idxs(n, limit):
    if limit is None or n <= limit:
        return list(range(n))
    return sorted(_RNG.sample(range(n), limit))


# ----------------------------------------------------------------------------- analysis
def analyse(tok, name, desc, loader, limit):
    comp_lens, prompt_lens, has_cot = [], [], True
    example = None
    n_seen = 0
    for row in loader(limit):
        comp = build_completion(row["reasoning"], row["final"])
        clen = len(tok.encode(comp, add_special_tokens=False)) + 1  # +1 EOS (collator)
        plen = len(tok.encode(row["question"], add_special_tokens=False))
        comp_lens.append(clen)
        prompt_lens.append(plen)
        has_cot = has_cot and row["has_cot"]
        if example is None:
            example = comp
        n_seen += 1
    a = np.array(comp_lens)
    p = np.array(prompt_lens)
    stats = {
        "dataset": name,
        "desc": desc,
        "has_cot": has_cot,
        "n_sampled": int(n_seen),
        "comp_mean": float(a.mean()),
        "comp_median": float(np.median(a)),
        "comp_p75": float(np.percentile(a, 75)),
        "comp_p90": float(np.percentile(a, 90)),
        "comp_p95": float(np.percentile(a, 95)),
        "comp_p99": float(np.percentile(a, 99)),
        "comp_max": int(a.max()),
        "prompt_p95": float(np.percentile(p, 95)),
        "prompt_max": int(p.max()),
        "frac_le": {str(t): float((a <= t).mean()) for t in CANVAS_TARGETS},
        "hist": np.histogram(a, bins=HIST_EDGES)[0].tolist(),
        "example": example,
    }
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="unsloth/diffusiongemma-26B-A4B-it")
    ap.add_argument("--sample", type=int, default=4000, help="max rows tokenized per dataset")
    ap.add_argument("--out_dir", default="eval_out")
    ap.add_argument("--only", nargs="*", default=None, help="subset of dataset keys to run")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    keys = args.only or list(DATASETS)
    results, examples = [], {}
    for name in keys:
        desc, loader = DATASETS[name]
        print(f"\n>>> {name}: {desc}")
        try:
            st = analyse(tok, name, desc, loader, args.sample)
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"    FAILED: {type(e).__name__}: {str(e)[:200]}")
            results.append({"dataset": name, "desc": desc, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            continue
        examples[name] = st.pop("example")
        results.append(st)
        fr = st["frac_le"]
        print(
            f"    n={st['n_sampled']} cot={st['has_cot']} median={st['comp_median']:.0f} "
            f"p90={st['comp_p90']:.0f} max={st['comp_max']} | "
            f"<=256:{fr['256']:.0%} <=512:{fr['512']:.0%} <=1024:{fr['1024']:.0%} <=2048:{fr['2048']:.0%}"
        )

    # ---- write CSV
    csv_path = os.path.join(args.out_dir, "math_sft_length_analysis.csv")
    cols = [
        "dataset", "has_cot", "n_sampled", "comp_mean", "comp_median", "comp_p75",
        "comp_p90", "comp_p95", "comp_p99", "comp_max", "prompt_p95", "prompt_max",
        "le256", "le512", "le1024", "le2048",
    ]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for st in results:
            if "error" in st:
                f.write(f"{st['dataset']},ERROR,,,,,,,,,,,,,,\n")
                continue
            fr = st["frac_le"]
            row = [
                st["dataset"], st["has_cot"], st["n_sampled"],
                f"{st['comp_mean']:.0f}", f"{st['comp_median']:.0f}", f"{st['comp_p75']:.0f}",
                f"{st['comp_p90']:.0f}", f"{st['comp_p95']:.0f}", f"{st['comp_p99']:.0f}", st["comp_max"],
                f"{st['prompt_p95']:.0f}", st["prompt_max"],
                f"{fr['256']:.3f}", f"{fr['512']:.3f}", f"{fr['1024']:.3f}", f"{fr['2048']:.3f}",
            ]
            f.write(",".join(str(x) for x in row) + "\n")

    # ---- write markdown
    md_path = os.path.join(args.out_dir, "math_sft_length_analysis.md")
    with open(md_path, "w") as f:
        f.write("# Math SFT dataset — completion token-length analysis\n\n")
        f.write(f"Tokenizer: `{args.tokenizer}` · sample cap: {args.sample} rows/dataset · ")
        f.write("completion = `<reasoning>…</reasoning><answer>\\boxed{…}</answer>` (+1 EOS).\n\n")
        f.write("| dataset | CoT? | n | median | p90 | p95 | max | ≤256 | ≤512 | ≤1024 | ≤2048 |\n")
        f.write("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n")
        for st in results:
            if "error" in st:
                f.write(f"| {st['dataset']} | — | — | ERROR: {st['error']} |||||||||\n")
                continue
            fr = st["frac_le"]
            f.write(
                f"| {st['dataset']} | {'yes' if st['has_cot'] else 'no'} | {st['n_sampled']} | "
                f"{st['comp_median']:.0f} | {st['comp_p90']:.0f} | {st['comp_p95']:.0f} | {st['comp_max']} | "
                f"{fr['256']:.0%} | {fr['512']:.0%} | {fr['1024']:.0%} | {fr['2048']:.0%} |\n"
            )
        f.write("\n_ ≤N = fraction of completions that fit a canvas of N tokens (rest get right-truncated, losing the \\boxed answer)._\n\n")
        f.write("## Source mapping\n\n")
        for st in results:
            f.write(f"- **{st['dataset']}** — {st['desc']}\n")

    # ---- write hist json (for the artifact)
    hist_path = os.path.join(args.out_dir, "math_sft_length_hist.json")
    with open(hist_path, "w") as f:
        json.dump({"edges": HIST_EDGES, "canvas_targets": CANVAS_TARGETS, "datasets": results}, f, indent=2)

    # ---- write examples
    ex_path = os.path.join(args.out_dir, "math_sft_length_examples.txt")
    with open(ex_path, "w") as f:
        for name, ex in examples.items():
            f.write(f"\n{'='*80}\n### {name}\n{'='*80}\n{ex}\n")

    print(f"\nWrote:\n  {md_path}\n  {csv_path}\n  {hist_path}\n  {ex_path}")


if __name__ == "__main__":
    main()
