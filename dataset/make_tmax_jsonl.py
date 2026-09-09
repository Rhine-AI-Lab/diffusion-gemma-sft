"""Convert allenai/tmax-sft trajectories -> block-SFT jsonl for DiffusionGemma.

The DiffGemma chat template drops the assistant THOUGHT (renders only tool_calls), so we render
each trajectory with the **tmax/Qwen** template (`allenai/tmax-sft-8b`), which keeps THOUGHT +
tool_call + tool_response, then tokenize that string with the **DiffGemma** tokenizer (the model we
train). Each output row =
    {"completion": <bos + rendered trajectory, trimmed to end on an assistant turn <4096 DiffGemma
                    tokens>, "assistant_spans": [[s,e], ...]}
where each span is the DiffGemma-token span of one assistant response's content (THOUGHT + tool_call
+ `<|im_end|>`). The collator samples one span, tiles it into canvas_length blocks, samples a block
(denoiser target), and encodes everything before it.

  HF_HUB_OFFLINE=0 python dataset/make_tmax_jsonl.py --out dataset/tmax_sft_only_success.jsonl
"""
import argparse
import json
import math
import os
import re
import statistics as st

import datasets
from transformers import AutoTokenizer

datasets.disable_progress_bars()
ASST_HDR = re.compile(r"<\|im_start\|>assistant\n")
IM_END = "<|im_end|>"


def clean(messages):
    """Normalise tmax messages: keep {role, content, reasoning_content?, tool_calls?}; drop the
    stray Qwen fields that are None (tool_call_ids) and give tool turns the tool name (all `bash`).

    `reasoning_content` carries the assistant's THOUGHT and MUST be forwarded: the template only
    emits the `<think>...</think>` block when it is a non-empty string, and otherwise renders a bare
    `<|im_start|>assistant\\n{content}` -- which on this (thinking_only) config silently strips the
    chain-of-thought from ~90% of assistant turns."""
    out, last = [], "bash"
    for m in messages:
        d = {"role": m["role"], "content": m.get("content") or ""}
        if m["role"] == "assistant":
            rc = m.get("reasoning_content")
            if isinstance(rc, str) and rc.strip():   # the template strips \n and tests truthiness
                d["reasoning_content"] = rc
        if m["role"] == "assistant" and m.get("tool_calls"):
            d["tool_calls"] = m["tool_calls"]
            try:
                last = m["tool_calls"][0]["function"]["name"]
            except Exception:
                pass
        if m["role"] == "tool":
            d["name"] = last
        out.append(d)
    return out


def convert_row(qtok, dtok, messages, tools, max_len):
    messages = clean(messages)
    if not any(m["role"] == "assistant" for m in messages):
        return None, "no_assistant"
    try:
        render = qtok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)
    except Exception:
        return None, "render_err"
    s = dtok.bos_token + render
    enc = dtok(s, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    n = len(enc["input_ids"])

    def tok_start(c):  # token index containing char c
        for ti, (a, b) in enumerate(offs):
            if a <= c < b:
                return ti
        return n

    def tok_end(c):    # token index just past the token containing char c-1
        for ti, (a, b) in enumerate(offs):
            if a < c <= b:
                return ti + 1
        return n

    spans = []  # DiffGemma token spans of assistant responses (THOUGHT+tool_call+<|im_end|>)
    for m in ASST_HDR.finditer(s):
        cs = m.end()
        ce = s.find(IM_END, cs)
        if ce < 0:
            continue
        ce += len(IM_END)
        ts, te = tok_start(cs), tok_end(ce)
        if te > ts:
            spans.append((ts, te))
    if not spans:
        return None, "no_span"

    elig = [sp for sp in spans if sp[1] <= max_len]
    if not elig:
        return None, "too_long"           # even the first assistant response exceeds max_len
    cutoff = max(e for _, e in elig)        # token index at end of the last fitting assistant response
    keep = [[a, b] for a, b in spans if b <= cutoff]
    comp_text = s[: offs[cutoff - 1][1]]    # truncate the string at that token boundary
    return {"completion": comp_text, "assistant_spans": keep}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="allenai/tmax-sft")
    ap.add_argument("--config", default="skill_tax_20260505_2.2k_combined_balanced_thinking_only_success")
    ap.add_argument("--render_tokenizer", default="allenai/tmax-sft-8b")     # native tmax/Qwen template
    ap.add_argument("--tokenizer", default="unsloth/diffusiongemma-26B-A4B-it")  # model we train
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--canvas_length", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    qtok = AutoTokenizer.from_pretrained(args.render_tokenizer)
    dtok = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = datasets.load_dataset(args.repo, args.config, split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"tmax rows: {len(ds)} -> {args.out}  (max_len={args.max_len}, L={args.canvas_length})")

    kept, drops = 0, {}
    traj_len, n_resp, resp_len, resp_canv = [], [], [], []
    with open(args.out, "w") as f:
        for r in ds:
            tools = json.loads(r["tools"]) if isinstance(r["tools"], str) else r["tools"]
            row, why = convert_row(qtok, dtok, r["messages"], tools, args.max_len)
            if row is None:
                drops[why] = drops.get(why, 0) + 1
                continue
            f.write(json.dumps(row) + "\n")
            kept += 1
            traj_len.append(len(dtok.encode(row["completion"], add_special_tokens=False)))
            n_resp.append(len(row["assistant_spans"]))
            for a, b in row["assistant_spans"]:
                resp_len.append(b - a)
                resp_canv.append(math.ceil((b - a) / args.canvas_length))

    def stats(x):
        x = sorted(x)
        return {} if not x else dict(n=len(x), mean=round(st.mean(x)), median=st.median(x),
                                     p90=x[int(0.9 * len(x))], max=x[-1])
    print(f"kept {kept} | dropped {drops}")
    print(f"  trajectory tokens (DiffGemma): {stats(traj_len)}")
    print(f"  assistant responses / trajectory: {stats(n_resp)}")
    print(f"  tokens / assistant response: {stats(resp_len)}")
    print(f"  canvases / assistant response: {stats(resp_canv)}")


if __name__ == "__main__":
    main()
