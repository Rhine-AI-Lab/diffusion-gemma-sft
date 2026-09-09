"""Analyze the allenai Tmax SFT datasets: schema, the prompt/trajectory structure, and which
field is the SFT target. Read-only (streams from the HF hub).

  HF_HUB_OFFLINE=0 python dataset/analyze_tmax.py
"""
import json
import os
import statistics as st
import sys

import datasets

datasets.disable_progress_bars()

SFT_REPOS = ["allenai/tmax-sft", "allenai/tmax-sft-big"]
SPEC_REPO = "allenai/TMax-SFT-16.5K"
TMAX_TOK = "allenai/tmax-sft-8b"          # native (Qwen-family) chat template + tool support
N_STATS = int(os.environ.get("N_STATS", "300"))


def peek(repo):
    cfgs = datasets.get_dataset_config_names(repo)
    ds = datasets.load_dataset(repo, cfgs[0] if cfgs else None, split="train", streaming=True)
    it = iter(ds)
    return cfgs, next(it), it


def trunc(s, n=110):
    s = str(s).replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + " …"


print("#" * 80)
print("# 1. SCHEMAS")
for repo in SFT_REPOS + [SPEC_REPO]:
    print("\n=== " + repo + " ===")
    try:
        cfgs, row, _ = peek(repo)
        print("  configs:", cfgs)
        print("  fields :", list(row.keys()))
        for k, v in row.items():
            ln = len(v) if hasattr(v, "__len__") else "-"
            print(f"    - {k:16s} {type(v).__name__:5s} len={ln}")
    except Exception as e:
        print("  ERROR:", type(e).__name__, str(e)[:160])

# ------------------------------------------------------------------ example trajectory
print("\n" + "#" * 80)
print("# 2. EXAMPLE ROW  (allenai/tmax-sft)  — the SFT `messages` trajectory")
_, row, stream = peek("allenai/tmax-sft")
msgs = row["messages"]
tools = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
print(f"\n  tools = {json.dumps(tools, indent=2)[:600]}")
print(f"\n  metadata = {trunc(row.get('metadata'), 200)}")
print(f"\n  messages: {len(msgs)} turns")
from collections import Counter
print("  roles   :", dict(Counter(m['role'] for m in msgs)))
print("\n  --- turn-by-turn (role : content preview) ---")
for i, m in enumerate(msgs):
    extra = ""
    if m.get("tool_calls"):
        extra = "  tool_calls=" + trunc(json.dumps(m["tool_calls"]), 80)
    print(f"  [{i:2d}] {m['role']:9s}: {trunc(m.get('content') or '', 130)}{extra}")

# ------------------------------------------------------------------ chat-template render
print("\n" + "#" * 80)
print("# 3. RENDERED PROMPT  (apply_chat_template with tools — what the model trains on)")
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TMAX_TOK)
    rendered = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
    print(f"\n  [tokenizer={TMAX_TOK}]  rendered length = {len(tok.encode(rendered))} tokens\n")
    print("  ----- first 1600 chars -----")
    print(rendered[:1600])
    print("\n  ----- last 500 chars -----")
    print(rendered[-500:])
except Exception as e:
    print("  (render skipped:", type(e).__name__, str(e)[:160], ")")
    tok = None

# ------------------------------------------------------------------ length + turns stats
print("\n" + "#" * 80)
print(f"# 4. LENGTH / TURNS over ~{N_STATS} tmax-sft rows")
lens, turns = [], []
rows = [row]
for r in stream:
    rows.append(r)
    if len(rows) >= N_STATS:
        break
for r in rows:
    m = r["messages"]
    turns.append(len(m))
    if tok is not None:
        try:
            t = json.loads(r["tools"]) if isinstance(r["tools"], str) else r["tools"]
            lens.append(len(tok.encode(tok.apply_chat_template(m, tools=t, tokenize=False))))
        except Exception:
            pass
def stats(xs):
    xs = sorted(xs)
    return dict(n=len(xs), mean=round(st.mean(xs)), median=st.median(xs),
                p90=xs[int(0.9*len(xs))], max=xs[-1]) if xs else {}
print("  turns/trajectory:", stats(turns))
print("  tokens/trajectory (rendered):", stats(lens) if lens else "(no tokenizer)")

sys.stdout.flush()
os._exit(0)   # avoid the pyarrow/streaming GIL crash at interpreter teardown
