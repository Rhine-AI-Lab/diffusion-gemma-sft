"""Cheap base-model triage probe on CRUXEval (output prediction): given a
short deterministic function + a call, does the model predict the exact
output?"""

from __future__ import annotations

import ast
import json
import random
import re
import urllib.request

import datasets

PROMPT_TEMPLATE = """Here is a Python function and a call to it:

```python
{code}
```

f{input}

What is the exact output of this call? Respond with ONLY the output value, no explanation."""


def _norm(text: str):
    text = text.strip()
    text = re.sub(r"^```(python)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = re.sub(r"^(thought|output|result)\s*[:=]?\s*\n?", "", text, flags=re.IGNORECASE)
    text = text.strip()
    # try the whole thing first, then progressively shorter suffixes from the
    # end (models sometimes prefix stray tokens/lines before the real value;
    # the true answer is usually the LAST parseable literal).
    candidates = [text] + text.rsplit("\n", 1)
    for c in candidates:
        c = c.strip()
        try:
            return ast.literal_eval(c)
        except Exception:
            continue
    return text


def main(api_base="http://localhost:8015/v1", model="diffgemma", n=40, max_tokens=200, seed=5):
    ds = datasets.load_dataset("cruxeval-org/cruxeval", split="test")
    random.seed(seed)
    idxs = random.sample(range(len(ds)), n)

    n_correct = 0
    for j, i in enumerate(idxs):
        ex = ds[i]
        prompt = PROMPT_TEMPLATE.format(code=ex["code"], input=ex["input"])
        body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
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

        pred = _norm(raw)
        target = _norm(ex["output"])
        correct = pred == target
        n_correct += correct
        print(f"[{j+1}/{n}] correct={correct}  pred={str(pred)[:60]!r}  target={str(target)[:60]!r}")

    print(f"\n=== {n_correct}/{n} = {100*n_correct/n:.1f}% ===")


if __name__ == "__main__":
    main()
