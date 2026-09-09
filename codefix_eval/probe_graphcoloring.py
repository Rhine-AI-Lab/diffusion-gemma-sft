"""Cheap base-model triage probe on planted k-coloring: given the graph +
edge list, does the model output a valid k-coloring (not necessarily the
planted one -- any valid coloring counts)?"""

from __future__ import annotations

import json
import re
import sys
import urllib.request

sys.path.insert(0, "codefix_eval")
from build_graphcoloring_data import is_valid_coloring, parse_coloring  # noqa: E402


def _extract_digits(text: str, n: int) -> str | None:
    m = re.search(r"\b\d{" + str(n) + r"}\b", text)
    if m:
        return m.group(0)
    # fall back: longest run of digits in the text
    runs = re.findall(r"\d+", text)
    if not runs:
        return None
    longest = max(runs, key=len)
    return longest if len(longest) == n else None


def _edges_from_prompt(prompt: str):
    body = prompt.split("Edges:\n")[1].split("\n\nOutput")[0]
    return [tuple(map(int, e.split("-"))) for e in body.split(",")]


def main(path="codefix_eval/graphcoloring_pilot/test_id.jsonl", api_base="http://localhost:8017/v1",
         model="google/diffusiongemma-26B-A4B-it", n=40, max_tokens=300, seed=0):
    import random

    rows = [json.loads(l) for l in open(path)]
    random.seed(seed)
    sample = random.sample(rows, min(n, len(rows)))

    n_valid = 0
    n_parsed = 0
    for j, ex in enumerate(sample):
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
                    timeout=180,
                )
            )
            msg = out["choices"][0]["message"]
            raw = msg.get("content") or msg.get("reasoning") or ""
        except Exception as e:
            raw = ""
            print(f"[{j}] REQUEST ERROR {e}")

        digits = _extract_digits(raw, ex["n"])
        coloring = parse_coloring(digits, ex["n"]) if digits else None
        if coloring is None:
            print(f"[{j+1}/{len(sample)}] n={ex['n']} k={ex['k']} PARSE FAIL raw={raw[:80]!r}")
            continue
        n_parsed += 1
        edges = _edges_from_prompt(ex["prompt"])
        ok, msg_ = is_valid_coloring(edges, coloring, ex["k"], ex["n"])
        n_valid += ok
        print(f"[{j+1}/{len(sample)}] n={ex['n']} k={ex['k']} valid={ok} ({msg_})")

    print(f"\n=== parsed: {n_parsed}/{len(sample)} ({100*n_parsed/len(sample):.1f}%) ===")
    print(f"=== valid coloring: {n_valid}/{len(sample)} ({100*n_valid/len(sample):.1f}%) ===")


if __name__ == "__main__":
    main()
