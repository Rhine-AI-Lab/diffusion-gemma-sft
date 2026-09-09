"""Cheap base-model triage probe on glaive-function-calling-v2: given a
tool-schema system prompt + a user request, does the model extract the
correct {name, arguments} JSON?"""

from __future__ import annotations

import json
import re
import urllib.request

PROMPT_SUFFIX = (
    "\n\nRespond with ONLY the function call as JSON, in the form "
    '{"name": "<function_name>", "arguments": {<key>: <value>, ...}}. '
    "No other text."
)


def extract_call(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    if "name" not in obj:
        return None
    args = obj.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            pass
    return obj.get("name"), args


def main(api_base="http://localhost:8015/v1", model="diffgemma", max_tokens=300):
    samples = json.load(open("codefix_eval/glaive_probe_samples.json"))
    n_correct = 0
    n_name_only = 0
    for i, ex in enumerate(samples):
        prompt = ex["system"] + PROMPT_SUFFIX + f"\n\nUSER: {ex['user']}\nASSISTANT:"
        body = json.dumps(
            {"model": model, "prompt": prompt, "temperature": 0.0, "max_tokens": max_tokens}
        ).encode()
        try:
            out = json.load(
                urllib.request.urlopen(
                    urllib.request.Request(
                        f"{api_base}/completions", body, {"Content-Type": "application/json"}
                    ),
                    timeout=120,
                )
            )
            raw = out["choices"][0]["text"] or ""
        except Exception as e:
            raw = ""
            print(f"[{i}] REQUEST ERROR {e}")
        parsed = extract_call(raw)
        name_ok = parsed is not None and parsed[0] == ex["name"]
        full_ok = name_ok and parsed[1] == ex["arguments"]
        n_correct += full_ok
        n_name_only += name_ok
        print(f"[{i+1}/{len(samples)}] target={ex['name']:28s} name_ok={name_ok} full_ok={full_ok}")

    print(f"\n=== name correct: {n_name_only}/{len(samples)} ({100*n_name_only/len(samples):.1f}%) ===")
    print(f"=== name+args exact: {n_correct}/{len(samples)} ({100*n_correct/len(samples):.1f}%) ===")


if __name__ == "__main__":
    main()
