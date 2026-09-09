"""Build restricted-function-calling SFT data from API-Bank level-1
(liminghao1630/API-Bank, training-data/lv1-train.json): each example
already presents a small (2-5), scenario-scoped set of tool schemas
in-context and a user request mapping to exactly one correct call.

Converts API-Bank's native `API-Request: [ApiName(key='value', ...)]`
target into the clean {"tool": ..., "arguments": {...}} JSON schema.
BFCL is reserved entirely for held-out eval (never touched here).
"""

from __future__ import annotations

import json
import re

from huggingface_hub import hf_hub_download

_PROMPT_TEMPLATE = """You have access to the following tools:

{tools}

User: {user}

Respond with ONLY the tool call as JSON, in the form {{"tool": "<tool_name>", "arguments": {{<key>: <value>, ...}}}}."""


def _parse_call(output: str):
    """Parse `Name(key1='v1', key2='v2')`. Values are treated as free text
    between one key= boundary and the next (rather than naive quote
    matching), since API-Bank's values routinely contain unescaped
    apostrophes (e.g. `patient_id='user's ID'`) that would otherwise
    truncate a quote-delimited regex mid-value."""
    m = re.search(r"API-Request:\s*\[([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\]", output.strip(), re.DOTALL)
    if not m:
        return None
    name, argstr = m.group(1), m.group(2)
    key_positions = [(km.group(1), km.start(), km.end()) for km in re.finditer(r"(\w+)\s*=\s*['\"]", argstr)]
    if not key_positions:
        return name, {}
    args = {}
    for i, (key, _, val_start) in enumerate(key_positions):
        val_end = key_positions[i + 1][1] if i + 1 < len(key_positions) else len(argstr)
        raw_val = argstr[val_start:val_end]
        # strip a trailing quote(+comma/space) left over from the value's own closing quote
        raw_val = re.sub(r"['\"]\s*,?\s*$", "", raw_val)
        args[key] = raw_val
    return name, args


def _extract_tool_schemas(input_text: str) -> list[dict]:
    """API-Bank's `input` field is one or more concatenated top-level JSON
    objects (tool schemas), each shaped {"apiCode", "description",
    "parameters", "response"}, followed by the dialogue turn."""
    tools = []
    dec = json.JSONDecoder()
    idx = 0
    text = input_text
    while idx < len(text):
        ch = text[idx]
        if ch != "{":
            idx += 1
            continue
        try:
            obj, end = dec.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx += 1
            continue
        if isinstance(obj, dict) and "apiCode" in obj:
            tools.append(obj)
        idx = end
    return tools


def _extract_user_turn(input_text: str) -> str | None:
    m = re.search(r"User:\s*(.*?)(?:\nAI:|\nGenerate API Request:|$)", input_text, re.DOTALL)
    return m.group(1).strip() if m else None


def _format_tools(tools: list[dict]) -> str:
    lines = []
    for t in tools:
        params = t.get("parameters", {})
        param_desc = ", ".join(
            f"{k}: {v.get('type', 'string')}" + (" (required)" if v.get("required") else "")
            for k, v in params.items()
        )
        lines.append(f"- {t['apiCode']}({param_desc}): {t.get('description', '')}")
    return "\n".join(lines)


def main():
    path = hf_hub_download("liminghao1630/API-Bank", "training-data/lv1-train.json", repo_type="dataset")
    data = json.load(open(path))

    out = []
    n_skipped = 0
    for ex in data:
        if not ex["output"].strip().startswith("API-Request:"):
            continue
        # Restrict to genuinely single-turn requests: API-Bank's raw examples
        # are frequently multi-turn dialogues where the labeled call answers
        # the LAST user turn (and sometimes invents content, like a session
        # description, with no textual basis at all) -- naive first-User-turn
        # extraction silently mismatches prompt and completion. Single-turn
        # examples sidestep that ambiguity by construction and match the
        # one-shot request->call shape of the task spec.
        if ex["input"].count("User:") != 1:
            n_skipped += 1
            continue
        call = _parse_call(ex["output"])
        if call is None:
            n_skipped += 1
            continue
        name, args = call
        tools = _extract_tool_schemas(ex["input"])
        matching = [t for t in tools if t["apiCode"] == name]
        if not matching:
            n_skipped += 1
            continue
        # Reject examples where the labeled call uses argument names outside
        # the tool's own declared parameter schema -- a small (~0.3%) but
        # real class of corrupted labels in API-Bank (e.g. a call whose args
        # were borrowed from the tool's *response* schema instead of its
        # *parameter* schema). Keeping these would violate the closed
        # argument-vocabulary requirement.
        schema_keys = set(matching[0].get("parameters", {}).keys())
        if not set(args.keys()).issubset(schema_keys):
            n_skipped += 1
            continue
        user_turn = _extract_user_turn(ex["input"])
        if not user_turn:
            n_skipped += 1
            continue

        prompt = _PROMPT_TEMPLATE.format(tools=_format_tools(tools), user=user_turn)
        completion = json.dumps({"tool": name, "arguments": args}, ensure_ascii=False)
        out.append({"prompt": prompt, "completion": completion})

    with open("codefix_eval/apibank_sft.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{len(out)} training examples (skipped {n_skipped}) -> codefix_eval/apibank_sft.jsonl")
    print("\nsample:")
    print(out[0]["prompt"][:500])
    print("-->", out[0]["completion"])


if __name__ == "__main__":
    main()
