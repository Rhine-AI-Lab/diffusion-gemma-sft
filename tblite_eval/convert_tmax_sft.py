"""Convert allenai/tmax-sft (Qwen3.6-27B traces) into terminus-2's JSON-action
schema, producing the same (messages, completion) training-pair format that
build_sft_data.py emits from real terminus-2 rollouts -- so the same collator/
training script consumes either source unchanged.

allenai/tmax-sft uses a different protocol from terminus-2:
  - native OpenAI tool-calling with a single `bash(command: str)` tool
    (one command per turn, no per-command duration), plus a free-text
    "THOUGHT: ..." prefix in `content` and separate `reasoning_content`.
  - task completion is signaled by issuing the literal command
    `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, not a JSON task_complete
    field or dedicated tool.
  - its own format-error recovery turns (e.g. "Format error: Your last
    response did not include a `bash` tool call...") when Qwen itself failed
    to call the tool -- these are dropped entirely (not useful exemplars,
    and Qwen's specific error wording doesn't match terminus-2's).

We rebuild the opening turn using terminus-2's own JSON-plain system prompt
template (not Qwen's system/user framing) so the *history* distribution
matches what the model sees at eval time, and convert each successful Qwen
turn into a canonical terminus-2 JSON completion.
"""

from __future__ import annotations

import argparse
import json

import datasets

_COMPLETE_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

# Same template harbor's terminus-2 ships (templates/terminus-json-plain.txt),
# reproduced here so converted examples match the real system prompt exactly.
_TERMINUS_JSON_PLAIN_TEMPLATE = """You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.

Format your response as JSON with the following structure:

{{
  "analysis": "Analyze the current state based on the terminal output provided. What do you see? What has been accomplished? What still needs to be done?",
  "plan": "Describe your plan for the next steps. What commands will you run and why? Be specific about what you expect each command to accomplish.",
  "commands": [
    {{
      "keystrokes": "ls -la\\n",
      "duration": 0.1
    }},
    {{
      "keystrokes": "cd project\\n",
      "duration": 0.1
    }}
  ],
  "task_complete": true
}}

Required fields:
- "analysis": Your analysis of the current situation
- "plan": Your plan for the next steps
- "commands": Array of command objects to execute

Optional fields:
- "task_complete": Boolean indicating if the task is complete (defaults to false if not present)

Command object structure:
- "keystrokes": String containing the exact keystrokes to send to the terminal (required)
- "duration": Number of seconds to wait for the command to complete (defaults to 1.0 if not present)

IMPORTANT: The text inside "keystrokes" will be used completely verbatim as keystrokes. Write commands exactly as you want them sent to the terminal:
- You must end every command with a newline (\\n) or it will not execute.
- For special key sequences, use tmux-style escape sequences:
  - C-c for Ctrl+C
  - C-d for Ctrl+D

Important notes:
- Each command's keystrokes are sent exactly as written to the terminal
- Extra text before or after the JSON will generate warnings but be tolerated
- The JSON must be valid - use proper escaping for quotes and special characters within strings
- Commands array can be empty if you want to wait without taking action

Task Description:
{instruction}

Current terminal state:
{terminal_state}"""


def _extract_instruction(qwen_user_content: str) -> str:
    prefix = "Please solve this task:\n\n"
    if qwen_user_content.startswith(prefix):
        return qwen_user_content[len(prefix) :]
    return qwen_user_content


def _clean_thought(content: str) -> str:
    content = content.strip()
    if content.upper().startswith("THOUGHT:"):
        content = content[len("THOUGHT:") :].strip()
    return content


def convert_example(ex: dict) -> list[dict]:
    messages = ex["messages"]
    if not messages or messages[0]["role"] != "system":
        return []

    # messages[1] is expected to be the task-instruction user turn.
    task_msg = next((m for m in messages if m["role"] == "user"), None)
    if task_msg is None:
        return []
    instruction = _extract_instruction(task_msg["content"])

    history: list[dict] = [
        {
            "role": "user",
            "content": _TERMINUS_JSON_PLAIN_TEMPLATE.format(
                instruction=instruction, terminal_state="(fresh shell)"
            ),
        }
    ]

    examples = []
    for m in messages[2:]:  # skip system + the initial task-instruction user turn
        role = m["role"]

        if role == "user":
            # Qwen's own format-error recovery turn -- not a real tool
            # observation and not present in terminus-2's protocol. Drop it
            # (and the preceding empty assistant turn, already not emitted
            # as a training target below) rather than splice in wording that
            # doesn't match terminus-2's own error messages.
            if m["content"].startswith("Format error:"):
                continue
            continue  # any other bare user turn: no such case observed; skip defensively

        if role == "tool":
            history.append(
                {"role": "user", "content": f"New Terminal Output:\n\n{m['content']}"}
            )
            continue

        if role != "assistant":
            continue

        tool_calls = m.get("tool_calls") or []
        if not tool_calls:
            # Qwen failed to call the tool this turn -- not a positive
            # exemplar; skip both as a target and from history.
            continue

        command = tool_calls[0]["function"]["arguments"]["command"]
        analysis = _clean_thought(m["content"]) or (m.get("reasoning_content") or "")
        plan = m.get("reasoning_content") or analysis

        if command.strip() == f"echo {_COMPLETE_SENTINEL}":
            completion_obj = {
                "analysis": analysis,
                "plan": plan,
                "commands": [],
                "task_complete": True,
            }
        else:
            completion_obj = {
                "analysis": analysis,
                "plan": plan,
                "commands": [{"keystrokes": command + "\n", "duration": 1.0}],
                "task_complete": False,
            }

        completion = json.dumps(completion_obj, ensure_ascii=False)
        examples.append(
            {
                "messages": [dict(h) for h in history],
                "completion": completion,
                "source_trajectory": ex["metadata"].get("trial_name", ""),
                "step_id": len(examples),
            }
        )

        # Advance history with the CANONICAL terminus-2 JSON (same string we
        # just emitted as this turn's target), NOT Qwen's raw prose.
        #
        # This is the fix for the collapse of the first run: Qwen's raw
        # `content` is plain "THOUGHT: ..." prose and DROPS the actual bash
        # command (it lived in tool_calls). Advancing history with that meant
        # the model never saw the terminus-2 JSON action schema in-context --
        # only as a prediction target -- so at eval (where terminus-2 feeds the
        # model's own JSON back into history) it faced a distribution it was
        # never trained on and drifted to prose/repetition. Appending the
        # canonical `completion` makes the training history self-consistent and
        # byte-compatible with what the model produces + sees at eval.
        # (build_sft_data.py's "use raw content" rationale does NOT transfer
        # here: for our own terminus-2 rollouts raw content already IS this
        # JSON; for Qwen traces it is a foreign protocol.)
        history.append({"role": "assistant", "content": completion})

    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="skill_tax_20260505_2.2k_combined_balanced_thinking_only_success",
    )
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=None, help="Cap number of source trajectories (debug)")
    args = ap.parse_args()

    ds = datasets.load_dataset("allenai/tmax-sft", args.config, split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    n_examples = 0
    with open(args.output, "w") as out:
        for ex in ds:
            for training_ex in convert_example(ex):
                out.write(json.dumps(training_ex, ensure_ascii=False) + "\n")
                n_examples += 1

    print(f"Converted {len(ds)} trajectories -> {n_examples} training examples -> {args.output}")


if __name__ == "__main__":
    main()
