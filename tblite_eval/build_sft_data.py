"""Extract terminal-agent SFT training pairs from Harbor rollout trajectories.

For each cleanly-parsed agent turn (harbor successfully extracted analysis/
plan/commands from the model's response) in any collected trajectory, emits
one training example:

  messages:   the exact conversation history the model saw up to and
              including that turn's user/tool-observation message (role/
              content dicts, matching Harbor's own Chat._messages format
              exactly -- requires trajectories collected with
              `--ak save_raw_content_in_trajectory=true` so prior assistant
              turns store the model's literal raw text, the same text that
              was actually fed back into subsequent calls at runtime).
  completion: a canonical, minimally-escaped JSON string re-serialized from
              the turn's *parsed* fields (analysis/plan/commands/
              task_complete) -- NOT the model's raw text. This is the fix:
              train on what the model meant to say, formatted perfectly,
              regardless of whatever raw-text quirks (stray escaping, code
              fences) it actually produced that one time.

We deliberately do not filter by final task success/reward: the failure mode
we're targeting is protocol adherence (staying in valid-JSON-action mode),
not task-solving skill, so every cleanly-parsed turn from any trajectory
(pass or fail) is a valid positive example of the format we want reinforced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _split_analysis_plan(message: str) -> tuple[str, str]:
    """Reverse Harbor's `"Analysis: {a}\\nPlan: {p}"` reconstruction.

    Only used as a fallback when raw content isn't available for a prior
    turn (older trajectories collected without save_raw_content_in_trajectory).
    """
    analysis, plan = "", ""
    if message.startswith("Analysis: "):
        rest = message[len("Analysis: ") :]
        if "\nPlan: " in rest:
            analysis, plan = rest.split("\nPlan: ", 1)
        else:
            analysis = rest
    elif message.startswith("Plan: "):
        plan = message[len("Plan: ") :]
    return analysis, plan


def _canonical_completion(step: dict) -> str | None:
    """Re-serialize a successfully-parsed agent step into clean target JSON.

    Returns None if the step has no tool_calls (parse failed) or the
    tool_calls don't decompose into the expected bash_command/
    mark_task_complete shape.
    """
    tool_calls = step.get("tool_calls")
    if not tool_calls:
        return None

    commands = []
    task_complete = False
    for tc in tool_calls:
        fn = tc.get("function_name")
        args = tc.get("arguments") or {}
        if fn == "bash_command":
            commands.append(
                {
                    "keystrokes": args.get("keystrokes", ""),
                    "duration": args.get("duration", 1.0),
                }
            )
        elif fn == "mark_task_complete":
            task_complete = True
        else:
            # Unknown synthetic tool -- don't silently mis-teach the format.
            return None

    message = step.get("message") or ""
    analysis, plan = _split_analysis_plan(message)

    return json.dumps(
        {
            "analysis": analysis,
            "plan": plan,
            "commands": commands,
            "task_complete": task_complete,
        },
        ensure_ascii=False,
    )


def extract_examples(trajectory_path: Path) -> list[dict]:
    data = json.loads(trajectory_path.read_text())
    steps = data["steps"]

    examples = []
    history: list[dict] = []
    for step in steps:
        source = step.get("source")
        message = step.get("message") or ""

        if source == "user":
            history.append({"role": "user", "content": message})
            continue

        if source != "agent":
            # system/subagent-handoff steps: fold into history as-is so later
            # turns' context matches runtime; not a training target itself.
            history.append({"role": step.get("source", "user"), "content": message})
            continue

        completion = _canonical_completion(step)
        if completion is not None:
            examples.append(
                {
                    "messages": [dict(m) for m in history],
                    "completion": completion,
                    "source_trajectory": str(trajectory_path),
                    "step_id": step.get("step_id"),
                }
            )

        # Advance history with the RAW text so later turns' prompts match
        # what the model actually saw (requires save_raw_content_in_trajectory).
        history.append({"role": "assistant", "content": message})

        # Feed the next turn's observation back in as the next user message
        # (mirrors Chat._messages: every assistant turn is followed by the
        # next user-role prompt, which is the tool observation/feedback).
        obs = step.get("observation")
        if obs and obs.get("results"):
            obs_text = "\n".join(
                r.get("content", "") for r in obs["results"] if r.get("content")
            )
            if obs_text:
                history.append({"role": "user", "content": obs_text})

    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-dir", action="append", required=True, help="Harbor jobs-dir root(s) to scan (repeatable)")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    args = ap.parse_args()

    n_trajectories = 0
    n_examples = 0
    with open(args.output, "w") as out:
        for jobs_dir in args.jobs_dir:
            for traj_path in sorted(Path(jobs_dir).glob("*/agent/trajectory.json")):
                n_trajectories += 1
                for ex in extract_examples(traj_path):
                    out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    n_examples += 1

    print(f"Scanned {n_trajectories} trajectories -> {n_examples} training examples -> {args.output}")


if __name__ == "__main__":
    main()
