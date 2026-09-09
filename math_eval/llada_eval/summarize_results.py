#!/usr/bin/env python3
"""
Summarize evaluation results from output_reproduce directory.

For each task_len folder (e.g. code_humaneval_len128), prints a table of
model checkpoints and their performance, with the best model highlighted.
"""

import json
import os
from collections import defaultdict

# Configuration

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_reproduce")

# Primary metric to show for each task (matched by substring)
# Format: {task_key_substring: metric_key_in_results}
PRIMARY_METRIC_MAP = {
    "humaneval":         "pass@1,create_test",
    "humaneval_plus":    "pass@1,create_test",
    "mbpp":              "pass@1,create_test",
    "gsm8k_cot":         "exact_match,flexible-extract",
    "gsm8k":             "exact_match,flexible-extract",
    "minerva_math":      "math_verify,none",          # averaged across subtasks
}

# Helpers

def shorten_model_name(dir_name: str) -> str:
    """Convert a path-like result directory name to a readable short name."""
    # Convert __ separator to /
    path = dir_name.replace("__", "/").strip("/")
    parts = path.split("/")

    # Look for a 'checkpoint-N' part
    for i, part in enumerate(parts):
        if part.startswith("checkpoint-"):
            # Take the experiment folder (2 levels up from checkpoint-N, skip 'checkpoints/')
            if i >= 2 and parts[i - 1] == "checkpoints":
                exp = parts[i - 2]
            elif i >= 1:
                exp = parts[i - 1]
            else:
                exp = "unknown"
            # Truncate long experiment names from the left (keep the distinguishing suffix)
            if len(exp) > 45:
                exp = "..." + exp[-42:]
            return f"{exp}/{part}"

    # No checkpoint found: use the last segment.
    last = parts[-1] if parts else dir_name
    return last[:60]


def get_primary_metric(task_name: str, task_result: dict):
    """Return (metric_display_name, value) for the primary metric of a task."""
    # Find matching rule
    for key_substr, metric_key in PRIMARY_METRIC_MAP.items():
        if key_substr in task_name:
            val = task_result.get(metric_key)
            if val is not None:
                return metric_key, val
    # Fallback: return first non-alias, non-stderr metric
    for k, v in task_result.items():
        if k == "alias" or "_stderr" in k:
            continue
        if isinstance(v, (int, float)):
            return k, v
    return None, None


def extract_results_from_json(json_path: str):
    """Parse a results JSON and return {task_name: {metric: value}} dict."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        return {}
    return data.get("results", {})


def merge_results(existing: dict, new: dict):
    """Keep the best (max) value for each metric when merging duplicate runs."""
    merged = dict(existing)
    for task, metrics in new.items():
        if task not in merged:
            merged[task] = dict(metrics)
        else:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    merged[task][k] = max(merged[task].get(k, v), v)
    return merged


def aggregate_math_subtasks(task_results: dict):
    """
    For math500: aggregate all minerva_math_* subtasks into a single
    'math500 (avg)' row using math_verify,none.
    """
    math_tasks = {k: v for k, v in task_results.items() if "minerva_math" in k}
    if not math_tasks:
        return task_results

    vals = []
    for task_vals in math_tasks.values():
        v = task_vals.get("math_verify,none")
        if v is not None:
            vals.append(v)

    aggregated = dict(task_results)
    # Remove individual subtasks
    for k in list(math_tasks.keys()):
        del aggregated[k]
    # Add aggregate
    if vals:
        aggregated["math500 (avg)"] = {
            "math_verify,none": sum(vals) / len(vals),
            "_subtask_count": len(vals),
        }
    return aggregated


# Main

def process_task_folder(task_folder_path: str, folder_name: str):
    """
    Process one task_len folder.
    Returns a list of (model_short_name, {task: (metric_name, value)}) tuples.
    """
    rows = []  # [(model_name, {task: (metric_name, value)})]

    checkpoint_dirs = sorted(
        d for d in os.listdir(task_folder_path)
        if os.path.isdir(os.path.join(task_folder_path, d))
    )

    for ckpt_dir in checkpoint_dirs:
        ckpt_path = os.path.join(task_folder_path, ckpt_dir)
        model_name = shorten_model_name(ckpt_dir)

        # Collect and merge all results files for this checkpoint
        merged = {}
        json_files = sorted(
            f for f in os.listdir(ckpt_path) if f.startswith("results_") and f.endswith(".json")
        )
        for jf in json_files:
            raw = extract_results_from_json(os.path.join(ckpt_path, jf))
            merged = merge_results(merged, raw)

        if not merged:
            continue

        # Aggregate math subtasks
        merged = aggregate_math_subtasks(merged)

        # Extract primary metric per task
        task_metrics = {}
        for task_name, task_result in merged.items():
            metric_name, value = get_primary_metric(task_name, task_result)
            if metric_name is not None:
                task_metrics[task_name] = (metric_name, value)

        if task_metrics:
            rows.append((model_name, task_metrics))

    return rows


def print_task_table(folder_name: str, rows):
    """Print a formatted table for one task_len folder."""
    if not rows:
        return

    # Collect all task names (preserve insertion order)
    all_tasks = []
    seen = set()
    for _, task_metrics in rows:
        for t in task_metrics:
            if t not in seen:
                all_tasks.append(t)
                seen.add(t)

    # Column widths, capped model name.
    MODEL_MAX = 55
    model_col_w = max(len("Model"), min(MODEL_MAX, max(len(m) for m, _ in rows))) + 2
    task_col_w = {t: max(len(t), 9) + 2 for t in all_tasks}

    total_w = model_col_w + sum(task_col_w.values())
    sep = "-" * total_w
    header = f"{'Model':<{model_col_w}}" + "".join(f"{t:^{task_col_w[t]}}" for t in all_tasks)

    print(f"\n{'=' * total_w}")
    print(f"  Task folder: {folder_name}")
    print(f"{'=' * total_w}")
    print(header)
    print(sep)

    # Track best per task
    best_val   = {t: -1.0 for t in all_tasks}
    best_model = {t: ""   for t in all_tasks}

    for model_name, task_metrics in rows:
        # Truncate long names from the left for display
        display_name = model_name if len(model_name) <= MODEL_MAX else "..." + model_name[-(MODEL_MAX - 3):]
        row_str = f"{display_name:<{model_col_w}}"
        for t in all_tasks:
            if t in task_metrics:
                _, val = task_metrics[t]
                pct = val * 100
                cell = f"{pct:6.2f}%"
                row_str += f"{cell:^{task_col_w[t]}}"
                if val > best_val[t]:
                    best_val[t] = val
                    best_model[t] = model_name
            else:
                row_str += f"{'-':^{task_col_w[t]}}"
        print(row_str)

    print(sep)

    # Best model summary
    print("  Best model per task:")
    for t in all_tasks:
        if best_model[t]:
            print(f"    [{t}]  {best_model[t]}  ->  {best_val[t]*100:.2f}%")
    print()


def main():
    if not os.path.isdir(BASE_DIR):
        print(f"ERROR: Directory not found: {BASE_DIR}")
        return

    task_folders = sorted(
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
    )

    for folder_name in task_folders:
        folder_path = os.path.join(BASE_DIR, folder_name)
        rows = process_task_folder(folder_path, folder_name)
        print_task_table(folder_name, rows)


if __name__ == "__main__":
    main()
