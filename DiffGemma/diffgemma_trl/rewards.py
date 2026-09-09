"""Reward functions for DiffusionGemma TRL experiments.

This file is intentionally self-contained so the DiffGemma training path does
not import the root-level ``gdsd`` package.
"""

from __future__ import annotations

import ast
import math
import re
from functools import partial, update_wrapper
from typing import Callable


def _completion_texts(completions) -> list[str]:
    if completions and isinstance(completions[0], list) and completions[0] and isinstance(completions[0][0], dict):
        return [completion[0]["content"] for completion in completions]
    return list(completions)


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####", 1)[1].strip()


def extract_answer(text: str) -> str:
    match_cot = re.search(r"The answer is (\-?[0-9\.,]+).", text)
    if match_cot:
        return match_cot.group(1).strip()

    match_hash = re.search(r"#### (\-?[0-9\.,]+)", text)
    if match_hash:
        return match_hash.group(1).strip()

    fallback_match = re.findall(r"(\-?[0-9]+)", text)
    if fallback_match:
        return fallback_match[-1].strip()

    return ""


def correctness_reward_func_gsm8k(prompts, completions, answer, step=None, run_name=None, **kwargs) -> list[float]:
    responses = _completion_texts(completions)
    extracted_responses = [extract_answer(r) for r in responses]
    return [1.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def _strip_boxed(s: str | None) -> str | None:
    r"""Return the contents of the last \boxed{...}/\fbox{...} if present, else s.

    The countdown_d2_eval prompt asks for the final expression inside \boxed{}, but
    validate/evaluate_equation only accept a bare arithmetic expression. Stripping here
    keeps all three countdown prompts gradeable by the same reward/scorer.
    """
    if not s:
        return s
    idx = s.rfind("\\boxed")
    if idx < 0:
        idx = s.rfind("\\fbox")
    if idx < 0:
        return s
    open_i = s.find("{", idx)
    if open_i < 0:
        return s
    depth = 0
    for j in range(open_i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[open_i + 1 : j]
    return s


def extract_solution(solution_str: str):
    matches = re.findall(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if not matches:
        return None
    return _strip_boxed(matches[-1].strip()).strip()


def validate_equation(equation_str: str, available_numbers) -> bool:
    try:
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        return sorted(numbers_in_eq) == sorted(available_numbers)
    except Exception:
        return False


def evaluate_equation(equation_str: str):
    try:
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")
        return eval(equation_str, {"__builtins__": None}, {})
    except Exception:
        return None


def reward_func_countdown(prompts, completions, run_name=None, step=None, rank=None, **kwargs) -> list[float]:
    responses = _completion_texts(completions)
    scores = []
    for i, response in enumerate(responses):
        target = kwargs["target"][i]
        numbers = kwargs["numbers"][i]
        equation = extract_solution(response)
        if equation is None:
            scores.append(0.0)
        elif not validate_equation(equation, numbers):
            scores.append(0.1)
        else:
            result = evaluate_equation(equation)
            scores.append(1.0 if result is not None and abs(result - target) < 1e-5 else 0.1)
    return scores


def reward_func_countdown_binary(prompts, completions, run_name=None, step=None, rank=None, **kwargs) -> list[float]:
    """Binary correctness reward for countdown (1.0 iff correct, else 0.0).

    Removes the 0.1 partial reward for format-valid-but-wrong answers, giving a
    cleaner signal when the model already masters format and needs only accuracy.
    """
    responses = _completion_texts(completions)
    scores = []
    for i, response in enumerate(responses):
        target = kwargs["target"][i]
        numbers = kwargs["numbers"][i]
        equation = extract_solution(response)
        if equation is None or not validate_equation(equation, numbers):
            scores.append(0.0)
        else:
            result = evaluate_equation(equation)
            scores.append(1.0 if result is not None and abs(result - target) < 1e-5 else 0.0)
    return scores


def extract_answer_sudoku(solution_str: str):
    matches = re.findall(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if matches:
        return "".join(char for char in matches[-1].strip() if char.isdigit())
    digits = "".join(char for char in solution_str if char.isdigit())
    return digits or None


def validate_sudoku_solution(solution_str: str | None, ground_truth: str, puzzle: str) -> float:
    if not solution_str:
        return 0.0
    size = len(ground_truth)
    if len(solution_str) < size:
        solution_str = solution_str + "0" * (size - len(solution_str))
    elif len(solution_str) > size:
        solution_str = solution_str[:size]

    empty_indices = [i for i, value in enumerate(puzzle) if value in {"0", "."}]
    if not empty_indices:
        return 0.0

    correct_cells = sum(1 for i in empty_indices if solution_str[i] == ground_truth[i])
    return correct_cells / len(empty_indices)


def reward_func_sudoku(prompts, completions, run_name=None, step=None, rank=None, **kwargs) -> list[float]:
    responses = _completion_texts(completions)
    scores = []
    for i, response in enumerate(responses):
        puzzle = kwargs["puzzle"][i]
        ground_truth = kwargs["solution"][i]
        scores.append(validate_sudoku_solution(extract_answer_sudoku(response), ground_truth, puzzle))
    return scores


def reward_func_sudoku_binary(prompts, completions, run_name=None, step=None, rank=None, **kwargs) -> list[float]:
    """Binary full-solve reward (1.0 iff every empty cell is correct, else 0.0).

    The dense cell-fraction reward saturates (the SFT model gets most cells right on
    every sample), giving near-zero within-group reward variance and thus no GRPO/wd1
    advantage signal. This bimodal reward restores within-group spread on medium/hard
    puzzles, where full-solve rate sits well away from 0/100%.
    """
    responses = _completion_texts(completions)
    scores = []
    for i, response in enumerate(responses):
        frac = validate_sudoku_solution(extract_answer_sudoku(response), kwargs["solution"][i], kwargs["puzzle"][i])
        scores.append(1.0 if frac >= 1.0 else 0.0)
    return scores


def _extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def _count_xml(text: str) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n<answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    return [_count_xml(c) for c in _completion_texts(completions)]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    return [0.5 if re.match(pattern, r, re.DOTALL) else 0.0 for r in _completion_texts(completions)]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    return [0.5 if re.match(pattern, r, re.DOTALL) else 0.0 for r in _completion_texts(completions)]


def int_reward_func(completions, **kwargs) -> list[float]:
    extracted = [_extract_xml_answer(r) for r in _completion_texts(completions)]
    return [0.5 if r.isdigit() else 0.0 for r in extracted]


def correctness_reward_func(prompts, completions, answer, step=None, run_name=None, **kwargs) -> list[float]:
    extracted_responses = [_extract_xml_answer(r) for r in _completion_texts(completions)]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def correctness_reward_func_math(prompts, completions, answer, step=None, run_name=None, **kwargs) -> list[float]:
    try:
        from math_verify import parse, verify
    except ImportError as exc:
        raise ImportError("Install math_verify to use the math reward.") from exc

    responses = _completion_texts(completions)
    parsed_responses = [parse(r) for r in responses]
    parsed_answer = [parse(a) for a in answer]
    return [2.0 if verify(r, a) else 0.0 for r, a in zip(parsed_responses, parsed_answer)]


def get_code_format_reward(language: str = "python"):
    pattern = re.compile(
        rf"^"
        r"(?:(?!```)[\s\S])*?"
        rf"```{language}\n"
        r"(?:(?!```)[\s\S])*?"
        rf"```\n?$",
        re.DOTALL,
    )

    def code_format_reward(completions, **kwargs):
        rewards = []
        for content in _completion_texts(completions):
            if not pattern.fullmatch(content):
                rewards.append(0.0)
                continue
            code_blocks = re.findall(rf"```{language}\n(.*?)```", content, re.DOTALL)
            if not code_blocks:
                rewards.append(0.0)
                continue
            try:
                ast.parse(code_blocks[0].strip())
                rewards.append(1.0)
            except SyntaxError:
                rewards.append(0.5)
            except Exception:
                rewards.append(0.0)
        return rewards

    return code_format_reward


def code_reward(completions, **kwargs) -> list[float]:
    raise NotImplementedError(
        "The DiffGemma-local port does not bundle a code sandbox yet. "
        "Use format-only code rewards or add a sandbox under DiffGemma/."
    )


def get_reward_funcs(script_args) -> list[Callable]:
    registry = {
        "gsm8k": [correctness_reward_func_gsm8k],
        "gsm8k_local": [correctness_reward_func_gsm8k],
        "gsm8k_xml": [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        "gsm8k_sft": [correctness_reward_func],
        "gsm8k_gdsd": [correctness_reward_func],
        "math": [correctness_reward_func_math],
        "math_xml": [correctness_reward_func_math],
        "dapo_math_17k": [correctness_reward_func_math],
        "countdown": [reward_func_countdown],
        "countdown_sft": [reward_func_countdown],
        "countdown_gdsd": [reward_func_countdown],
        "countdown_binary": [reward_func_countdown_binary],
        "sudoku": [reward_func_sudoku],
        "sudoku_binary": [reward_func_sudoku_binary],
        "json": [correctness_reward_func],
        "jsonl": [correctness_reward_func],
        "code": [
            get_code_format_reward(language=script_args.code_language),
            update_wrapper(partial(code_reward), code_reward),
        ],
    }
    # An explicit --reward_funcs name that matches a registry key overrides the
    # dataset_name default (e.g. --reward_funcs sudoku_binary with --dataset_name
    # sudoku, so dataset loading stays on "sudoku" while the reward switches).
    explicit = [name for name in getattr(script_args, "reward_funcs", []) or [] if name in registry]
    if explicit:
        reward_funcs = [f for name in explicit for f in registry[name]]
    else:
        reward_funcs = registry.get(script_args.dataset_name, [])
    if not reward_funcs:
        raise ValueError(f"No reward functions found for dataset {script_args.dataset_name}")
    return reward_funcs
