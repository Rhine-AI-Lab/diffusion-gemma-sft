"""Reward functions used by the GDSD training scripts."""

import math
import re
import subprocess
from functools import partial, update_wrapper
from typing import Callable

import numpy as np
from math_verify import parse, verify

from .utils.local_sandbox import local_execute
from .utils.math500_utils import boxed_in_answer, is_equiv, last_boxed_only_string, remove_boxed


def extract_code(completion: str, language: str = "python") -> str:
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    return matches[0] if matches else ""


def get_code_format_reward(language: str = "python"):
    import ast

    pattern = re.compile(
        rf"^"
        r"(?:(?!```)[\s\S])*?"
        rf"```{language}\n"
        r"(?:(?!```)[\s\S])*?"
        rf"```\n?$",
        re.DOTALL,
    )

    def code_format_reward(completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content in completion_contents:
            if not pattern.fullmatch(content):
                rewards.append(0.0)
                continue

            code_blocks = re.findall(rf"```{language}\n(.*?)```", content, re.DOTALL)
            if not code_blocks:
                rewards.append(0.0)
                continue

            code = code_blocks[0].strip()
            try:
                ast.parse(code)
                rewards.append(1.0)
            except SyntaxError:
                rewards.append(0.5)
            except Exception:
                rewards.append(0.0)

        return rewards

    return code_format_reward


def code_reward(
    completions,
    num_parallel: int = 2,
    provider_type: str = "local",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    """Evaluate Python code snippets against bundled/local tests.

    This anonymized supplement intentionally includes only the local executor. Use it
    only on trusted benchmark tests inside an isolated environment.
    """

    if provider_type != "local":
        raise ValueError("Only provider_type='local' is included in this anonymized supplement.")

    format_rewards = get_code_format_reward(language="python")(completions)
    pairs = []
    valid_indices = []
    for i, (reward, completion) in enumerate(zip(format_rewards, completions)):
        if reward < 1:
            continue
        pairs.append((extract_code(completion[-1]["content"]), kwargs["test_cases"][i]))
        valid_indices.append(i)

    results = local_execute(pairs)
    final_results = [0.0] * len(completions)
    for idx, result in zip(valid_indices, results):
        final_results[idx] = result
    return final_results


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
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]
    return [1.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def extract_solution(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    return matches[-1].strip() if matches else None


def validate_equation(equation_str, available_numbers):
    try:
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        return sorted(numbers_in_eq) == sorted(available_numbers)
    except Exception:
        return False


def evaluate_equation(equation_str):
    try:
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")
        return eval(equation_str, {"__builtins__": None}, {})
    except Exception:
        return None


def compute_score(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0):
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]
    equation = extract_solution(solution_str)

    if equation is None:
        return 0
    if not validate_equation(equation, numbers):
        return format_score

    result = evaluate_equation(equation)
    if result is None:
        return format_score
    if abs(result - target) < 1e-5:
        return score
    return format_score


def reward_func_countdown(prompts, completions, run_name=None, step=None, rank=None, **kwargs) -> list[float]:
    if isinstance(completions[0], list) and isinstance(completions[0][0], dict) and "content" in completions[0][0]:
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    scores = []
    for i, response in enumerate(responses):
        ground_truth = {"target": kwargs["target"][i], "numbers": kwargs["numbers"][i]}
        scores.append(compute_score(response, ground_truth))
    return scores


def extract_answer_sudoku(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    if matches:
        return "".join(char for char in matches[-1].strip() if char.isdigit())
    return None


def validate_sudoku_solution(solution_str, ground_truth, puzzle):
    if solution_str is None or len(solution_str) == 0:
        return 0.0

    if len(solution_str) < 16:
        solution_str = solution_str + "0" * (16 - len(solution_str))
    elif len(solution_str) > 16:
        solution_str = solution_str[:16]

    empty_indices = [i for i in range(16) if puzzle[i] == "0"]
    if not empty_indices:
        return 0.0

    correct_cells = sum(1 for i in empty_indices if solution_str[i] == ground_truth[i])
    return correct_cells / len(empty_indices)


def reward_func_sudoku(prompts, completions, run_name=None, step=None, rank=None, **kwargs) -> list[float]:
    if isinstance(completions[0], list) and isinstance(completions[0][0], dict) and "content" in completions[0][0]:
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    scores = []
    for i, response in enumerate(responses):
        puzzle = kwargs["puzzle"][i]
        ground_truth = kwargs["solution"][i]
        solution = extract_answer_sudoku(response)
        scores.append(0.0 if solution is None else validate_sudoku_solution(solution, ground_truth, puzzle))
    return scores


def correctness_reward_func_math(prompts, completions, answer, step=None, run_name=None, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    parsed_responses = [parse(r) for r in responses]
    parsed_answer = [parse(a) for a in answer]
    return [2.0 if verify(r, a) else 0.0 for r, a in zip(parsed_responses, parsed_answer)]


def correctness_reward_func_math_xml(prompts, completions, answer, step=None, run_name=None, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    clean_answers = [remove_boxed(last_boxed_only_string(a)) for a in answer]
    extracted_responses = []
    for response in responses:
        try:
            answer_text = response.split("<answer>", 1)[1].split("</answer>", 1)[0]
        except Exception:
            answer_text = response
        extracted_responses.append(remove_boxed(last_boxed_only_string(answer_text)))
    return [2.0 if is_equiv(r, a) else 0.0 for r, a in zip(extracted_responses, clean_answers)]


def boxed_and_answer_tags_format_reward_math(prompts, completions, answer, step=None, run_name=None, **kwargs) -> list[float]:
    raw_rewards = boxed_in_answer(prompts, completions, answer, step=step)
    return [r * 0.5 for r in raw_rewards]


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
    contents = [completion[0]["content"] for completion in completions]
    return [_count_xml(c) for c in contents]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted = [_extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted]


def correctness_reward_func(prompts, completions, answer, step=None, run_name=None, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [_extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def get_reward_funcs(script_args) -> list[Callable]:
    reward_funcs_registry = {
        "gsm8k": [correctness_reward_func_gsm8k],
        "gsm8k_xml": [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        "math": [correctness_reward_func_math],
        "math_xml": [
            correctness_reward_func_math_xml,
            boxed_and_answer_tags_format_reward_math,
        ],
        "countdown": [reward_func_countdown],
        "sudoku": [reward_func_sudoku],
        "code": [
            get_code_format_reward(language=script_args.code_language),
            update_wrapper(
                partial(
                    code_reward,
                    num_parallel=script_args.parallel_code_exec_per_proc,
                    provider_type=script_args.code_provider,
                    enforce_same_language=False,
                ),
                code_reward,
            ),
        ],
    }
    reward_funcs = reward_funcs_registry.get(script_args.dataset_name, [])
    if not reward_funcs:
        raise ValueError(f"No reward functions found for dataset {script_args.dataset_name}")
    return reward_funcs
