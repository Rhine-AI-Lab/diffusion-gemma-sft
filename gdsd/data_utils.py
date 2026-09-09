import os
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset

from gdsd.rewards import extract_hash_answer


REASONING_SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

SUDOKU_SYSTEM_PROMPT = """
Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character
string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Rules:
- Fill empty cells with digits 1-4.
- Each row must contain digits 1-4 exactly once.
- Each column must contain digits 1-4 exactly once.
- Each 2x2 box must contain digits 1-4 exactly once.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>
"""

CODE_SYSTEM_PROMPT = (
    "You are a helpful assistant. When you output code, it must be in a single fenced "
    "code block using triple backticks and the language specifier. The code block must "
    "be the last part of your response."
)

GSM8K_XML_SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

MATH_XML_SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def make_code_conversation(example):
    prompt = [
        {"role": "system", "content": CODE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": example["question"]
            + ("\nTest cases: " + example["test_cases"][0] if "\n```\n" not in example["question"] else ""),
        },
    ]
    return {"prompt": prompt}


def get_code_questions(split="train", dataset_path=None) -> Dataset:
    if dataset_path is None:
        raise ValueError(
            "The code dataset is not bundled. Pass --dataset_path with a local JSONL file "
            "containing question/test_cases fields."
        )

    dataset = load_dataset("json", data_files=dataset_path)
    dataset = dataset.map(make_code_conversation)
    for split_name in dataset:
        if "messages" in dataset[split_name].column_names:
            dataset[split_name] = dataset[split_name].remove_columns("messages")
    return dataset[split]


def get_gsm8k_questions_xml(split="train"):
    data = load_dataset("openai/gsm8k", "main")[split]

    def format_example(x):
        return {
            "prompt": [{"role": "user", "content": GSM8K_XML_SYSTEM_PROMPT + "\n\n" + x["question"]}],
            "answer": extract_hash_answer(x["answer"]),
        }

    return data.map(format_example)


def get_gsm8k_questions(split="train"):
    data = load_dataset("openai/gsm8k", "main")[split]

    def format_example(x):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{x['question']}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
                }
            ],
            "answer": extract_hash_answer(x["answer"]),
        }

    return data.map(format_example)


def get_math_questions(split="train") -> Dataset:
    data = load_dataset("ankner/math-500", split=split)
    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{x['problem']}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
                }
            ],
            "answer": x["solution"],
        }
    )


def get_math_questions_xml(split="train") -> Dataset:
    data = load_dataset("ankner/math-500", split=split)
    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        f"{MATH_XML_SYSTEM_PROMPT}\n\nYou are a math expert. Solve the problem step by step. "
                        f"Wrap the final answer in a \\boxed{{}}.\n\n{x['problem']}"
                    ),
                }
            ],
            "answer": x["solution"],
        }
    )


def get_countdown_questions(split="train") -> Dataset:
    data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split=split)
    data = data.filter(lambda x: len(x["nums"]) == 3)

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        f"{REASONING_SYSTEM_PROMPT}\nUsing only the numbers {x['nums']}, create an arithmetic "
                        f"expression that evaluates to exactly {x['target']}. Use every number exactly once. "
                        "Use +, -, *, and / as needed. Put only the final expression inside <answer></answer> tags."
                    ),
                }
            ],
            "target": x["target"],
            "numbers": x["nums"],
        }
    )


def get_sudoku_questions(dataset_path=None) -> Dataset:
    if dataset_path is None:
        dataset_path = _repo_root() / "dataset" / "4x4_sudoku_unique_puzzles.csv"
    else:
        dataset_path = Path(os.path.expanduser(dataset_path))

    df = pd.read_csv(dataset_path, dtype={"Puzzle": str, "Solution": str})
    data = Dataset.from_pandas(df)

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SUDOKU_SYSTEM_PROMPT}\n\nNow solve this Sudoku puzzle: {x['Puzzle']}\n",
                }
            ],
            "puzzle": x["Puzzle"],
            "solution": x["Solution"],
        }
    )


def get_datasets(name: str, dataset_path=None) -> Dataset:
    if name == "code":
        return get_code_questions(dataset_path=dataset_path)
    if name == "gsm8k":
        return get_gsm8k_questions()
    if name == "gsm8k_xml":
        return get_gsm8k_questions_xml()
    if name == "math":
        return get_math_questions()
    if name == "math_xml":
        return get_math_questions_xml()
    if name == "countdown":
        return get_countdown_questions()
    if name == "sudoku":
        return get_sudoku_questions(dataset_path=dataset_path)
    raise ValueError(f"Dataset {name} not supported.")
