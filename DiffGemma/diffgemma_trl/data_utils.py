"""Dataset helpers for DiffusionGemma TRL training."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset

from .rewards import extract_hash_answer
from .sft_data_utils import (
    DIFFUGEMMA_SUDOKU_PROMPT,
    GSM8K_PROMPT,
    COUNTDOWN_PROMPT,
    countdown_gdsd_user_content,
    gsm8k_gdsd_user_content,
)


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
Please solve the following Sudoku puzzle. The puzzle is provided as a string
reading left-to-right, top-to-bottom, where '0' or '.' represents empty cells.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
solution digits with no spaces or separators
</answer>
"""

GSM8K_XML_SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

THINK_SYSTEM_PROMPT = "<|think|>"


def _diffgemma_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json_dataset(dataset_path: str | os.PathLike, split: str = "train") -> Dataset:
    if dataset_path is None:
        raise ValueError("Pass --dataset_path for dataset_name=json/jsonl.")
    dataset = load_dataset("json", data_files=str(dataset_path))
    return dataset[split] if split in dataset else dataset["train"]


def get_gsm8k_questions_xml(split: str = "train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]

    def format_example(x):
        return {
            "prompt": [{"role": "user", "content": GSM8K_XML_SYSTEM_PROMPT + "\n\n" + x["question"]}],
            "answer": extract_hash_answer(x["answer"]),
        }

    return data.map(format_example)


def get_gsm8k_questions(split: str = "train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]

    def format_example(x):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        f"{x['question']}\nPlease reason step by step, and put the final answer after ####."
                    ),
                }
            ],
            "answer": extract_hash_answer(x["answer"]),
        }

    return data.map(format_example)


def get_gsm8k_questions_local(split: str = "train", data_dir: str | None = None) -> Dataset:
    """Load GSM8K from local parquet files (for environments without HuggingFace access).

    Expected directory structure: <data_dir>/main/<split>-*.parquet
    Default data_dir on Nebula: /data/oss_bucket_0/dllm_data/gsm8k
    """
    if data_dir is None:
        data_dir = "/data/oss_bucket_0/dllm_data/gsm8k"
    pattern = os.path.join(data_dir, "main", f"{split}-*.parquet")
    data = load_dataset("parquet", data_files={split: pattern})[split]

    def format_example(x):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        f"{x['question']}\nPlease reason step by step, and put the final answer after ####."
                    ),
                }
            ],
            "answer": extract_hash_answer(x["answer"]),
        }

    return data.map(format_example)


def get_gsm8k_questions_sft(dataset_path=None) -> Dataset:
    """Continue-path GSM8K: load the SFT prompt/completion jsonl and rebuild the EXACT
    SFT prompt (GSM8K_PROMPT, literal string -- not a chat template) so the continued
    RL policy sees in-distribution prompts, byte-identical to SFT. The reward
    (correctness_reward_func) needs the ground-truth final answer, which we read from
    the completion's <answer>...</answer> tag.
    """
    if dataset_path is None:
        raise ValueError("Pass --dataset_path for dataset_name=gsm8k_sft.")
    path = Path(os.path.expanduser(str(dataset_path)))
    data = load_dataset("json", data_files=str(path))["train"]

    def _xml_answer(completion: str) -> str:
        return completion.split("<answer>")[-1].split("</answer>")[0].strip()

    return data.map(
        lambda x: {
            "prompt": GSM8K_PROMPT.format(text=x["prompt"].strip()),
            "answer": _xml_answer(x["completion"]),
        }
    )


def get_countdown_questions_sft(dataset_path=None) -> Dataset:
    """Countdown for RL on DiffusionGemma (d1/wd1 style): load a local jsonl of
    {"nums": [...], "target": int} and build the literal COUNTDOWN_PROMPT (not a chat
    template), byte-identical between train and eval. reward_func_countdown reads the
    ground truth from the "target" and "numbers" columns.
    """
    if dataset_path is None:
        raise ValueError("Pass --dataset_path for dataset_name=countdown_sft.")
    path = Path(os.path.expanduser(str(dataset_path)))
    data = load_dataset("json", data_files=str(path))["train"]

    return data.map(
        lambda x: {
            "prompt": COUNTDOWN_PROMPT.format(
                text=(
                    f"Numbers: {list(x['nums'])}\\n"
                    f"Target: {int(x['target'])}"
                ),
                numbers=list(x["nums"]),
                target=int(x["target"]),
            ),
            "target": int(x["target"]),
            "numbers": list(x["nums"]),
        }
    )


def get_gsm8k_questions_gdsd(dataset_path=None) -> Dataset:
    """GSM8K in the GDSD/native-chat-template setup: a CONVERSATIONAL prompt so the
    trainer applies the model's real chat template (native reasoning channel). Loads
    the SFT prompt/completion jsonl; the reward reads the gold answer from the
    completion's <answer> tag. Matches gsm8k_eval gdsd-style for train==eval."""
    if dataset_path is None:
        raise ValueError("Pass --dataset_path for dataset_name=gsm8k_gdsd.")
    path = Path(os.path.expanduser(str(dataset_path)))
    data = load_dataset("json", data_files=str(path))["train"]

    def _xml_answer(completion: str) -> str:
        return completion.split("<answer>")[-1].split("</answer>")[0].strip()

    return data.map(
        lambda x: {
            "prompt": [{"role": "user", "content": gsm8k_gdsd_user_content(x["prompt"])}],
            "answer": _xml_answer(x["completion"]),
        },
        remove_columns=data.column_names,  # drop the raw 'completion' str (TRL would do prompt[list]+completion[str])
    )


def get_countdown_questions_gdsd(dataset_path=None) -> Dataset:
    """Countdown in the GDSD/d1 setup: a CONVERSATIONAL prompt so the trainer applies
    the model's REAL chat template (DiffusionGemma's native reasoning channel), with
    d1's exact instruction and bare <answer>. Matches countdown_eval --answer_format gdsd
    so train and eval use the identical prompt.
    """
    if dataset_path is None:
        raise ValueError("Pass --dataset_path for dataset_name=countdown_gdsd.")
    path = Path(os.path.expanduser(str(dataset_path)))
    data = load_dataset("json", data_files=str(path))["train"]
    return data.map(
        lambda x: {
            "prompt": [{"role": "user", "content": countdown_gdsd_user_content(list(x["nums"]), int(x["target"]))}],
            "target": int(x["target"]),
            "numbers": list(x["nums"]),
        }
    )


def get_countdown_questions(split: str = "train") -> Dataset:
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
                        "Put only the final expression inside <answer></answer> tags."
                    ),
                }
            ],
            "target": x["target"],
            "numbers": x["nums"],
        }
    )


def get_countdown_questions_templated(dataset_path=None, prompt_style: str = "countdown") -> Dataset:
    """Countdown RL prompts formatted with a selectable literal <|turn> template.

    The ``prompt`` is built as a plain STRING (not chat messages) via the chosen
    ``PROMPT_TEMPLATES`` entry, so TRL's ``maybe_apply_chat_template`` passes it through
    unchanged -> the training prompt matches the eval prompt (countdown_eval) byte-for-byte.
    Emits ``target``/``numbers`` columns for ``reward_func_countdown``.

    ``dataset_path``: local jsonl with ``input`` ("a,b,c") / ``output`` (target). When None,
    falls back to the HF Countdown set (3-number subset) -- needs internet.
    """
    from .sft_data_utils import PROMPT_TEMPLATES

    template = PROMPT_TEMPLATES.get(prompt_style)
    if template is None:
        raise ValueError(
            f"Unknown countdown_prompt_style={prompt_style!r}; expected one of {sorted(PROMPT_TEMPLATES)}."
        )

    def _to_row(nums, target):
        nums = [int(n) for n in nums]
        target = int(target)
        text = f"Numbers: {nums}\nTarget: {target}"
        return {
            "prompt": template.format(text=text, numbers=nums, target=target),
            "target": target,
            "numbers": nums,
        }

    if dataset_path is not None:
        data = _load_json_dataset(dataset_path)
        return data.map(
            lambda x: _to_row(str(x["input"]).split(","), x["output"]),
            remove_columns=data.column_names,
        )

    data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train").filter(lambda x: len(x["nums"]) == 3)
    return data.map(lambda x: _to_row(x["nums"], x["target"]), remove_columns=data.column_names)


def get_math_questions(split: str = "train") -> Dataset:
    data = load_dataset("ankner/math-500", split=split)
    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{x['problem']}\nPlease reason step by step, and put your final answer in a box.",
                }
            ],
            "answer": x["solution"],
        }
    )


def get_dapo_math_17k_questions(split: str = "train") -> Dataset:
    """Load BytedTsinghua-SIA/DAPO-Math-17k with official DiffusionGemma thinking.

    The HF dataset stores prompts as a one-message chat list and gold answers under
    reward_model.ground_truth. We prepend the official <|think|> system turn so the
    tokenizer's native chat template enables the thought channel.
    """
    data = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split=split)

    def _prompt_content(raw_prompt) -> str:
        if isinstance(raw_prompt, list):
            parts = []
            for message in raw_prompt:
                if isinstance(message, dict) and message.get("content") is not None:
                    parts.append(str(message["content"]))
                else:
                    parts.append(str(message))
            return "\n\n".join(parts).strip()
        return str(raw_prompt).strip()

    def _answer(reward_model) -> str:
        if isinstance(reward_model, dict):
            return str(reward_model.get("ground_truth", "")).strip()
        return ""

    return data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": THINK_SYSTEM_PROMPT},
                {"role": "user", "content": _prompt_content(x["prompt"])},
            ],
            "answer": _answer(x.get("reward_model", {})),
        },
        remove_columns=data.column_names,
    )


def get_sudoku_questions(dataset_path=None) -> Dataset:
    if dataset_path is None:
        dataset_path = _diffgemma_root() / "dataset" / "4x4_test_sudoku.csv"
    else:
        dataset_path = Path(os.path.expanduser(str(dataset_path)))

    suffix = dataset_path.suffix.lower()

    # SFT training set path: the same {"prompt": "<space-sep puzzle>", "completion":
    # "<space-sep solution>"} jsonl base-sft was trained on. Reusing it guarantees the
    # exact puzzles/format SFT saw -- no CSV format conversion. The reward needs the
    # CONTIGUOUS 81-char forms (positional cell indexing), so strip non-digits.
    if suffix in (".jsonl", ".json"):
        data = load_dataset("json", data_files=str(dataset_path))["train"]
        return data.map(
            lambda x: {
                "prompt": DIFFUGEMMA_SUDOKU_PROMPT.format(text=x["prompt"].strip()),
                "puzzle": "".join(c for c in x["prompt"] if c.isdigit()),
                "solution": "".join(c for c in x["completion"] if c.isdigit()),
            }
        )

    # CSV path (kaggle Puzzle/Solution, contiguous 81-char strings).
    # NOTE: avoid is_dir() because OSS FUSE mounts may not report it correctly.
    if suffix != ".csv":
        dataset_path = dataset_path / "4x4_test_sudoku.csv"
    df = pd.read_csv(dataset_path, dtype={"Puzzle": str, "Solution": str})
    data = Dataset.from_pandas(df)
    return data.map(
        lambda x: {
            # Match the SFT sudoku prompt byte-for-byte (literal DIFFUGEMMA template, not a
            # chat template). SFT used SPACE-SEPARATED puzzle digits ("0 7 0 ..."); this CSV
            # stores a contiguous 81-char string, so space-join it for the prompt. The
            # reward keeps the contiguous forms (positional cell indexing).
            "prompt": DIFFUGEMMA_SUDOKU_PROMPT.format(text=" ".join(x["Puzzle"])),
            "puzzle": x["Puzzle"],
            "solution": x["Solution"],
        }
    )


def get_datasets(name: str, dataset_path=None, countdown_prompt_style: str = "countdown") -> Dataset:
    if name in {"json", "jsonl"}:
        return _load_json_dataset(dataset_path)
    if name == "gsm8k":
        return get_gsm8k_questions()
    if name == "gsm8k_sft":
        return get_gsm8k_questions_sft(dataset_path=dataset_path)
    if name == "gsm8k_gdsd":
        return get_gsm8k_questions_gdsd(dataset_path=dataset_path)
    if name == "gsm8k_local":
        return get_gsm8k_questions_local(data_dir=dataset_path)
    if name == "gsm8k_xml":
        return get_gsm8k_questions_xml()
    if name in {"math", "math_xml"}:
        return get_math_questions()
    if name == "dapo_math_17k":
        return get_dapo_math_17k_questions()
    if name == "countdown":
        return get_countdown_questions_templated(
            dataset_path=dataset_path,
            prompt_style=countdown_prompt_style,
        )
    if name == "countdown_sft":
        return get_countdown_questions_sft(dataset_path=dataset_path)
    if name == "countdown_gdsd":
        return get_countdown_questions_gdsd(dataset_path=dataset_path)
    if name == "sudoku":
        return get_sudoku_questions(dataset_path=dataset_path)
    raise ValueError(f"Dataset {name} not supported.")
