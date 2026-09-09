"""Dataset loading + collation for DiffusionGemma SFT.

Produces, per example:
  prompt_ids   [P]  left context (prompt), padded/truncated to max_prompt_length
  prompt_mask  [P]
  canvas       [L]  clean completion tokens (+EOS), padded to canvas_length
  canvas_mask  [L]  1.0 on real completion tokens (incl. EOS), 0.0 on padding

The completion is the model's target (scored by the denoising loss); the prompt
conditions the encoder. Matches the JAX hackable_diffusion CanvasChunker layout
(single canvas, EOS-terminated, pad tail ignored).

With ``block_diffusion`` set, the completion is instead split into canvas_length blocks
and ONE sampled block is the canvas, with the clean prior blocks appended to prompt_ids
(the encoder context) and marked in an extra ``prompt_comp_mask`` -- see ``_call_block``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import datasets
import torch


def get_sft_dataset(script_args) -> datasets.Dataset:
    """Load an SFT dataset with prompt/completion columns.

    Accepts: a ``.jsonl``/``.json`` file, a ``save_to_disk`` directory, or an HF
    dataset id.
    """
    import os

    path = str(script_args.dataset_path or script_args.dataset_name)
    if path.endswith((".jsonl", ".json")):
        ds = datasets.load_dataset("json", data_files=path)
    elif os.path.isdir(path) and os.path.exists(os.path.join(path, "dataset_info.json")):
        ds = datasets.load_from_disk(path)
    else:
        ds = datasets.load_dataset(path)
    if isinstance(ds, datasets.DatasetDict):
        split = getattr(script_args, "dataset_train_split", None)
        ds = ds[split] if split in ds else ds[list(ds)[0]]
    return ds


# Exact diffuGemma JAX sudoku prompt (gemma/diffusion/.../data/sudoku/sudoku_data.py
# _DEFAULT_SUDOKU_PROMPT). Tokenized as a literal string with add_bos -- NOT via a
# chat template -- to match the JAX SFT byte-for-byte. {text} = the puzzle.
DIFFUGEMMA_SUDOKU_PROMPT = (
    "<|turn>system Solve the following Sudoku puzzle. Empty cells are"
    " represented by 0. Output ONLY the solved puzzle immediately as"
    " a 9x9 grid of numbers separated by spaces. Do not include ####,"
    " explanations, or any other text.<turn|>\n<|turn>user"
    " {text}<turn|>\n<|turn>model\n"
)

# GSM8K: matches this repo's RL convention (gdsd/data_utils.GSM8K_XML_SYSTEM_PROMPT)
# so the SFT output format == the RL prompt/format (the format rewards require the
# <reasoning>/<answer> tags). The instruction goes in the user turn, as the RL
# loader does (user content = system prompt + "\n\n" + question), adapted to the
# DiffusionGemma <|turn> syntax. Completions must be reformatted to match (see
# dataset/make_gsm8k_jsonl.py).
GSM8K_PROMPT = (
    "<|turn>user Respond in the following format:\n"
    "<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n\n"
    "{text}<turn|>\n<|turn>model\n"
)

# Single-shot bug-fix (codefix_eval/build_bugfix_data.py): {text} is already a
# complete, self-contained instruction (task + buggy code + what to produce),
# so this is a bare user-turn wrapper -- no extra system framing needed.
BUGFIX_PROMPT = "<|turn>user {text}<turn|>\n<|turn>model\n"

# GDSD/d1 countdown user message, VERBATIM from gdsdv2/eval/data_utils.py CTDDataset
# (REASONING_SYSTEM_PROMPT + the exact d1 instruction, bare <answer>). Returned as a
# CONVERSATIONAL message so the trainer/eval apply the model's REAL chat template
# (DiffusionGemma's native reasoning channel) -- i.e. the GDSD apply_chat_template path,
# not our hand-written <|turn> wrapper.
_COUNTDOWN_GDSD_REASONING = (
    "\nRespond in the following format:\n<reasoning>\n...\n</reasoning>\n"
    "<answer>\n...\n</answer>\n"
)


def gsm8k_gdsd_user_content(question) -> str:
    """GSM8K user message for the native-chat-template (GDSD) path: the same
    <reasoning>/<answer> format instruction as GSM8K_PROMPT, but returned as a
    CONVERSATIONAL message so the trainer/eval apply the model's real chat template."""
    return (
        "Respond in the following format:\n<reasoning>\n...\n</reasoning>\n"
        "<answer>\n...\n</answer>\n\n" + str(question).strip()
    )


def countdown_gdsd_user_content(numbers, target) -> str:
    return (
        f"{_COUNTDOWN_GDSD_REASONING}\nUsing only the numbers {numbers}, create an "
        f"arithmetic expression that evaluates to exactly {target}. You must use all "
        "numbers from the list, and each number must be used exactly once. You may use "
        "the operations +, -, *, and / as needed. After reasoning, provide only your "
        "final expression inside <answer></answer> tags without including an equals "
        "sign or the target number. For example, if the numbers are [2, 3, 4] and the "
        "target is 5, a valid answer is: <answer>\n2*4-3\n</answer>"
    )


# MATH (MATH-500): step-by-step reasoning with a \boxed{} final answer. Scored with
# math_verify on the model output vs the gold answer (boxed fallback), so the prompt
# only needs to elicit a boxed final answer. Mirrors math_eval/math500.py instruction.
#   NOTE: literal braces are doubled ({{}}) because this template is consumed by
#   str.format(text=...) (run_eval and the collator's _encode_prompt).
MATH_PROMPT = (
    "<|turn>user You are a math expert. Solve the problem step by step. "
    "Wrap the final answer in a \\boxed{{}}.\n"
    "Respond in the following format:\n"
    "<reasoning>\n...\n</reasoning>\n<answer>\n\\boxed{{...}}\n</answer>\n\n"
    "{text}<turn|>\n<|turn>model\n"
)

# CHAT-ROUTE instruction (single braces -- this is chat *message content* passed to
# apply_chat_template, NOT a str.format template). Placed in the user turn, then a
# blank line, then the question. Same \boxed{}/<reasoning>/<answer> structure as
# MATH_PROMPT. This is the exact instruction the math-benchmark eval uses on its
# (default) chat route, so training via the chat route matches eval byte-for-byte.
MATH_INSTRUCTION = (
    "You are a math expert. You will be given a question to solve. Solve it step by "
    "step. Wrap the final answer in a \\boxed{}.\n"
    "Respond in the following format:\n"
    "<reasoning>\nYour reasoning here\n</reasoning>\n<answer>\n\\boxed{...}\n</answer>"
)

# String the chat/eval route appends after the model-turn header to prime the CoT.
# When it is prefilled into the PROMPT, the completion must not repeat it (the
# collator strips a leading occurrence -- see _encode_completion).
REASONING_PREFILL = "<reasoning>\n"

# Countdown: produce a single arithmetic expression using each number exactly once.
# The reward (rewards.extract_solution -> validate_equation/evaluate_equation) reads the
# BARE expression inside <answer></answer> (no '=', no \boxed, digits/operators only).
COUNTDOWN_PROMPT = (
    "<|turn>user Using only the provided numbers, create an arithmetic expression "
    "that evaluates to exactly the target. You may use +, -, *, and / as needed, but "
    "each number must be used exactly once. Think step by step. Put ONLY the final "
    "expression (no '=' and no target) inside <answer></answer> tags, e.g. "
    "<answer>a + b * c</answer>.\n"
    "Respond in the following format:\n"
    "<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n\n"
    "{text}<turn|>\n<|turn>model\n"
)

# Countdown (d2 train variant): the d2 project's TRAIN-time countdown prompt. Uses
# {numbers}/{target} placeholders (not {text}); the countdown loader formats with both
# styles. Bare expression inside <answer></answer>.
COUNTDOWN_D2_TRAIN_PROMPT = (
    "<|turn>user Respond in the following format:\n"
    "<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n\n"
    "Using only the numbers {numbers}, create an arithmetic expression that evaluates to "
    "exactly {target}. You must use all numbers from the list, and each number must be used "
    "exactly once. You may use +, -, *, and /. Put only the final expression (no '=' sign and "
    "no target number) inside <answer></answer> tags. For example, for numbers [2, 3, 4] and "
    "target 5 a valid answer is <answer>\n2*4-3\n</answer>.<turn|>\n<|turn>model\n"
)

# Countdown (d2 eval variant): the d2 project's CTD_SYSTEM_PROMPT, wrapped in <|turn>
# formatting. Asks for the final expression inside \boxed{} (the countdown reward/scorer
# strips \boxed via rewards._strip_boxed). Literal braces are doubled for str.format.
COUNTDOWN_D2_EVAL_PROMPT = (
    "<|turn>user Using only the provided numbers, create an arithmetic expression that "
    "evaluates to exactly the provided target number. You may use the operations +, -, *, and / "
    "as needed, but each number must be used exactly once. Think step-by-step. After reasoning, "
    "provide only your final expression inside \\boxed{{}} tags without including an equals sign "
    "or the target number. For example: \\boxed{{a + b * c}}"
    "Respond in the following format:\n"
    "<reasoning>\nYour reasoning here\n</reasoning>\n<answer>\n\\boxed{{...}}\n</answer>\n\n"
    "{text}<turn|>\n<|turn>model\n"
)

# MBPP: generate a self-contained Python solution. The code is extracted from the last
# fenced ```python block and run against the task's assert-style test_list.
MBPP_PROMPT = (
    "<|turn>user You are an expert Python programmer. Write a correct, self-contained "
    "Python solution for the task below. Output the solution as a single fenced code "
    "block using triple backticks with the `python` language tag, and nothing else.\n\n"
    "{text}<turn|>\n<|turn>model\n"
)

# HumanEval: complete the given function. The model is shown the function signature +
# docstring and asked to return the full implementation in a fenced python block.
HUMANEVAL_PROMPT = (
    "<|turn>user Complete the following Python function. Output the COMPLETE function "
    "(signature included) as a single fenced code block using triple backticks with the "
    "`python` language tag, and nothing else.\n\n"
    "```python\n{text}```<turn|>\n<|turn>model\n"
)

def apply_prompt_format(
    template: str, *, think: bool = False, reasoning_prefill: bool = False
) -> str:
    """Apply prompt-format ablation knobs to a raw ``<|turn>`` template string.

    Both knobs are pure string transforms that preserve the ``{text}`` (and any
    other) ``str.format`` placeholders, so the result is still consumed by
    ``run_eval`` / the SFT collator exactly like the base template.

    * ``reasoning_prefill`` -- prime the model's turn by opening a ``<reasoning>``
      block right after the trailing ``<|turn>model\\n`` so the model continues its
      chain-of-thought immediately instead of re-emitting the tag.
    * ``think`` -- prepend the DiffusionGemma ``<|think|>`` thinking token to the
      very front of the prompt (dev-guide: thinking is enabled by prefixing the
      system/prompt content with ``<|think|>``; cf. examples/hf_inference.py:84).
    """
    if reasoning_prefill:
        template = template.rstrip("\n") + "\n<reasoning>\n"
    if think:
        template = "<|think|>" + template
    return template


# Task -> prompt template. --prompt_style selects one (or pass a literal template).
PROMPT_TEMPLATES = {
    "sudoku": DIFFUGEMMA_SUDOKU_PROMPT,
    "gsm8k": GSM8K_PROMPT,
    "math": MATH_PROMPT,
    "countdown": COUNTDOWN_PROMPT,
    "countdown_d2_train": COUNTDOWN_D2_TRAIN_PROMPT,
    "countdown_d2_eval": COUNTDOWN_D2_EVAL_PROMPT,
    "mbpp": MBPP_PROMPT,
    "humaneval": HUMANEVAL_PROMPT,
    "bugfix": BUGFIX_PROMPT,
}

# --prompt_style -> chat-route user-turn instruction (for --chat_route). The
# instruction carries the output-format block; the question is appended after it.
CHAT_INSTRUCTIONS = {
    "math": MATH_INSTRUCTION,
}


@dataclass
class DiffusionGemmaSFTCollator:
    """Tokenize prompt/completion text into the canvas layout."""

    tokenizer: Any
    canvas_length: int = 256
    max_prompt_length: int = 256
    prompt_column: str = "prompt"
    completion_column: str = "completion"
    # When set, the prompt column (the raw puzzle) is substituted into this
    # template ({text}) and tokenized as a literal string with add_bos -- the
    # JAX-exact path. When None, falls back to the tokenizer's chat template.
    prompt_template: str | None = DIFFUGEMMA_SUDOKU_PROMPT
    chat_template: str | None = None
    add_bos: bool = True
    # Chat route (matches sft_eval_mathbench._apply_chat): when chat_mode is set, the
    # prompt is built with tok.apply_chat_template -- chat_instruction + "\n\n" + the
    # raw prompt in the user turn, native <bos>/role separators, optional thinking
    # channel (enable_thinking) and a trailing REASONING_PREFILL. Overrides
    # prompt_template. The rendered string already carries <bos>, so it is tokenized
    # with add_special_tokens=False.
    chat_mode: bool = False
    chat_instruction: str | None = None
    enable_thinking: bool = False
    reasoning_prefill: bool = False
    # Block-structured (multi-canvas) SFT. When block_diffusion is set, the whole
    # completion (up to max_completion_length, +EOS) is split into canvas_length blocks;
    # per example a block index b is sampled, the clean prior blocks 0..b-1 are appended
    # to the encoder input after the prompt, block b becomes the (single) denoising canvas,
    # and the tail is dropped. Matches the block-autoregressive eval. See __call__.
    block_diffusion: bool = False
    max_completion_length: int = 4096
    # Upsample answer-content canvases (blocks containing the <answer> section) to this fraction
    # of sampled canvases; 0.0 = uniform block sampling. See _call_block.
    answer_content_frac: float = 0.0
    # tmax mode: row = {completion (pre-rendered trajectory), assistant_spans}; sample a response
    # span then a canvas within it, encode the whole prefix. See _call_tmax.
    tmax_response_sampling: bool = False

    def __post_init__(self):
        tok = self.tokenizer
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.eos_id = tok.eos_token_id

    @staticmethod
    def _as_int_list(x) -> list[int]:
        """Coerce tokenizer output to a plain list[int].

        Handles list[int], tokenizers.Encoding (.ids), and BatchEncoding/dict
        (["input_ids"]), so both fast tokenizers and raw tokenizers.Tokenizer work.
        """
        if hasattr(x, "ids"):  # tokenizers.Encoding
            return list(x.ids)
        if isinstance(x, dict) and "input_ids" in x:
            x = x["input_ids"]
        if x and isinstance(x[0], (list, tuple)):  # batched -> first row
            x = x[0]
        return [int(t) for t in x]

    def _encode_prompt(self, text: str) -> list[int]:
        tok = self.tokenizer
        # Chat route: render exactly like sft_eval_mathbench._apply_chat so training
        # matches eval byte-for-byte. apply_chat_template supplies <bos>, the native
        # role separators and (enable_thinking) the thought-channel priming; then
        # REASONING_PREFILL opens the CoT. Already has <bos> -> add_special_tokens=False.
        if self.chat_mode:
            content = (
                f"{self.chat_instruction}\n\n{text}" if self.chat_instruction else text
            )
            rendered = tok.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            if self.reasoning_prefill:
                rendered += REASONING_PREFILL
            ids = self._as_int_list(tok.encode(rendered, add_special_tokens=False))
            # left-truncate to keep the trailing model-turn header / prefill.
            return ids[-self.max_prompt_length :]
        # JAX-exact path: substitute the raw puzzle into the literal diffuGemma
        # prompt template and tokenize with add_bos (no chat template).
        if self.prompt_template is not None:
            full = self.prompt_template.format(text=text)
            ids = self._as_int_list(tok.encode(full, add_special_tokens=True))
            # right-side truncation would drop the trailing "<|turn>model\n" that
            # signals generation; keep the tail by left-truncating if needed.
            return ids[-self.max_prompt_length :]
        # Fallback: tokenizer chat template.
        try:
            out = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True, add_generation_prompt=True,
            )
            ids = self._as_int_list(out)
        except Exception:
            ids = self._as_int_list(tok.encode(text, add_special_tokens=self.add_bos))
        return ids[: self.max_prompt_length]

    def _encode_completion(self, text: str) -> list[int]:
        # When the prompt is prefilled with REASONING_PREFILL, the model's target
        # (canvas) begins with the reasoning BODY -- drop the leading <reasoning>\n so
        # it isn't taught twice (mirrors eval, where <reasoning>\n is in the prompt).
        if self.reasoning_prefill and text.startswith(REASONING_PREFILL):
            text = text[len(REASONING_PREFILL) :]
        ids = self._as_int_list(self.tokenizer.encode(text, add_special_tokens=False))
        return ids[: self.canvas_length - 1] + [self.eos_id]

    def _encode_completion_full(self, text: str) -> list[int]:
        """Block mode: the FULL completion (+EOS), capped at max_completion_length.

        Unlike ``_encode_completion`` (which truncates to a single canvas), this keeps the
        whole trace so ``__call__`` can split it into canvas_length blocks. The single EOS
        lives at the very end, so it only appears in the final block."""
        if self.reasoning_prefill and text.startswith(REASONING_PREFILL):
            text = text[len(REASONING_PREFILL) :]
        ids = self._as_int_list(self.tokenizer.encode(text, add_special_tokens=False))
        return ids[: self.max_completion_length - 1] + [self.eos_id]

    def _strip_prefill(self, text: str) -> str:
        """Same leading-<reasoning> strip _encode_completion_full applies (reasoning_prefill)."""
        if self.reasoning_prefill and text.startswith(REASONING_PREFILL):
            return text[len(REASONING_PREFILL):]
        return text

    def _answer_tok_index(self, text: str, n_full: int) -> int:
        """Token index (in the completion ids) where the ``<answer>`` section begins, via the
        fast tokenizer's char->token offset map. Returns ``n_full`` if ``<answer>`` is absent or
        truncated off (so every block classifies as reasoning-only)."""
        ac = text.rfind("<answer>")
        if ac < 0:
            return n_full
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        for i, (s, e) in enumerate(enc["offset_mapping"]):
            if s <= ac < e:
                return i if i < n_full else n_full
        return n_full

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        if self.tmax_response_sampling:
            return self._call_tmax(examples)
        if self.block_diffusion:
            return self._call_block(examples)
        P, L = self.max_prompt_length, self.canvas_length
        prompt_ids, prompt_mask, canvas, canvas_mask = [], [], [], []
        for ex in examples:
            p = self._encode_prompt(ex[self.prompt_column])
            c = self._encode_completion(ex[self.completion_column])
            # left-pad prompt (so the prompt ends adjacent to the canvas)
            pad_p = P - len(p)
            prompt_ids.append([self.pad_id] * pad_p + p)
            prompt_mask.append([0] * pad_p + [1] * len(p))
            # right-pad canvas
            n = len(c)
            canvas.append(c + [self.pad_id] * (L - n))
            canvas_mask.append([1] * n + [0] * (L - n))
        return {
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            # bool, not long: transformers' masking_utils (create_diffusion_decoder_
            # attention_mask -> sdpa_mask -> and_mask) combines masks via bitwise AND,
            # which requires a genuine bool tensor (int64 raises at the final SDPA
            # call; float raises earlier inside and_mask's bitwise_and).
            "prompt_mask": torch.tensor(prompt_mask, dtype=torch.bool),
            "canvas": torch.tensor(canvas, dtype=torch.long),
            "canvas_mask": torch.tensor(canvas_mask, dtype=torch.float32),
        }

    def _call_block(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        """Block-structured (multi-canvas) collation. Per example: tokenize the full
        completion, split into ``ceil(len/L)`` blocks of length ``L=canvas_length``, sample
        a block index ``b = min(floor(u*num_blocks), num_blocks-1)`` with ``u~U(0,1)``, then

          * encoder input = ``prompt + blocks[0..b-1]`` (clean prior blocks as context),
          * canvas        = block ``b`` (the denoising target, right-padded to ``L``),
          * ``prompt_comp_mask`` marks the prefix-completion tokens (blocks 0..b-1) -- the
            span the encoder AR loss scores; 0 on the prompt, so b=0 gives an all-zero mask.

        The tail (blocks ``b+1..``) is dropped. Because the encoder writes its output into
        the decoder's KV cache, block ``b``'s canvas is auto-positioned right after the
        encoder input -- identical to the block-autoregressive eval (see modeling forward)."""
        L = self.canvas_length
        enc_ids, enc_mask, enc_comp, canvas, canvas_mask, bins = [], [], [], [], [], []
        for ex in examples:
            p = self._encode_prompt(ex[self.prompt_column])
            full = self._encode_completion_full(ex[self.completion_column])
            num_blocks = max(1, math.ceil(len(full) / L))
            # `a` = <answer> start token index; b_a = first answer-content block (the transition).
            a = self._answer_tok_index(self._strip_prefill(ex[self.completion_column]), len(full))
            b_a = min(a // L, num_blocks - 1)
            acf = self.answer_content_frac
            if acf > 0 and a < len(full) and random.random() < acf:
                b = random.randint(b_a, num_blocks - 1)          # upsample: answer-content block
            elif acf > 0 and b_a >= 1:
                b = random.randint(0, b_a - 1)                   # complement: reasoning-only block
            else:
                b = min(int(random.random() * num_blocks), num_blocks - 1)  # uniform (acf=0 / fallback)
            prefix = full[: b * L]              # clean prior blocks 0..b-1
            block = full[b * L : (b + 1) * L]   # target block b (<= L tokens)
            e = p + prefix
            enc_ids.append(e)
            enc_mask.append([1] * len(e))
            enc_comp.append([0] * len(p) + [1] * len(prefix))
            n = len(block)
            canvas.append(block + [self.pad_id] * (L - n))
            canvas_mask.append([1] * n + [0] * (L - n))
            # mutually-exclusive canvas-type bin vs `a`:
            #   reasoning-only (hi<=a) | transition (lo<=a<hi) | answer-only (a<lo)
            lo, hi = b * L, min((b + 1) * L, len(full))
            bins.append([1.0, 0.0, 0.0] if hi <= a else
                        [0.0, 1.0, 0.0] if lo <= a < hi else
                        [0.0, 0.0, 1.0])
        # left-pad the (variable-length) encoder tensors to the batch max (no-op at batch 1)
        Pmax = max(len(e) for e in enc_ids)
        prompt_ids, prompt_mask, prompt_comp = [], [], []
        for e, m, cm in zip(enc_ids, enc_mask, enc_comp):
            pad = Pmax - len(e)
            prompt_ids.append([self.pad_id] * pad + e)
            prompt_mask.append([0] * pad + m)
            prompt_comp.append([0] * pad + cm)
        return {
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "prompt_mask": torch.tensor(prompt_mask, dtype=torch.long),
            "canvas": torch.tensor(canvas, dtype=torch.long),
            "canvas_mask": torch.tensor(canvas_mask, dtype=torch.float32),
            "prompt_comp_mask": torch.tensor(prompt_comp, dtype=torch.float32),
            "canvas_bins": torch.tensor(bins, dtype=torch.float32),  # [B,3] one-hot: reason/transition/answer
        }

    def _call_tmax(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        """Tmax response-anchored block collation. Each row = {completion: <pre-rendered trajectory
        (already chat-templated, carries <bos>)>, assistant_spans: [[s,e], ...] token offsets of
        each assistant response}. Per example: sample one response span [s,e]; tile it into
        canvas_length canvases; sample canvas j; canvas = completion[s+jL : min(s+(j+1)L, e)]
        (denoiser target); encoder input (prompt_ids) = completion[: s+jL] -- everything before the
        canvas, with prompt_comp_mask = 1 on ALL real prefix tokens so the encoder AR loss scores
        the whole prefix. Reuses the block loss (shared encoder forward + single backward)."""
        L = self.canvas_length
        enc_ids, enc_mask, canvas, canvas_mask = [], [], [], []
        for ex in examples:
            ids = self._as_int_list(self.tokenizer.encode(ex[self.completion_column], add_special_tokens=False))
            ids = ids[: self.max_completion_length]
            spans = [sp for sp in ex["assistant_spans"] if sp[1] <= len(ids) and sp[1] > sp[0]]
            if not spans:                       # safety: treat the whole sequence as one response
                spans = [[0, len(ids)]]
            s, e = spans[random.randrange(len(spans))]
            nb = max(1, math.ceil((e - s) / L))
            j = random.randrange(nb)
            cs, ce = s + j * L, min(s + (j + 1) * L, e)
            prefix = ids[:cs] or ids[:1]        # never an empty encoder input
            block = ids[cs:ce]
            enc_ids.append(prefix)
            enc_mask.append([1] * len(prefix))
            n = len(block)
            canvas.append(block + [self.pad_id] * (L - n))
            canvas_mask.append([1] * n + [0] * (L - n))
        # left-pad the (variable-length) encoder tensors to the batch max (no-op at batch 1)
        Pmax = max(len(x) for x in enc_ids)
        prompt_ids, prompt_mask, prompt_comp = [], [], []
        for x, m in zip(enc_ids, enc_mask):
            pad = Pmax - len(x)
            prompt_ids.append([self.pad_id] * pad + x)
            prompt_mask.append([0] * pad + m)
            prompt_comp.append([0] * pad + m)   # score ALL real prefix tokens (whole-prefix encoder AR)
        return {
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "prompt_mask": torch.tensor(prompt_mask, dtype=torch.long),
            "canvas": torch.tensor(canvas, dtype=torch.long),
            "canvas_mask": torch.tensor(canvas_mask, dtype=torch.float32),
            "prompt_comp_mask": torch.tensor(prompt_comp, dtype=torch.float32),
        }
