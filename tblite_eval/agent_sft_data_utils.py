"""Dataset loading + collation for the terminal-agent distillation SFT.

Unlike DiffGemma.diffgemma_trl.sft_data_utils (flat prompt-string + template,
built for sudoku/gsm8k/countdown), each example here carries a full multi-turn
`messages` list (as produced by tblite_eval/build_sft_data.py from real
terminus-2 rollouts) that must go through the tokenizer's own chat template
(tool-call formatting and all) to match what the model actually saw at
rollout time. Only the prompt-encoding side differs; canvas/completion
encoding and the collator's output shape are identical to the sudoku
collator's, so DiffusionGemmaSFTTrainer/sft_loss need no changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import datasets
import torch

from DiffGemma.diffgemma_trl.sft_data_utils import DiffusionGemmaSFTCollator


def get_agent_sft_dataset(jsonl_path: str) -> datasets.Dataset:
    return datasets.load_dataset("json", data_files=jsonl_path)["train"]


@dataclass
class DiffusionGemmaAgentSFTCollator(DiffusionGemmaSFTCollator):
    """Same canvas layout as the base collator; prompt = chat-templated
    multi-turn history instead of a single templated/flat string."""

    def _encode_prompt(self, messages: list[dict]) -> list[int]:
        tok = self.tokenizer
        out = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if hasattr(out, "input_ids"):  # BatchEncoding: not a dict subclass
            out = out["input_ids"]
        ids = self._as_int_list(out)
        # Left-truncate: keep the tail (most recent turns + the trailing
        # generation-prompt marker), matching the sudoku collator's rationale.
        return ids[-self.max_prompt_length :]

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        batch = super().__call__(examples)
        # The base collator emits prompt_mask as int64 while canvas_mask is
        # float32; sft_trainer.py torch.cats them into a single
        # decoder_attention_mask that transformers' newer masking_utils
        # (create_diffusion_decoder_attention_mask -> sdpa_mask -> and_mask)
        # combines via bitwise AND, which requires a genuine bool tensor
        # (int64 or float both raise, just at different stages). Cast here
        # rather than in the shared base collator, since it's unclear whether
        # the sudoku path currently exercises this code path under the same
        # torch/transformers versions.
        batch["prompt_mask"] = batch["prompt_mask"].to(torch.bool)
        return batch
