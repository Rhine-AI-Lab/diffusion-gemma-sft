"""Config + script arguments for DiffusionGemma SFT (TRL/transformers Trainer)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import transformers
import trl

from .sft_loss import VARIANTS, normalize_variant


@dataclass
class DiffusionGemmaSFTConfig(transformers.TrainingArguments):
    """TrainingArguments + DiffusionGemma SFT knobs.

    Mirrors the JAX hackable_diffusion SFT config so the PyTorch/TRL pipeline can
    reproduce the same four objectives with
    self-conditioning, the encoder AR loss, the uniform-state corruption, the
    rectified-flow schedule, and a configurable diffusion-time range.
    """

    # --- objective ---
    sft_variant: str = field(
        default="base-sft",
        metadata={"help": f"SFT loss variant. One of {VARIANTS}."},
    )
    decoder_loss_weight: float = field(
        default=1.0, metadata={"help": "Weight on the canvas denoising loss."}
    )
    encoder_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight on the encoder AR (next-token) loss over [prompt; x0]. "
                          "Set 0.0 for a decoder-only (Unsloth-style) objective."},
    )

    # --- diffusion forward process / schedule ---
    diffusion_vocab_size: int = field(
        default=262144, metadata={"help": "Vocab size K (used by the LOO shift)."}
    )
    diffusion_canvas_length: Optional[int] = field(
        default=None, metadata={"help": "Canvas length; defaults to model config."}
    )
    block_diffusion: bool = field(
        default=False,
        metadata={"help": "Block-structured (multi-canvas) SFT: per example sample a block "
                          "index b, feed the clean prior blocks 0..b-1 into the encoder as "
                          "context, denoise block b (one canvas of diffusion_canvas_length), "
                          "and drop the tail. Matches the block-autoregressive eval. Off = the "
                          "legacy single-canvas objective (whole completion in one canvas)."},
    )
    max_completion_length: int = field(
        default=4096,
        metadata={"help": "block_diffusion: cap on the full (un-blocked) completion length in "
                          "tokens before it is split into canvas_length blocks."},
    )
    answer_content_frac: float = field(
        default=0.0,
        metadata={"help": "block_diffusion: upsample answer-content canvases (blocks containing "
                          "the <answer> section) to this fraction of sampled canvases. 0 = uniform "
                          "block sampling (default). E.g. 0.4 -> ~40% of training canvases are the "
                          "reasoning->answer transition/answer, teaching the model to box."},
    )
    tmax_response_sampling: bool = field(
        default=False,
        metadata={"help": "block_diffusion (tmax mode): the row is a pre-rendered multi-turn "
                          "trajectory (`completion`) + `assistant_spans`. Per step sample one "
                          "assistant response span, tile it into canvas_length canvases, sample a "
                          "canvas (denoiser target); encode everything before it (whole-prefix "
                          "encoder AR loss). No prompt/completion split, no chat re-templating."},
    )
    interleaved_backward: bool = field(
        default=False,
        metadata={"help": "Do decoder fwd+bwd then encoder fwd+bwd so only one activation graph "
                          "is resident at a time (~halves peak memory; needed for the 4096 "
                          "single-canvas run). Off = single combined dec+enc backward (fine for "
                          "the 256-token block-diffusion canvas). Requires encoder_loss_weight>0."},
    )
    diffusion_noise_min: float = field(
        default=1e-3, metadata={"help": "Min diffusion time t (uniform corruption rate)."}
    )
    diffusion_noise_max: float = field(
        default=1.0 - 1e-3, metadata={"help": "Max diffusion time t."}
    )
    diffusion_weight_clip: Optional[float] = field(
        default=None,
        metadata={"help": "Optional cap on the SFT diffusion time weight w(t). "
                          "Use with reweighted-ce to bound 1/(1-t)."},
    )
    diffusion_self_conditioning_prob: float = field(
        default=0.5,
        metadata={"help": "Probability of feeding the (detached) first-pass logits "
                          "as self-conditioning to a second forward pass."},
    )

    # --- model loading / LoRA (consumed by model_utils.load_model_and_processor) ---
    use_unsloth: bool = field(default=True, metadata={"help": "Load DiffusionGemma via Unsloth FastModel."})
    unsloth_lora_rank: int = field(default=64, metadata={"help": "LoRA rank (Unsloth path)."})
    unsloth_lora_alpha: int = field(default=128, metadata={"help": "LoRA alpha = 2*rank (Unsloth path)."})
    unsloth_gradient_checkpointing: bool = field(
        default=False, metadata={"help": "Gradient checkpointing in FastModel.get_peft_model."}
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "Optional chat template override."})

    def __post_init__(self):
        super().__post_init__()
        self.sft_variant = normalize_variant(self.sft_variant)
        if self.diffusion_noise_max < self.diffusion_noise_min:
            raise ValueError("diffusion_noise_max must be >= diffusion_noise_min.")
        if self.diffusion_weight_clip is not None and self.diffusion_weight_clip <= 0:
            raise ValueError("diffusion_weight_clip must be positive when set.")


@dataclass
class DiffusionGemmaSFTScriptArguments(trl.ScriptArguments):
    """Dataset/model script arguments for SFT."""

    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "HF dataset id or local path."}
    )
    prompt_column: str = field(default="prompt", metadata={"help": "Prompt text column."})
    completion_column: str = field(
        default="completion", metadata={"help": "Target completion text column."}
    )
    max_prompt_length: int = field(
        default=256, metadata={"help": "Max prompt tokens (truncate/pad)."}
    )
    prompt_style: str = field(
        default="sudoku",
        metadata={"help": "Prompt template: a key in PROMPT_TEMPLATES (sudoku|gsm8k) "
                          "or a literal template string containing {text}."},
    )
    system_prompt: Optional[str] = field(default=None)

    # --- chat route (match the math-benchmark eval's apply_chat_template prompts) ---
    chat_route: bool = field(
        default=False,
        metadata={"help": "Build prompts via tok.apply_chat_template (chat_instruction "
                          "from prompt_style) instead of the raw <|turn> template. "
                          "Matches sft_eval_mathbench's default chat route."},
    )
    reasoning_prefill: bool = field(
        default=False,
        metadata={"help": "Prefill '<reasoning>\\n' at the end of the prompt (and drop it "
                          "from the completion). Match the eval's --reasoning_prefill."},
    )
    enable_thinking: bool = field(
        default=False,
        metadata={"help": "apply_chat_template enable_thinking flag (chat route)."},
    )
