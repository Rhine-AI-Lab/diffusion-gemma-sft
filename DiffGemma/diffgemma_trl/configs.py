"""Command-line configs for DiffusionGemma TRL training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import trl


@dataclass
class DiffusionGemmaGRPOConfig(trl.GRPOConfig):
    """Training arguments for DiffusionGemma canvas-denoising RL."""

    rl_loss_type: str = field(
        default="gdsd",
        metadata={"help": "Objective to use: 'gdsd' or 'grpo'."},
    )

    chat_template: Optional[str] = field(default=None, metadata={"help": "Optional chat template override."})
    system_prompt: Optional[str] = field(default=None, metadata={"help": "Optional system prompt."})

    diffusion_steps: int = field(default=48, metadata={"help": "Denoising steps used during generation."})
    diffusion_canvas_length: Optional[int] = field(
        default=None,
        metadata={"help": "Training canvas length. Defaults to model.config.canvas_length."},
    )
    diffusion_noise_min: float = field(default=0.1, metadata={"help": "Minimum random-token corruption rate."})
    diffusion_noise_max: float = field(default=1.0, metadata={"help": "Maximum random-token corruption rate."})
    diffusion_num_mc: int = field(default=1, metadata={"help": "Monte Carlo samples for denoise logprob estimates."})
    diffusion_logp_variant: str = field(
        default="base-sft",
        metadata={"help": "Surrogate for the RL denoise log-prob: 'base-sft' (plain "
                          "denoise CE, current) or 'loo-ce' (leave-one-out->denoiser "
                          "logit correction, matching the loo-ce SFT objective)."},
    )
    canvas_window: str = field(
        default="start",
        metadata={"help": "When max_completion_length > canvas_length, which "
                          "canvas_length-token window of the completion is scored by "
                          "the RL loss: 'start' (first tokens, current default) or "
                          "'end' (last tokens of the real completion, before EOS "
                          "padding). Use 'end' for long chain-of-thought tasks (e.g. "
                          "DAPO-math) where the answer sits near the end of a "
                          "completion longer than canvas_length -- 'start' trains the "
                          "loss on a region decoupled from the reward."},
    )
    diffusion_reduce_var: bool = field(
        default=False,
        metadata={"help": "Reserved for a coupled estimator. The first implementation keeps this off."},
    )
    diffusion_self_conditioning_prob: float = field(
        default=0.0,
        metadata={"help": "Probability of self-conditioning during denoise logprob estimation."},
    )
    generation_batch_size: int = field(
        default=1,
        metadata={"help": "Micro-batch size for rollout generation inside _generate_and_score_completions."},
    )

    diffusion_t_min: float = field(default=0.4, metadata={"help": "Generation temperature schedule minimum."})
    diffusion_t_max: float = field(default=0.8, metadata={"help": "Generation temperature schedule maximum."})
    diffusion_entropy_bound: float = field(default=0.1, metadata={"help": "Entropy-bound sampler threshold."})
    diffusion_stability_threshold: int = field(default=1, metadata={"help": "Adaptive stopping stability threshold."})
    diffusion_confidence_threshold: float = field(
        default=0.005,
        metadata={"help": "Adaptive stopping confidence threshold."},
    )

    psi: float = field(default=1.0, metadata={"help": "Scale coefficient for the GDSD square objective."})
    wd1_temperature: float = field(
        default=1.0,
        metadata={"help": "wd1 (arXiv 2507.08838) softmax temperature tau for advantage "
                          "weighting of the denoise log-likelihood. 1.0 matches the reference "
                          "impl (xiaohangt/wd1). Used only with rl_loss_type=wd1."},
    )

    # TRL 1.6.0 removed max_prompt_length from GRPOConfig; re-declare for our trainer.
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={"help": "Maximum prompt token length. Prompts will be left-truncated to this length."},
    )

    tie_encoder_decoder_lora: bool = field(
        default=False,
        metadata={"help": "Share one LoRA adapter across the tied encoder.language_model "
                          "<-> decoder weights (single delta trained by both passes), "
                          "instead of two independent adapters. Fixes the tied-weight "
                          "double-adapter (PEFT #1035) and halves the text LoRA params."},
    )

    use_unsloth: bool = field(default=True, metadata={"help": "Load DiffusionGemma through Unsloth FastModel."})
    # NOTE: load_in_4bit is provided by TRL's ModelConfig; do NOT redeclare here.
    unsloth_lora_rank: int = field(default=64, metadata={"help": "LoRA rank for FastModel.get_peft_model."})
    unsloth_lora_alpha: int = field(default=128, metadata={"help": "LoRA alpha for FastModel.get_peft_model."})
    unsloth_gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Passed to FastModel.get_peft_model(use_gradient_checkpointing=...)."},
    )

    wandb_entity: Optional[str] = field(default=None, metadata={"help": "Optional W&B entity."})
    wandb_project: Optional[str] = field(default=None, metadata={"help": "Optional W&B project."})
    wandb_run_group: Optional[str] = field(default=None, metadata={"help": "Optional W&B run group."})


@dataclass
class DiffusionGemmaScriptArguments(trl.ScriptArguments):
    """Dataset and reward arguments for DiffusionGemma TRL training."""

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy"],
        metadata={"help": "Reward function names. Usually inferred from dataset_name."},
    )
    dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional local dataset path for json/jsonl/sudoku/code-style datasets."},
    )
    dataset_prompt_column: str = field(default="prompt", metadata={"help": "Column to use as prompts."})
    sft_adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to an SFT LoRA adapter to start RL from. By default it is "
                          "CONTINUED as the RL policy (loaded trainable; no merge, no fresh "
                          "adapter), reusing its adapter_config so --lora_* / --use_peft are "
                          "ignored. Set --sft_adapter_merge to instead merge it into the base "
                          "and train a fresh RL adapter (d1 convention)."},
    )
    sft_adapter_merge: bool = field(
        default=False,
        metadata={"help": "With --sft_adapter_path, MERGE the SFT adapter into the base "
                          "(in-memory merge_and_unload) and train a FRESH RL LoRA on top "
                          "(requires --use_peft + --lora_*). Off = continue the SFT adapter."},
    )

    code_language: str = field(default="python", metadata={"help": "Reserved for code rewards."})
    parallel_code_exec_per_proc: int = field(default=2, metadata={"help": "Reserved for code rewards."})
    code_provider: Optional[str] = field(default="local", metadata={"help": "Reserved for code rewards."})
