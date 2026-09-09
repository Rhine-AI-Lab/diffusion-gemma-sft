from dataclasses import dataclass, field
from typing import Optional

import trl


@dataclass
class GRPOConfig(trl.GRPOConfig):
    """Training arguments used by the GDSD trainers."""

    chat_template: Optional[str] = field(default=None, metadata={"help": "Optional chat template."})
    system_prompt: Optional[str] = field(default=None, metadata={"help": "Optional system prompt."})

    wandb_entity: Optional[str] = field(default=None, metadata={"help": "Optional W&B entity."})
    wandb_project: Optional[str] = field(default=None, metadata={"help": "Optional W&B project."})
    wandb_run_group: Optional[str] = field(default=None, metadata={"help": "Optional W&B run group."})

    random_masking: bool = field(default=True, metadata={"help": "Whether to use random masking."})
    p_mask_prompt: float = field(default=0.15, metadata={"help": "Probability of masking prompt tokens."})
    diffusion_steps: int = field(default=128, metadata={"help": "Number of diffusion generation steps."})
    generation_temperature: float = field(default=1.2, metadata={"help": "Diffusion generation temperature."})
    generation_batch_size: int = field(default=10, metadata={"help": "Batch size used during generation."})

    sample_train_steps: Optional[int] = field(default=1, metadata={"help": "Number of train samples."})
    num_mc: Optional[int] = field(default=1, metadata={"help": "Monte Carlo samples for ELBO estimation."})
    num_mc_at: Optional[int] = field(default=4, metadata={"help": "Monte Carlo samples for auxiliary estimates."})
    reduce_var: bool = field(default=True, metadata={"help": "Whether to use the coupled variance-reduction term."})

    rl_loss_type: str = field(
        default="gdsd",
        metadata={"help": "Trainer objective: 'gdsd', 'gdsd_tlc', 'espo', or 'spg'."},
    )
    psi: float = field(default=1.0, metadata={"help": "Scale coefficient for GDSD guidance."})

    # SPG-specific options. These defaults match the public SPG-style launch recipe.
    block_length: int = field(default=32, metadata={"help": "Diffusion block length for SPG."})
    cfg_scale: float = field(default=0.0, metadata={"help": "Classifier-free guidance scale for SPG."})
    remasking: str = field(default="low_confidence", metadata={"help": "Diffusion remasking strategy."})
    mask_id: int = field(default=126336, metadata={"help": "Mask token id for LLaDA-style models."})
    use_mask_prompt: bool = field(default=True, metadata={"help": "Whether SPG may mask prompt tokens."})
    forward_type: str = field(
        default="block_random",
        metadata={"help": "SPG forward process: all, random, block_all, or block_random."},
    )
    num_t: int = field(default=2, metadata={"help": "Number of timesteps sampled for SPG estimation."})
    min_t: float = field(default=0.0, metadata={"help": "Minimum SPG masking ratio."})
    max_t: float = field(default=1.0, metadata={"help": "Maximum SPG masking ratio."})
    logp_estimation: Optional[str] = field(
        default="mix",
        metadata={"help": "SPG estimate for negative-advantage traces: eubo, mix, elbo, or zero."},
    )
    eubo_beta: float = field(default=1.5, metadata={"help": "EUBO beta for SPG."})
    mix_weight: float = field(default=0.5, metadata={"help": "Mixing weight for SPG mix estimation."})


@dataclass
class GRPOScriptArguments(trl.ScriptArguments):
    """Dataset and reward arguments for the GDSD training entry point."""

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy"],
        metadata={"help": "Reward function names. Usually inferred from dataset_name."},
    )
    dataset_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional local dataset path. Used for sudoku/code datasets; public Hugging Face "
                "datasets are used when this is not set."
            )
        },
    )
    dataset_prompt_column: str = field(default="prompt", metadata={"help": "Column to use as prompts."})

    code_language: str = field(
        default="python",
        metadata={"help": "Programming language for code-format rewards.", "choices": ["python"]},
    )
    code_eval_test_batch_size: int = field(default=1, metadata={"help": "Reserved for code reward batching."})
    parallel_code_exec_per_proc: int = field(default=2, metadata={"help": "Local code executions per process."})
    code_provider: Optional[str] = field(
        default="local",
        metadata={"help": "Only the local provider is included in the anonymized supplement."},
    )

    max_completion_len: int = field(default=16384, metadata={"help": "Maximum completion characters."})
    soft_punish_cache: int = field(default=4096, metadata={"help": "Reserved for compatibility."})
