"""SFT entry point for the terminal-agent self-distillation dataset.

Mirrors DiffGemma/diffgemma_trl/sft_train.py (same model loading, PEFT
wiring, DiffusionGemmaSFTTrainer, loss variants) but swaps in
DiffusionGemmaAgentSFTCollator + a plain JSONL loader for the
messages/completion dataset produced by build_sft_data.py, instead of the
flat prompt/completion + template path built for sudoku/gsm8k/countdown.

Example:
    accelerate launch -m tblite_eval.train_agent_sft \
        --model_name_or_path google/diffusiongemma-26B-A4B-it \
        --dataset_path tblite_eval/agent_sft_data.jsonl \
        --sft_variant base-sft \
        --use_peft true --lora_r 64 --lora_alpha 128 \
        --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
        --diffusion_canvas_length 1024 --max_prompt_length 4096 \
        --output_dir ./tblite_eval/xp_agent_sft
"""

from __future__ import annotations

import logging
import sys

import transformers
from transformers import set_seed
from trl import ModelConfig, TrlParser

from DiffGemma.diffgemma_trl.model_utils import get_processing_tokenizer
from DiffGemma.diffgemma_trl.sft_configs import (
    DiffusionGemmaSFTConfig,
    DiffusionGemmaSFTScriptArguments,
)
from DiffGemma.diffgemma_trl.sft_train import _load_diffgemma, _unwrap_clippable_linears
from DiffGemma.diffgemma_trl.sft_trainer import DiffusionGemmaSFTTrainer

from tblite_eval.agent_sft_data_utils import (
    DiffusionGemmaAgentSFTCollator,
    get_agent_sft_dataset,
)

logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(training_args.get_process_log_level())
    transformers.utils.logging.set_verbosity(training_args.get_process_log_level())

    logger.info("*** Loading DiffusionGemma (variant=%s) ***", training_args.sft_variant)
    model, processor = _load_diffgemma(model_args, training_args)
    # transformers' @merge_with_config_defaults decorator injects
    # kwargs["use_cache"] = getattr(self.config, "use_cache", None) into
    # every forward call *whenever that attribute is not None* -- and
    # DiffusionGemma's decoder explicitly raises if `use_cache` is present in
    # kwargs at all (it always caches internally, regardless of value). A
    # fresh config has no `use_cache` attribute at all (getattr returns None,
    # decorator no-ops), but transformers.Trainer.__init__ unconditionally
    # does `self.model.config.use_cache = self.args.use_cache` (default
    # True), which *adds* the attribute and triggers the crash. Since
    # `text_config` is a distinct config object from the top-level one (not
    # just a view), and it's ambiguous which one the decoder submodule
    # actually reads, force both to None explicitly so the decorator's
    # `getattr(..., None) is not None` check is always False regardless of
    # aliasing. training_args.use_cache is also set to None (not False) so
    # if Trainer's own reassignment does reach one of these objects, it's a
    # harmless no-op instead of re-triggering the bug.
    training_args.use_cache = None
    model.config.use_cache = None
    if getattr(model.config, "text_config", None) is not None:
        model.config.text_config.use_cache = None
    tokenizer = get_processing_tokenizer(processor)
    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template
    _unwrap_clippable_linears(model)

    peft_config = None
    if not (training_args.use_unsloth and training_args.unsloth_lora_rank > 0):
        from trl import get_peft_config
        peft_config = get_peft_config(model_args)
        if peft_config is not None:
            peft_config.task_type = None  # DiffusionGemma is not a CausalLM

    if peft_config is not None:
        from peft import get_peft_model
        model = get_peft_model(model, peft_config)
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    canvas_len = training_args.diffusion_canvas_length or int(
        getattr(getattr(model.config, "text_config", model.config), "canvas_length", 256)
    )
    collator = DiffusionGemmaAgentSFTCollator(
        tokenizer=tokenizer,
        canvas_length=canvas_len,
        max_prompt_length=script_args.max_prompt_length,
        prompt_column=script_args.prompt_column,
        completion_column=script_args.completion_column,
        prompt_template=None,  # unused: _encode_prompt is overridden
    )
    dataset = get_agent_sft_dataset(script_args.dataset_path)

    trainer = DiffusionGemmaSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor,
    )

    frozen = 0
    for name, p in trainer.model.named_parameters():
        if "router" in name and p.requires_grad:
            p.requires_grad = False
            frozen += 1
    if frozen:
        logger.info("Froze %d MoE router parameters.", frozen)

    # Opt-in per-layer gradient checkpointing (DIFFGEMMA_GRAD_CKPT=1). NOT
    # needed for the standard LoRA recipe (that fits ~121GB on one H200), but
    # available for heavier configs (full fine-tune, larger batch). The model
    # class sets `supports_gradient_checkpointing = False`, so the top-level
    # `gradient_checkpointing_enable()` refuses -- but the encoder/decoder text
    # layers are `GradientCheckpointingLayer` subclasses whose __call__ already
    # checkpoints when `self.gradient_checkpointing and self.training`, so
    # flipping the flag + a checkpoint fn directly bypasses the disabled API.
    import os
    if os.environ.get("DIFFGEMMA_GRAD_CKPT", "0") == "1":
        from functools import partial
        import torch.utils.checkpoint as _cp
        from transformers.modeling_layers import GradientCheckpointingLayer
        _fn = partial(_cp.checkpoint, use_reentrant=False)
        gc_n = 0
        for _m in trainer.model.modules():
            if isinstance(_m, GradientCheckpointingLayer):
                _m.gradient_checkpointing = True
                _m._gradient_checkpointing_func = _fn
                gc_n += 1
        logger.info("Enabled per-layer gradient checkpointing on %d layers.", gc_n)

    logger.info("*** Train ***")
    result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    logger.info("Model saved to %s", training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((DiffusionGemmaSFTScriptArguments, DiffusionGemmaSFTConfig, ModelConfig))
    main(*parser.parse_args_and_config())
