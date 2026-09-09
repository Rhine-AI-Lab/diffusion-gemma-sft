"""Entry point for DiffusionGemma SFT with TRL and Transformers.

Mirrors `train.py` (the RL entry) but runs the supervised canvas-denoising
objective via `DiffusionGemmaSFTTrainer`.

Example:
    accelerate launch -m DiffGemma.diffgemma_trl.sft_train \
        --model_name_or_path unsloth/diffusiongemma-26B-A4B-it \
        --dataset_name ./dataset/sudoku_sft \
        --dataset_path ./dataset/sudoku_sft \
        --sft_variant reweighted-ce \
        --output_dir ./xp_sft_reweighted_ce_trl
"""

from __future__ import annotations

import logging
import sys

import transformers
from transformers import set_seed
from trl import ModelConfig, TrlParser

from .model_utils import get_processing_tokenizer
from .sft_configs import DiffusionGemmaSFTConfig, DiffusionGemmaSFTScriptArguments
from .sft_data_utils import DiffusionGemmaSFTCollator, get_sft_dataset
from .sft_trainer import DiffusionGemmaSFTTrainer

logger = logging.getLogger(__name__)


def _load_diffgemma(model_args, training_args):
    """Load DiffusionGemma + tokenizer.

    Prefer the native Transformers DiffusionGemma class, with the repository's
    bundled implementation as a compatibility fallback. Unsloth's ``FastModel``
    path is used when ``use_unsloth`` is enabled and Unsloth is installed.
    """
    import os
    import torch

    if getattr(training_args, "use_unsloth", False):
        try:
            from unsloth import FastModel  # noqa: F401
            from .model_utils import load_model_and_processor
            return load_model_and_processor(model_args, training_args)
        except Exception as exc:  # unsloth not installed -> fall back to bundled HF
            logger.warning("Unsloth unavailable (%s); loading bundled HF model.", exc)
            training_args.use_unsloth = False

    from transformers import AutoProcessor, AutoTokenizer
    try:
        # transformers >= 5.11 ships diffusion_gemma natively.
        from transformers import DiffusionGemmaForBlockDiffusion
    except ImportError:
        from DiffGemma.diffusion_gemma.modeling_diffusion_gemma import (
            DiffusionGemmaForBlockDiffusion,
        )

    # MoE experts kernel selection. transformers auto-selects "grouped_mm" on
    # torch>=2.9 / SM90, but torch._grouped_mm asserts (GroupMMCommon.cuh: "dynamic
    # dimension byte size must be a multiple of 16") on the unaligned per-expert
    # token counts produced here — crashing the first forward on GH200. The
    # alternatives: "batched_mm" replicates [experts x tokens] and OOMs a single
    # 96GB GH200 at batch 4 (~60GB extra); "eager" loops per expert with a low
    # memory peak. Default to "eager" so the single-GPU command fits; override via
    # DIFFGEMMA_EXPERTS_IMPL=batched_mm when memory allows (e.g. smaller batch).
    experts_impl = os.environ.get("DIFFGEMMA_EXPERTS_IMPL", "eager")
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_args.attn_implementation,
        experts_implementation=experts_impl,
    )
    try:
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    except Exception:
        processor = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    return model, processor


def _unwrap_clippable_linears(model) -> None:
    """Replace Gemma4ClippableLinear wrappers with their inner nn.Linear (PEFT compat)."""
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
    except ImportError:
        return
    count = 0
    for _, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if isinstance(child, Gemma4ClippableLinear):
                setattr(parent, child_name, child.linear)
                count += 1
    if count:
        logger.info("Unwrapped %d Gemma4ClippableLinear -> nn.Linear for PEFT.", count)


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
    # kwargs["use_cache"] = getattr(self.config, "use_cache", None) into every
    # forward call whenever that attribute is not None -- and DiffusionGemma's
    # decoder explicitly raises if `use_cache` is present in kwargs at all (it
    # always caches internally). transformers.Trainer.__init__ unconditionally
    # does `self.model.config.use_cache = self.args.use_cache` (default True),
    # which *adds* the attribute and triggers the crash. Force both config
    # objects (text_config is a distinct object) to None so the decorator's
    # `is not None` check is always False regardless of aliasing.
    training_args.use_cache = None
    model.config.use_cache = None
    if getattr(model.config, "text_config", None) is not None:
        model.config.text_config.use_cache = None
    tokenizer = get_processing_tokenizer(processor)
    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template
    _unwrap_clippable_linears(model)

    # PEFT: when not using Unsloth's FastModel LoRA, apply a LoRA config here.
    peft_config = None
    if not (training_args.use_unsloth and training_args.unsloth_lora_rank > 0):
        from trl import get_peft_config
        peft_config = get_peft_config(model_args)
        if peft_config is not None:
            peft_config.task_type = None  # DiffusionGemma is not a CausalLM

    # transformers.Trainer (unlike trl trainers) takes no peft_config -> apply here.
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
    from .sft_data_utils import CHAT_INSTRUCTIONS, PROMPT_TEMPLATES
    if script_args.chat_route:
        # Chat route: prompts via apply_chat_template (matches the math-benchmark eval).
        chat_instruction = CHAT_INSTRUCTIONS.get(script_args.prompt_style)
        logger.info(
            "Chat route: apply_chat_template, instruction=%s, enable_thinking=%s, "
            "reasoning_prefill=%s",
            "<set>" if chat_instruction else "<none>",
            script_args.enable_thinking, script_args.reasoning_prefill,
        )
        collator = DiffusionGemmaSFTCollator(
            tokenizer=tokenizer,
            canvas_length=canvas_len,
            max_prompt_length=script_args.max_prompt_length,
            prompt_column=script_args.prompt_column,
            completion_column=script_args.completion_column,
            prompt_template=None,
            chat_mode=True,
            chat_instruction=chat_instruction,
            enable_thinking=script_args.enable_thinking,
            reasoning_prefill=script_args.reasoning_prefill,
            chat_template=training_args.chat_template,
            block_diffusion=training_args.block_diffusion,
            max_completion_length=training_args.max_completion_length,
            answer_content_frac=training_args.answer_content_frac,
            tmax_response_sampling=training_args.tmax_response_sampling,
        )
    else:
        prompt_template = PROMPT_TEMPLATES.get(script_args.prompt_style, script_args.prompt_style)
        collator = DiffusionGemmaSFTCollator(
            tokenizer=tokenizer,
            canvas_length=canvas_len,
            max_prompt_length=script_args.max_prompt_length,
            prompt_column=script_args.prompt_column,
            completion_column=script_args.completion_column,
            prompt_template=prompt_template,
            chat_template=training_args.chat_template,
            block_diffusion=training_args.block_diffusion,
            max_completion_length=training_args.max_completion_length,
            answer_content_frac=training_args.answer_content_frac,
            tmax_response_sampling=training_args.tmax_response_sampling,
        )
    dataset = get_sft_dataset(script_args)

    trainer = DiffusionGemmaSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor,
    )

    # Never train MoE router params via LoRA (mirrors the RL path).
    frozen = 0
    for name, p in trainer.model.named_parameters():
        if "router" in name and p.requires_grad:
            p.requires_grad = False
            frozen += 1
    if frozen:
        logger.info("Froze %d MoE router parameters.", frozen)

    logger.info("*** Train ***")
    result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    logger.info("Model saved to %s", training_args.output_dir)

    # Dump the portion of sampled canvases per mutually-exclusive type (block mode) so the
    # LR sweep can report the bins alongside eval. Only the main process writes.
    counts = getattr(trainer, "_bin_counts", None)
    if counts is not None:
        import json, os
        # all-reduce is a COLLECTIVE -> every rank must call it (outside the rank-0 guard),
        # then only rank 0 writes the file.
        acc = getattr(trainer, "accelerator", None)
        if acc is not None and getattr(acc, "num_processes", 1) > 1:
            counts = acc.reduce(counts.to(acc.device), reduction="sum").cpu()
    if counts is not None and trainer.is_world_process_zero():
        c = [float(x) for x in counts]
        tot = sum(c) or 1.0
        stats = {"counts": c, "total": sum(c),
                 "fracs": {"reasoning_only": c[0] / tot, "transition": c[1] / tot,
                           "answer_only": c[2] / tot}}
        with open(os.path.join(training_args.output_dir, "canvas_bin_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("Canvas-bin stats: %s", stats["fracs"])


if __name__ == "__main__":
    parser = TrlParser((DiffusionGemmaSFTScriptArguments, DiffusionGemmaSFTConfig, ModelConfig))
    main(*parser.parse_args_and_config())
