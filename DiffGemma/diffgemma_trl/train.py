"""Training entry point for DiffusionGemma TRL/Unsloth experiments."""

from __future__ import annotations

import logging
import sys

import datasets
import transformers
from transformers import set_seed
from trl import ModelConfig, TrlParser, get_peft_config

from .configs import DiffusionGemmaGRPOConfig, DiffusionGemmaScriptArguments
from .data_utils import get_datasets
from .model_utils import get_processing_tokenizer, load_model_and_processor
from .rewards import get_reward_funcs
from .trainer import DiffusionGemmaGDSDTrainer, DiffusionGemmaGRPOTrainer, DiffusionGemmaWD1Trainer
from .wandb_logging import init_wandb_training


logger = logging.getLogger(__name__)


def _unwrap_clippable_linears(model: "PreTrainedModel") -> None:
    """Replace Gemma4ClippableLinear wrappers with their inner nn.Linear.

    transformers 5.x uses Gemma4ClippableLinear (a wrapper around nn.Linear with
    optional input/output clipping) for attention projections. PEFT cannot inject
    LoRA into non-standard module types, so we unwrap them in-place.
    """
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
    except ImportError:
        # Not using a transformers version that has Gemma4ClippableLinear; nothing to do.
        return

    count = 0
    for parent_name, parent_module in model.named_modules():
        for child_name, child_module in list(parent_module.named_children()):
            if isinstance(child_module, Gemma4ClippableLinear):
                # Replace the wrapper with its inner nn.Linear
                setattr(parent_module, child_name, child_module.linear)
                count += 1

    if count > 0:
        logger.info("Unwrapped %d Gemma4ClippableLinear -> nn.Linear for PEFT compatibility.", count)


def _tie_encoder_decoder_lora(model) -> None:
    """Share LoRA params across the tied encoder.language_model<->decoder weights.

    DiffusionGemma ties ~505 attn/MLP weights between encoder.language_model and the
    decoder, but PEFT injects an INDEPENDENT adapter on each view -> two deltas on one
    physical weight (also breaks merging, PEFT #1035). This reassigns each decoder LoRA
    module's lora_A/lora_B/scaling to the matching encoder.language_model module's, so a
    single shared delta is trained by both forward passes. Only ties pairs whose base
    weights are physically shared (data_ptr match); the vision tower stays independent.
    """
    mods = dict(model.named_modules())
    n_tied = 0
    for name, dec_mod in mods.items():
        if "decoder.layers." not in name or not hasattr(dec_mod, "lora_A"):
            continue
        enc_mod = mods.get(name.replace("decoder.layers.", "encoder.language_model.layers."))
        if enc_mod is None or not hasattr(enc_mod, "lora_A"):
            continue
        try:
            if dec_mod.base_layer.weight.data_ptr() != enc_mod.base_layer.weight.data_ptr():
                continue
        except AttributeError:
            continue
        dec_mod.lora_A = enc_mod.lora_A
        dec_mod.lora_B = enc_mod.lora_B
        dec_mod.scaling = enc_mod.scaling
        n_tied += 1
    logger.info("Tied encoder<->decoder LoRA on %d modules (shared adapter).", n_tied)


def main(script_args: DiffusionGemmaScriptArguments, training_args: DiffusionGemmaGRPOConfig, model_args: ModelConfig):
    set_seed(training_args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info("Model parameters: %s", model_args)
    logger.info("Script parameters: %s", script_args)
    logger.info("Training parameters: %s", training_args)

    if training_args.report_to and "wandb" in training_args.report_to:
        if training_args.local_rank <= 0:
            init_wandb_training(training_args)
        else:
            # Disable wandb on non-main ranks to avoid multiple runs
            import os
            os.environ["WANDB_MODE"] = "disabled"

    dataset = get_datasets(script_args.dataset_name, dataset_path=script_args.dataset_path)
    reward_funcs = get_reward_funcs(script_args)

    logger.info("*** Loading DiffusionGemma ***")
    model, processor = load_model_and_processor(model_args, training_args)
    tokenizer = get_processing_tokenizer(processor)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"
    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    # Unwrap Gemma4ClippableLinear → nn.Linear for PEFT compatibility.
    # Gemma4ClippableLinear is a wrapper that adds optional clipping around nn.Linear;
    # PEFT does not recognize it as a supported LoRA target module type.
    _unwrap_clippable_linears(model)

    # RL on top of an SFT checkpoint. Two modes:
    #   continue (default): load the SFT LoRA trainable and keep optimizing it.
    #   merge (--sft_adapter_merge): merge it into the base in-memory, then train a
    #     FRESH RL LoRA on top (d1 convention). In-memory merge avoids the on-disk
    #     key mismatch (unwrapped q_proj.weight vs the model's wrapped q_proj.linear).
    sft_adapter_path = getattr(script_args, "sft_adapter_path", None)
    merge_sft_adapter = bool(getattr(script_args, "sft_adapter_merge", False))
    continue_sft_adapter = bool(sft_adapter_path) and not merge_sft_adapter
    if sft_adapter_path:
        from peft import PeftModel
        if merge_sft_adapter:
            logger.info("Merging SFT LoRA into base, then training a fresh RL adapter: %s", sft_adapter_path)
            model = PeftModel.from_pretrained(model, sft_adapter_path).merge_and_unload()
        else:
            logger.info("Continuing SFT LoRA adapter (trainable): %s", sft_adapter_path)
            model = PeftModel.from_pretrained(model, sft_adapter_path, is_trainable=True)

    trainer_cls = {
        "gdsd": DiffusionGemmaGDSDTrainer,
        "diffgemma_gdsd": DiffusionGemmaGDSDTrainer,
        "grpo": DiffusionGemmaGRPOTrainer,
        "diffgemma_grpo": DiffusionGemmaGRPOTrainer,
        "wd1": DiffusionGemmaWD1Trainer,
        "diffu_wd1": DiffusionGemmaWD1Trainer,
    }.get(training_args.rl_loss_type)
    if trainer_cls is None:
        raise ValueError("Unsupported --rl_loss_type. Use one of: gdsd, diffgemma_gdsd, grpo, diffgemma_grpo, wd1.")

    # Skip fresh PEFT when continuing an SFT adapter (model is already a PeftModel)
    # or when Unsloth applied its own LoRA.
    if continue_sft_adapter or (training_args.use_unsloth and training_args.unsloth_lora_rank > 0):
        peft_config = None
    else:
        peft_config = get_peft_config(model_args)
    # DiffusionGemmaForBlockDiffusion is NOT a CausalLM; override task_type to avoid
    # PeftModelForCausalLM which requires prepare_inputs_for_generation.
    if peft_config is not None:
        peft_config.task_type = None
    trainer = trainer_cls(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # Freeze MoE router parameters - they should never be trained via LoRA.
    # Must be done AFTER Trainer init (which applies PEFT) to ensure flags stick.
    frozen_count = 0
    for name, param in trainer.model.named_parameters():
        if "router" in name:
            param.requires_grad = False
            frozen_count += 1
    if frozen_count > 0:
        logger.info("Froze %d MoE router parameters.", frozen_count)

    # Optionally share one LoRA adapter across the tied encoder.language_model<->decoder
    # weights (single delta trained by both forward passes). Must be AFTER PEFT init.
    if getattr(training_args, "tie_encoder_decoder_lora", False):
        _tie_encoder_decoder_lora(trainer.model)

    logger.info("*** Train ***")
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info("Model saved to %s", training_args.output_dir)

    if trainer.accelerator.is_main_process:
        trainer.create_model_card(dataset_name=script_args.dataset_name, tags=["diffusiongemma", "trl", "grpo"])
        if hasattr(trainer.model, "config"):
            trainer.model.config.use_cache = True
            trainer.model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((DiffusionGemmaScriptArguments, DiffusionGemmaGRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
