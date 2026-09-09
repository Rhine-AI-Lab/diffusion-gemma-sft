#!/usr/bin/env python3
"""Commented Hugging Face inference example for DiffusionGemma.

This file is intentionally small: it demonstrates the official Transformers
path for inference and keeps every DiffusionGemma-specific knob visible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


DEFAULT_MODEL_ID = "google/diffusiongemma-26B-A4B-it"


def import_diffusiongemma():
    """Import the latest DiffusionGemma APIs with a clear error message."""
    try:
        from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
        from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
            DiffusionGemmaGenerationConfig,
            EntropyBoundSamplerConfig,
        )
    except Exception as exc:  # pragma: no cover - depends on installed package.
        raise RuntimeError(
            "This example needs a recent Transformers build that contains "
            "`DiffusionGemmaForBlockDiffusion`. Install with:\n"
            "  pip install -U transformers torch accelerate"
        ) from exc

    return (
        AutoProcessor,
        DiffusionGemmaForBlockDiffusion,
        DiffusionGemmaGenerationConfig,
        EntropyBoundSamplerConfig,
    )


def parse_dtype(dtype: str) -> str | torch.dtype:
    """Map friendly CLI strings to Torch dtypes accepted by from_pretrained."""
    if dtype == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]


def move_inputs_to_model(inputs: Any, model: torch.nn.Module) -> Any:
    """Move processor outputs to the device expected by the model.

    With `device_map=auto`, Transformers normally exposes `model.device` as the
    first execution device. Keeping this helper separate makes the example easy
    to adapt for manual device placement.
    """
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    return inputs.to(device) if hasattr(inputs, "to") else inputs


def image_part(image: str) -> dict[str, str]:
    """Build one image content part for the chat template.

    The HF processor accepts URLs. Recent versions also accept local file paths
    through the same multimodal chat-template route.
    """
    if image.startswith(("http://", "https://")):
        return {"type": "image", "url": image}
    return {"type": "image", "path": image}


def build_messages(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Create Gemma chat messages.

    The model card recommends putting images before text for multimodal input.
    The developer guide says thinking can be enabled by prefixing the system
    prompt with `<|think|>`.
    """
    messages: list[dict[str, Any]] = []

    system_prompt = args.system_prompt or ""
    if args.thinking:
        system_prompt = "<|think|>" + system_prompt
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if args.image:
        content = [image_part(image) for image in args.image]
        content.append({"type": "text", "text": args.prompt})
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": args.prompt})

    return messages


def build_generation_config(args: argparse.Namespace, processor: Any) -> Any:
    """Build DiffusionGemma generation config with official sampling defaults."""
    _, _, DiffusionGemmaGenerationConfig, EntropyBoundSamplerConfig = (
        import_diffusiongemma()
    )

    return DiffusionGemmaGenerationConfig(
        # Output length. The model denoises in 256-token canvases internally,
        # so non-multiples of 256 are fine; generation truncates via stopping.
        max_new_tokens=args.max_new_tokens,
        # Official recommended diffusion sampling setup.
        max_denoising_steps=args.max_denoising_steps,
        sampler_config=EntropyBoundSamplerConfig(entropy_bound=args.entropy_bound),
        t_max=args.t_max,
        t_min=args.t_min,
        stability_threshold=args.stability_threshold,
        confidence_threshold=args.confidence_threshold,
        # Do not override eos_token_id/pad_token_id here: DiffusionGemma's
        # generation config has model-specific defaults, including multiple
        # end tokens.
        cache_implementation=args.cache_implementation or None,
    )


def main() -> None:
    args = parse_args()
    AutoProcessor, DiffusionGemmaForBlockDiffusion, _, _ = import_diffusiongemma()

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir or None,
        local_files_only=args.local_files_only,
    )

    # dtype="auto" and device_map="auto" match the model-card usage and let
    # Accelerate place the 26B MoE model across available devices.
    load_kwargs = {
        "device_map": args.device_map,
        "cache_dir": args.cache_dir or None,
        "local_files_only": args.local_files_only,
    }
    try:
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            args.model_id,
            dtype=parse_dtype(args.dtype),
            **load_kwargs,
        )
    except TypeError:
        # Compatibility for intermediate Transformers builds.
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            args.model_id,
            torch_dtype=parse_dtype(args.dtype),
            **load_kwargs,
        )
    model.eval()

    messages = build_messages(args)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = move_inputs_to_model(inputs, model)

    generation_config = build_generation_config(args, processor)
    with torch.inference_mode():
        outputs = model.generate(**inputs, generation_config=generation_config)

    sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
    prompt_length = inputs["input_ids"].shape[-1]
    generated_ids = sequences[0, prompt_length:]
    tokenizer = getattr(processor, "tokenizer", processor)
    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=args.skip_special_tokens,
    )

    print(text)
    if hasattr(outputs, "tokens_per_forward"):
        print(f"\ntokens_per_forward={outputs.tokens_per_forward}")

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(
                {
                    "model_id": args.model_id,
                    "messages": messages,
                    "generation": text,
                    "tokens_per_forward": getattr(
                        getattr(outputs, "tokens_per_forward", None),
                        "tolist",
                        lambda: None,
                    )(),
                },
                f,
                indent=2,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompt", default="Why is the sky blue?")
    parser.add_argument("--system_prompt", default="")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Image URL or local path. Repeat to pass multiple images.",
    )
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_denoising_steps", type=int, default=48)
    parser.add_argument("--entropy_bound", type=float, default=0.1)
    parser.add_argument("--t_max", type=float, default=0.8)
    parser.add_argument("--t_min", type=float, default=0.4)
    parser.add_argument("--stability_threshold", type=int, default=1)
    parser.add_argument("--confidence_threshold", type=float, default=0.005)
    parser.add_argument("--cache_implementation", default="")
    parser.add_argument("--skip_special_tokens", action="store_true")
    parser.add_argument("--output_file", default="")
    return parser.parse_args()


if __name__ == "__main__":
    main()
