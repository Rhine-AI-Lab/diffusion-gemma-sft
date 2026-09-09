"""Model and processor loading utilities for DiffusionGemma TRL training."""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from .configs import DiffusionGemmaGRPOConfig


logger = logging.getLogger(__name__)


def _warmup_fuse_path(path: str, max_retries: int = 3, delay: float = 2.0):
    """Warm up an OSS FUSE mount path by listing parent directories.

    Some FUSE implementations require accessing parent directories before
    child directories become visible (lazy mount behavior).
    """
    parts = Path(path).parts  # e.g. ('/', 'data', 'oss_bucket_0', 'models', ...)
    # Walk through parent paths to trigger FUSE directory caching
    for i in range(2, len(parts) + 1):
        subpath = os.path.join(*parts[:i])
        try:
            entries = os.listdir(subpath)
            logger.info("[FUSE warmup] ls %s -> %d entries", subpath, len(entries))
        except (FileNotFoundError, OSError) as e:
            logger.warning("[FUSE warmup] ls %s FAILED: %s", subpath, e)

    # Now retry the full path
    for attempt in range(max_retries):
        try:
            entries = os.listdir(path)
            logger.info("[FUSE warmup] Model dir accessible: %s (%d entries)", path, len(entries))
            return True
        except (FileNotFoundError, OSError) as e:
            logger.warning(
                "[FUSE warmup] Attempt %d/%d - model dir not accessible: %s (error: %s)",
                attempt + 1, max_retries, path, e,
            )
            if attempt < max_retries - 1:
                time.sleep(delay)
    return False


@contextlib.contextmanager
def _patch_fs_for_fuse(path_prefix: str):
    """Monkey-patch os.path.isdir and os.listdir for OSS FUSE mount paths.

    OSS FUSE does not reliably support stat() on directories, and os.listdir()
    may fail with FileNotFoundError even when the directory exists. This patches
    both functions to handle FUSE-specific quirks.
    """
    _orig_isdir = os.path.isdir
    _orig_listdir = os.listdir

    def _patched_isdir(p):
        if isinstance(p, str) and p.startswith(path_prefix):
            return True
        return _orig_isdir(p)

    def _patched_listdir(p):
        try:
            return _orig_listdir(p)
        except (FileNotFoundError, OSError):
            if not (isinstance(p, str) and p.startswith(path_prefix)):
                raise
            # Fallback: try using subprocess ls for FUSE paths
            logger.warning("[FUSE patch] os.listdir(%s) failed, trying subprocess ls", p)
            try:
                result = subprocess.run(
                    ["ls", "-1", p],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0 and result.stdout.strip():
                    entries = result.stdout.strip().split("\n")
                    logger.info("[FUSE patch] subprocess ls succeeded: %d entries", len(entries))
                    return entries
            except Exception as ls_err:
                logger.warning("[FUSE patch] subprocess ls also failed: %s", ls_err)
            # Last resort: return empty list so find_adapter_config_file
            # won't crash (it just won't find any adapter config).
            logger.warning("[FUSE patch] Returning empty list for %s", p)
            return []

    os.path.isdir = _patched_isdir
    os.listdir = _patched_listdir
    try:
        yield
    finally:
        os.path.isdir = _orig_isdir
        os.listdir = _orig_listdir


def _resolve_torch_dtype(model_args: ModelConfig):
    dtype = getattr(model_args, "torch_dtype", None)
    if dtype in (None, "auto"):
        return torch.bfloat16
    if isinstance(dtype, str):
        return getattr(torch, dtype)
    return dtype


def _token(model_args: ModelConfig) -> str | None:
    return getattr(model_args, "token", None) or getattr(model_args, "use_auth_token", None)


def get_processing_tokenizer(processing_class: Any):
    """Return the tokenizer object from either a processor or a tokenizer."""

    return getattr(processing_class, "tokenizer", processing_class)


def load_model_and_processor(model_args: ModelConfig, training_args: DiffusionGemmaGRPOConfig):
    """Load DiffusionGemma and its processor/tokenizer.

    The recommended path is Unsloth because the public Sudoku notebook uses
    ``FastModel`` for DiffusionGemma LoRA. A Transformers fallback is kept for
    environments where DiffusionGemma is already available in ``transformers``.
    """

    model_name = model_args.model_name_or_path
    torch_dtype = _resolve_torch_dtype(model_args)

    if training_args.use_unsloth:
        try:
            from unsloth import FastModel
        except ImportError as exc:
            raise ImportError(
                "use_unsloth=True but unsloth is not installed. Install unsloth or pass --use_unsloth false."
            ) from exc

        logger.info("Loading DiffusionGemma with Unsloth FastModel: %s", model_name)
        model, processor = FastModel.from_pretrained(
            model_name=model_name,
            dtype=torch_dtype,
            load_in_4bit=getattr(model_args, "load_in_4bit", False),
            token=_token(model_args),
        )

        if training_args.unsloth_lora_rank and training_args.unsloth_lora_rank > 0:
            logger.info(
                "Adding Unsloth LoRA adapters: r=%s alpha=%s",
                training_args.unsloth_lora_rank,
                training_args.unsloth_lora_alpha,
            )
            model = FastModel.get_peft_model(
                model,
                r=training_args.unsloth_lora_rank,
                lora_alpha=training_args.unsloth_lora_alpha,
                use_gradient_checkpointing=training_args.unsloth_gradient_checkpointing,
            )
        return model, processor

    logger.info("Loading DiffusionGemma with Transformers fallback: %s", model_name)
    from transformers import AutoProcessor, AutoTokenizer
    from transformers import DiffusionGemmaForBlockDiffusion

    # If model_name looks like a local path (starts with / or ./), patch filesystem
    # functions so transformers recognizes it as a local directory.
    # OSS FUSE mounts do not reliably support stat()/listdir() on directories.
    _is_local = model_name.startswith(("/", "./", "../"))

    if _is_local:
        # Warm up the FUSE mount path (trigger directory caching)
        logger.info("[FUSE] Warming up mount path: %s", model_name)
        _warmup_fuse_path(model_name)

    _fuse_ctx = _patch_fs_for_fuse(model_name) if _is_local else contextlib.nullcontext()

    quantization_config = get_quantization_config(model_args)
    # MoE experts kernel: transformers auto-selects "grouped_mm" on torch>=2.9/SM90,
    # but torch._grouped_mm device-asserts on the unaligned per-expert token counts
    # produced here (crashes the first forward on GH200). Default to "eager" (memory
    # -safe, slower); override via DIFFGEMMA_EXPERTS_IMPL. Mirrors sft_train.py.
    experts_impl = os.environ.get("DIFFGEMMA_EXPERTS_IMPL", "eager")
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=getattr(model_args, "trust_remote_code", False),
        attn_implementation=model_args.attn_implementation,
        experts_implementation=experts_impl,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    if _is_local:
        model_kwargs["local_files_only"] = True

    with _fuse_ctx:
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(model_name, **model_kwargs)
        try:
            processor = AutoProcessor.from_pretrained(
                model_name,
                revision=model_args.model_revision,
                trust_remote_code=getattr(model_args, "trust_remote_code", False),
                **(dict(local_files_only=True) if _is_local else {}),
            )
        except Exception:
            processor = AutoTokenizer.from_pretrained(
                model_name,
                revision=model_args.model_revision,
                trust_remote_code=getattr(model_args, "trust_remote_code", False),
                **(dict(local_files_only=True) if _is_local else {}),
            )
    return model, processor
