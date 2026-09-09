"""Shared backbone for DiffusionGemma checkpoint evaluation.

Provides the pieces every per-dataset eval script needs:

* model/generation helpers (moved here from ``sft_eval_bands.py`` so all eval
  scripts share one definition): ``_set_seed``, ``_gen_config``,
  ``_unwrap_clippable_linears``.
* ``load_local_model`` -- base model, optionally with a LoRA adapter attached.
  ``adapter_path=None`` => base-model-only eval (the required "no further
  adapters" setting).
* two generation backends behind a common ``generate(prompts, ...) -> list[str]``
  interface:
    - ``LocalBackend``  -- in-process HF ``model.generate()`` (the diffusion
      denoising loop), can run the base model or a LoRA-fused model.
    - ``VLLMBackend``   -- OpenAI-compatible client against a ``vllm serve``
      endpoint (see ``DiffGemma/scripts/serve_diffgemma_vllm.sh``). The diffusion
      sampler / canvas length are fixed at *serve* time; per request we only set
      ``max_tokens``/``temperature``/``seed``.
* ``run_eval`` -- the generic read -> format -> generate -> extract -> score loop,
  plus ``build_argparser`` with the flags common to every eval script.
"""

from __future__ import annotations

import argparse
import random
import re

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Generation / model helpers (shared by every eval script)
# --------------------------------------------------------------------------- #
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _gen_config(max_denoising_steps: int, max_new_tokens: int):
    """HF DiffusionGemma native entropy-bound (AR-commit) sampler defaults.

    These match the existing band/GSM8K eval (empirically better for the HF model
    than forcing the JAX DDIM params); each framework's model is best served by
    its own native sampler.
    """
    try:
        from transformers import (
            DiffusionGemmaGenerationConfig,
            EntropyBoundSamplerConfig,
        )
    except ImportError:
        from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
            DiffusionGemmaGenerationConfig,
            EntropyBoundSamplerConfig,
        )
    return DiffusionGemmaGenerationConfig(
        max_new_tokens=max_new_tokens,
        max_denoising_steps=max_denoising_steps,
        sampler_config=EntropyBoundSamplerConfig(entropy_bound=0.1),
        t_max=0.8, t_min=0.4, stability_threshold=1, confidence_threshold=0.005,
    )


def _unwrap_clippable_linears(model) -> None:
    """Replace Gemma4ClippableLinear -> inner nn.Linear (same as training, for PEFT)."""
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
    except ImportError:
        return
    for _, parent in model.named_modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, Gemma4ClippableLinear):
                setattr(parent, name, child.linear)


def load_local_model(model_path: str, adapter_path: str | None = None):
    """Load the base DiffusionGemma model and (optionally) a LoRA adapter.

    Returns ``(model, tokenizer)``. ``adapter_path=None`` -> base model only.
    Mirrors ``sft_eval_gsm8k.py`` (unwrap clippable linears only when attaching a
    LoRA adapter, so PEFT can find the inner Linears).
    """
    import os

    from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # MoE experts kernel: transformers auto-selects "grouped_mm" on torch>=2.9/SM90,
    # but torch._grouped_mm device-asserts on GH200 (crashes the first forward).
    # Default to "eager" (override via DIFFGEMMA_EXPERTS_IMPL). Mirrors sft_train.py /
    # model_utils.py so eval matches training.
    experts_impl = os.environ.get("DIFFGEMMA_EXPERTS_IMPL", "eager")
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda",
        experts_implementation=experts_impl,
    )
    if adapter_path:
        from peft import PeftModel

        _unwrap_clippable_linears(model)
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tok


# --------------------------------------------------------------------------- #
# Code extraction (shared by the MBPP / HumanEval scripts)
# --------------------------------------------------------------------------- #
_FENCED = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Return the last fenced code block; fall back to the whole text."""
    blocks = _FENCED.findall(text or "")
    if blocks:
        return blocks[-1].strip()
    return (text or "").strip()


# --------------------------------------------------------------------------- #
# Generation backends
# --------------------------------------------------------------------------- #
class LocalBackend:
    """In-process HF ``model.generate()`` diffusion decoding."""

    def __init__(self, model, tok, max_denoising_steps: int, batch_size: int = 4,
                 add_special_tokens: bool = True):
        self.model = model
        self.tok = tok
        self.max_denoising_steps = max_denoising_steps
        self.batch_size = batch_size
        # Raw templates carry no <bos> -> add it (True). apply_chat_template prompts
        # already include <bos> -> the caller sets this False to avoid a double-BOS.
        self.add_special_tokens = add_special_tokens

    def generate(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        gc = _gen_config(self.max_denoising_steps, max_new_tokens)
        out_texts: list[str] = []
        tok = self.tok
        for i in range(0, len(prompts), self.batch_size):
            chunk = prompts[i : i + self.batch_size]
            enc = tok(
                chunk, add_special_tokens=self.add_special_tokens, return_tensors="pt",
                padding="longest", padding_side="left",
            )
            inp = {k: v.to(self.model.device) for k, v in enc.items()}
            with torch.inference_mode():
                out = self.model.generate(**inp, generation_config=gc)
            seq = out.sequences if hasattr(out, "sequences") else out
            plen = inp["input_ids"].shape[-1]
            out_texts.extend(
                tok.batch_decode(seq[:, plen:], skip_special_tokens=True)
            )
        return out_texts


class VLLMBackend:
    """OpenAI-compatible client against a ``vllm serve`` DiffusionGemma endpoint.

    The diffusion sampler (entropy_bound), canvas length and denoising steps are
    set at serve time (see serve_diffgemma_vllm.sh / --hf-overrides). Here we send
    the raw ``<|turn>`` prompt to the *completions* endpoint (NOT chat completions)
    so the template matches the local backend exactly.
    """

    def __init__(self, base_url: str, model_name: str, seed: int = 0,
                 temperature: float = 0.0, add_special_tokens: bool = True,
                 api_key: str = "EMPTY"):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.seed = seed
        self.temperature = temperature
        # Raw <|turn> templates carry no <bos> -> let vLLM add it (True). Prompts
        # built via apply_chat_template already include <bos> -> set False to avoid
        # a double-BOS. Set on the backend by the caller (see sft_eval_mathbench).
        self.add_special_tokens = add_special_tokens

    def generate(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        out_texts: list[str] = []
        for p in prompts:
            resp = self.client.completions.create(
                model=self.model_name,
                prompt=p,
                max_tokens=max_new_tokens,
                temperature=self.temperature,
                seed=self.seed,
                extra_body={"add_special_tokens": self.add_special_tokens},
            )
            out_texts.append(resp.choices[0].text)
        return out_texts


def build_backend(args):
    """Construct the generation backend selected by ``--backend``."""
    if args.backend == "vllm":
        if args.max_denoising_steps is not None:
            print(
                "[warn] --max_denoising_steps is ignored with --backend vllm; the "
                "diffusion sampler is fixed at serve time (see serve_diffgemma_vllm.sh)."
            )
        return VLLMBackend(args.vllm_base_url, args.vllm_model, seed=args.seed,
                           temperature=getattr(args, "temperature", 0.0))
    model, tok = load_local_model(args.model_path, args.adapter_path)
    steps = args.max_denoising_steps if args.max_denoising_steps is not None else 64
    return LocalBackend(model, tok, max_denoising_steps=steps, batch_size=args.batch_size)


# --------------------------------------------------------------------------- #
# Generic eval driver + shared CLI
# --------------------------------------------------------------------------- #
def build_argparser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--backend", choices=["local", "vllm"], default="local")
    ap.add_argument("--model_path", default="unsloth/diffusiongemma-26B-A4B-it",
                    help="Base model (local backend).")
    ap.add_argument("--adapter_path", default=None,
                    help="LoRA checkpoint dir; omit to evaluate the BASE model only.")
    ap.add_argument("--vllm_base_url", default="http://localhost:8000/v1")
    ap.add_argument("--vllm_model", default="google/diffusiongemma-26B-A4B-it",
                    help="Served model name (or a --lora-modules name for adapter eval).")
    ap.add_argument("--max_denoising_steps", type=int, default=None,
                    help="Local backend only (default 64). Ignored for vllm.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Sampling temperature sent to the vLLM completions endpoint "
                         "(local backend ignores this; use for avg@k diversity).")
    ap.add_argument("--batch_size", type=int, default=4, help="Local backend batch size.")
    ap.add_argument("--limit", type=int, default=0, help="Eval only first N (0 = all).")
    ap.add_argument("--seed", type=int, default=0)
    # Multi-GPU sharding: launch one process per GPU, each over a strided slice of
    # the rows (shard_id = 0..num_shards-1), then aggregate the per-shard result_file.
    ap.add_argument("--num_shards", type=int, default=1,
                    help="Split the eval set into this many strided shards (one process/GPU each).")
    ap.add_argument("--shard_id", type=int, default=0, help="Which shard this process evaluates.")
    ap.add_argument("--result_file", default=None,
                    help="If set, write {label,correct,n,score} JSON here (for shard aggregation).")
    return ap


def run_eval(rows, prompt_template, score_fn, backend, args, *, label: str):
    """Generate completions and report mean score, batch by batch.

    ``rows``           list of dicts.
    ``prompt_template`` str with a ``{text}`` placeholder; ``row["_prompt"]`` is
                        substituted in (each loader sets ``_prompt``).
    ``score_fn(completion, row) -> float`` in [0, 1].

    Generation and scoring are interleaved per batch so progress (running score,
    s/example, ETA) is printed live and a partial result survives a wall-clock
    timeout.
    """
    import time

    _set_seed(args.seed)
    if args.limit:
        rows = rows[: args.limit]
    num_shards = max(1, getattr(args, "num_shards", 1))
    if num_shards > 1:
        shard_id = getattr(args, "shard_id", 0)
        rows = rows[shard_id::num_shards]
        label = f"{label}[shard {shard_id}/{num_shards}]"
    n = len(rows)
    bs = max(1, args.batch_size)

    total = 0.0
    done = 0
    t0 = time.time()
    for start in range(0, n, bs):
        chunk = rows[start : start + bs]
        # A row may carry an explicit format dict (_fmt) when the template needs more
        # than {text} (e.g. countdown's {numbers}/{target}); otherwise use {text}.
        prompts = [
            prompt_template.format(**r["_fmt"]) if "_fmt" in r else prompt_template.format(text=r["_prompt"])
            for r in chunk
        ]
        completions = backend.generate(prompts, args.max_new_tokens)
        for comp, r in zip(completions, chunk):
            total += float(score_fn(comp, r))
        done += len(chunk)
        elapsed = time.time() - t0
        rate = elapsed / done
        eta = rate * (n - done)
        print(
            f"  [{done}/{n}] running score = {total / done * 100:.1f}%  "
            f"({rate:.2f}s/ex, ETA {eta / 60:.1f}m)",
            flush=True,
        )

    src = args.adapter_path or (args.vllm_model if args.backend == "vllm" else "BASE (no LoRA)")
    print(
        f"{label} | backend={args.backend} | model={src} | seed={args.seed} | "
        f"score = {total / n * 100:.1f}%  ({total:.1f}/{n})"
    )
    result_file = getattr(args, "result_file", None)
    if result_file:
        import json as _json
        with open(result_file, "w") as fh:
            _json.dump({"label": label, "correct": total, "n": n, "score": total / n}, fh)
    return total / n
