"""Merge a DiffusionGemma LoRA adapter into its base model for vLLM serving.

vLLM's DiffusionGemmaForConditionalGeneration does not implement SupportsLoRA,
so adapters must be folded into the base weights offline rather than served
via --enable-lora. Per DiffGemma/diffgemma_trl/diag_merge.py's finding, merging
in bf16 accumulates additive rounding error large enough to corrupt the policy
(full-solve accuracy dropped from ~98% to ~26% in that experiment); merging in
fp32 and only casting to bf16 afterward avoids this.

Runs entirely on CPU (this box has ~1TB RAM) to keep GPUs free for serving.
"""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion

from DiffGemma.diffgemma_trl.sft_eval_bands import _unwrap_clippable_linears


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="google/diffusiongemma-26B-A4B-it")
    ap.add_argument("--adapter_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--keep_all_adapters", action="store_true",
                    help="Merge encoder AND decoder LoRA (default: decoder-only, "
                         "avoids the weight-tying double-merge corruption).")
    args = ap.parse_args()

    print(f"Loading base model {args.base_model} in fp32 on CPU...")
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.base_model, torch_dtype=torch.float32, device_map="cpu"
    )
    _unwrap_clippable_linears(model)

    print(f"Attaching adapter {args.adapter_path}...")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    # CRITICAL: DiffusionGemma's encoder and decoder are weight-TIED (the same
    # weight tensors are shared, confirmed via data_ptr: 627/631 decoder params
    # alias an encoder param). But LoRA creates SEPARATE adapters for the
    # encoder (`model.encoder.language_model.*`, `model.encoder.vision_tower.*`)
    # and decoder (`model.decoder.*`) module objects. Because they alias the
    # same base tensor, merge_and_unload() would add BOTH deltas to it
    # (W + Δ_dec + Δ_enc), corrupting the served weights -- generation only
    # wants the decoder's Δ. So zero out every NON-decoder LoRA delta before
    # merging, leaving a clean W + Δ_dec. (Fix for the double-merge bug.)
    if not args.keep_all_adapters:
        import torch.nn as nn
        zeroed = 0
        for name, mod in model.named_modules():
            if hasattr(mod, "lora_B") and ".decoder." not in name:
                for adapter in list(mod.lora_B.keys()):
                    nn.init.zeros_(mod.lora_B[adapter].weight)
                    zeroed += 1
        print(f"Zeroed {zeroed} non-decoder LoRA adapters (weight-tying fix).")

    print("Merging (fp32)...")
    model = model.merge_and_unload()
    _unwrap_clippable_linears(model)

    print("Casting to bf16...")
    model = model.to(torch.bfloat16)

    print(f"Saving merged model to {args.output_dir}...")
    model.save_pretrained(args.output_dir, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(args.adapter_path)
    tok.save_pretrained(args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
