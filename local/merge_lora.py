"""Merge a DiffusionGemma LoRA checkpoint into the base weights -> a full model dir
that vLLM can serve (the diffusion vLLM fork does NOT support LoRA adapters directly).

  python local/merge_lora.py <base> <adapter_ckpt_dir> <out_dir>
"""
import os
import shutil
import sys

from DiffGemma.diffgemma_trl.sft_eval_common import load_local_model

base, adapter, out = sys.argv[1], sys.argv[2], sys.argv[3]
print(f"[merge] base={base}\n[merge] adapter={adapter}\n[merge] out={out}", flush=True)

model, tok = load_local_model(base, adapter)      # base (cuda, eager MoE) + PEFT adapter
model = model.merge_and_unload()                   # fold LoRA into the base Linears
os.makedirs(out, exist_ok=True)
model.save_pretrained(out, safe_serialization=True)
tok.save_pretrained(out)

# copy aux files vLLM / the tokenizer template need but save_pretrained may not emit
def _base_dir(b):
    if os.path.isdir(b):
        return b
    from huggingface_hub import snapshot_download
    return snapshot_download(b)  # offline -> resolves to the local cache snapshot

bdir = _base_dir(base)
for f in ("chat_template.jinja", "processor_config.json", "generation_config.json"):
    src = os.path.join(bdir, f)
    dst = os.path.join(out, f)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy(src, dst)
        print(f"[merge] copied {f}", flush=True)
print(f"[merge] DONE -> {out}", flush=True)
