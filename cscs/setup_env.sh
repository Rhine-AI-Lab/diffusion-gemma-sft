#!/bin/bash
# Set up the gdsdv2 SFT (PyTorch/TRL) Python env INSIDE the container, as a venv
# on $SCRATCH (writable; the squashfs image itself is read-only at runtime).
#
# Run it through the container environment on a compute node:
#   srun -A a0112 -p debug -t 00:30:00 \
#     --environment=$SCRATCH/projects/gdsdv2/cscs/gdsdv2-pt.toml \
#     bash cscs/setup_env.sh
#
# --system-site-packages lets the venv reuse the container's GPU-tested arm64
# PyTorch instead of pip pulling a (CPU/arm64-broken) wheel over the top. We do
# NOT install torch/torchvision here for exactly that reason.
set -euo pipefail

REPO="${SCRATCH}/projects/gdsdv2"
VENV="${REPO}/.venv"

cd "$REPO"
echo "[env] python: $(python3 --version)  host: $(hostname)"
echo "[env] baseline torch from container:"
python3 -c "import torch; print('  torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())" || true

if [ ! -d "$VENV" ]; then
  echo "[env] creating venv (--system-site-packages) at $VENV"
  python3 -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python3 -m pip install --upgrade pip

# The SFT path needs a newer HF stack than the repo's pinned requirements.txt:
#   transformers>=5.11 ships diffusion_gemma + gemma4 natively (the bundled
#     DiffGemma/diffusion_gemma uses transformers-internal relative imports and
#     only resolves via the installed package);
#   peft>=0.19 (0.17 imports HybridCache, removed in transformers 5.x);
#   trl is imported at module load by model_utils/sft_configs.
# torch/torchvision are deliberately omitted (kept from the container).
python3 -m pip install \
  "transformers==5.12.1" \
  "peft>=0.19" \
  trl \
  "accelerate>=1.4.0" \
  "datasets>=3.2.0" \
  sentencepiece \
  safetensors \
  einops \
  bitsandbytes \
  "huggingface-hub[cli,hf_xet]"

# Some trl releases pin a different transformers; re-pin 5.12.1 (the version that
# ships diffusion_gemma) while freezing the container's torch so pip can't swap
# the GPU arm64 build for a generic CPU wheel.
TORCHVER="$(python3 -c 'import importlib.metadata as m; print(m.version("torch"))')"
CONS="$(mktemp)"
printf 'torch==%s\n' "$TORCHVER" > "$CONS"
python3 -m pip install -c "$CONS" "transformers==5.12.1"
rm -f "$CONS"

# The NGC image ships torchao 0.11, but peft>=0.19's is_torchao_available() RAISES
# (rather than returning False) when torchao < 0.16 — breaking get_peft_model() for
# any LoRA run. Shadow it with a >=0.16 build in the venv. --no-deps keeps the
# container's torch; peft only reads torchao's version metadata for nn.Linear
# targets (the cpp kernels stay unloaded, which is fine).
python3 -m pip install -U --no-deps "torchao>=0.16.0"

echo "[env] verifying import graph + GPU visibility:"
python3 -c "
import torch, transformers, trl, peft, datasets
from transformers import DiffusionGemmaForBlockDiffusion  # native diffusion_gemma
from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
print('  torch', torch.__version__, 'cuda_avail', torch.cuda.is_available(),
      'device', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'))
print('  transformers', transformers.__version__, 'trl', trl.__version__, 'peft', peft.__version__)
print('  import graph OK (DiffusionGemmaForBlockDiffusion + Gemma4ClippableLinear)')
"
echo "[env] done. Activate with: source $VENV/bin/activate"
