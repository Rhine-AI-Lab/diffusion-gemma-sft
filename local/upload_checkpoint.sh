#!/bin/bash
# Publish a DiffusionGemma tmax block-SFT LoRA checkpoint to the Hub.
#
# Stages only the adapter + tokenizer (never optimizer.pt / rng_state.pth / trainer_state.json),
# writes a model card, and uploads in a single commit.
#
#   hf auth login                                    # once; needs a WRITE token
#   bash local/upload_checkpoint.sh out/tmax_sft/checkpoint-500 Rhine-AI/diffgemma-tmax-sft-500
#   DRY_RUN=1 bash local/upload_checkpoint.sh ...    # stage + print the card, don't push
#
# NOTE env var appends a caveat section to the model card.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

CKPT="${1:?usage: upload_checkpoint.sh <checkpoint_dir> <repo_id>}"
REPO_ID="${2:?usage: upload_checkpoint.sh <checkpoint_dir> <repo_id>}"
PRIVATE="${PRIVATE:-1}"
BASE="${BASE:-google/diffusiongemma-26B-A4B-it}"
[[ -f "$CKPT/adapter_model.safetensors" ]] || { echo "no adapter in $CKPT" >&2; exit 1; }

STEP="$(basename "$CKPT" | sed 's/checkpoint-//')"
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
for f in adapter_config.json adapter_model.safetensors tokenizer.json tokenizer_config.json chat_template.jinja; do
  [[ -f "$CKPT/$f" ]] && cp "$CKPT/$f" "$STAGE/"
done

LOSS="$(python - "$CKPT" <<'PY' 2>/dev/null || echo "n/a"
import json,sys
h=[e for e in json.load(open(sys.argv[1]+"/trainer_state.json"))["log_history"] if "loss" in e]
print(f"{h[-1]['loss']:.4f}" if h else "n/a")
PY
)"

cat > "$STAGE/README.md" <<EOF
---
base_model: ${BASE}
library_name: peft
tags: [diffusion-lm, block-diffusion, lora, terminal-agent, tmax]
---

# DiffusionGemma tmax terminal-agent block-SFT — step ${STEP}

LoRA adapter for \`${BASE}\`, supervised-finetuned on Allen AI's
[tmax](https://huggingface.co/datasets/allenai/tmax-sft) terminal-agent trajectories
(config \`skill_tax_20260505_2.2k_combined_balanced_thinking_only_success\`).

## Training objective — response-anchored block diffusion

Each trajectory is trimmed to <4096 tokens, always ending on an assistant turn. Per step we
sample **one assistant response**, tile it into 256-token canvases, and sample **one canvas**:

- **Denoiser loss** on the sampled canvas (the diffusion target).
- **Encoder AR loss** over the *entire* prefix preceding that canvas (system prompt, user task,
  every prior assistant turn and tool output).
- Everything after the canvas is dropped.

| hyperparameter | value |
| --- | --- |
| LoRA rank / alpha | 64 / 128 |
| target modules | q,k,v,o,gate,up,down |
| canvas length | 256 |
| max completion / prompt | 4096 |
| learning rate | 1.5e-5, cosine, 100 warmup |
| batch (per-device x accum) | 1 x 8 |
| step | ${STEP} |
| last logged train loss | ${LOSS} |

## Usage

\`\`\`python
from peft import PeftModel
from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion

base = DiffusionGemmaForBlockDiffusion.from_pretrained("${BASE}", torch_dtype="bfloat16", device_map="cuda")
model = PeftModel.from_pretrained(base, "${REPO_ID}")
tok = AutoTokenizer.from_pretrained("${REPO_ID}")
\`\`\`

Prompts are rendered with the **tmax/Qwen** chat template (not DiffGemma's, which drops the
assistant reasoning) and tokenized with the DiffGemma tokenizer. \`<|im_end|>\` is ordinary text to
this tokenizer, not an EOS token, so generation does not stop at a turn boundary — use a stop-string.
EOF

if [[ -n "${NOTE:-}" ]]; then printf '\n## Caveats\n\n%s\n' "$NOTE" >> "$STAGE/README.md"; fi

echo "==== staged for ${REPO_ID} (step ${STEP}, loss ${LOSS}) ===="
ls -la "$STAGE" | sed 's/^/  /'
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "---- model card ----"; cat "$STAGE/README.md"; echo "---- DRY_RUN: not uploading ----"; exit 0
fi
PRIV_FLAG=(--private); [[ "$PRIVATE" == "0" ]] && PRIV_FLAG=(--no-private)
"$REPO/.venv/bin/hf" upload "$REPO_ID" "$STAGE" . --repo-type model "${PRIV_FLAG[@]}" \
  --commit-message "tmax terminal-agent block-SFT, step ${STEP}"
echo "==== https://huggingface.co/${REPO_ID} ===="
