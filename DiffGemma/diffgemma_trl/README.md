# DiffusionGemma TRL/Unsloth Training Port

This package is a DiffGemma-local port of the useful parts of the root-level
`gdsd` trainer. It keeps the TRL/GRPO training shell, reward functions,
advantage computation, buffering, checkpointing, and logging, but replaces the
LLaDA-style masked-token denoising path with DiffusionGemma canvas denoising.

## Review

The original `gdsd` trainer assumes this interface:

1. Concatenate `prompt_ids + completion_ids`.
2. Replace some completion tokens with a `mask_token_id`.
3. Run the model on the whole sequence.
4. Compute CE/logprob only on masked completion positions.

DiffusionGemma does not use that interface. Its training/inference path is:

1. Encode `prompt_ids` into a read-only KV cache.
2. Feed a fixed-size noisy `canvas` to the diffusion decoder.
3. Compute CE/logprob against the clean canvas.

So the migration is feasible, but the denoise-logprob estimator must be
DiffusionGemma-specific.

## Contents

- `trainer.py`
  - `DiffusionGemmaGDSDTrainer`: keeps the GDSD square objective.
  - `DiffusionGemmaGRPOTrainer`: adds a clipped GRPO/PPO-style objective.
  - Both trainers estimate policy logprob from
    `prompt_ids + noisy canvas -> clean canvas`.
- `model_utils.py`
  - Loads DiffusionGemma through Unsloth `FastModel` by default.
  - Adds Unsloth LoRA adapters before passing the model to TRL.
- `data_utils.py` and `rewards.py`
  - Local copies of the common dataset/reward utilities so this path does not
    import the root-level `gdsd` package.
- `train.py`
  - CLI entry point using `TrlParser`.

## Environment Setup Notes

Use a fresh environment for this path. DiffusionGemma support depends on recent
`transformers` code, and Unsloth may patch model classes at import time, so avoid
mixing this with older LLaDA/GDSD environments.

Recommended setup order:

```bash
python -m venv .venv-diffgemma
source .venv-diffgemma/bin/activate
python -m pip install --upgrade pip

# Install a PyTorch build that matches your CUDA driver first.
# Pick the official torch command for your CUDA version / cluster image.

pip install sentencepiece protobuf datasets accelerate peft trl bitsandbytes triton hf_transfer
pip install unsloth
pip install --no-deps --upgrade --force-reinstall \
  git+https://github.com/unslothai/unsloth-zoo.git \
  git+https://github.com/unslothai/unsloth.git

# The public DiffusionGemma Sudoku notebook used a Transformers build that
# includes DiffusionGemma classes.
pip install --no-deps transformers==5.11.0 "tokenizers>=0.22.0,<=0.23.0"
```

Hardware notes:

- Use bf16 on a large GPU. The 26B-A4B checkpoint still needs roughly 50GB+
  just to keep the full MoE weights resident.
- `load_in_4bit=False` is the default here because the fused MoE expert tensors
  are not meaningfully reduced by the usual 4-bit path.
- An A100 80GB, H100, B200, or similar large-memory GPU is the realistic target.
  A 40GB GPU may offload weights and become impractically slow.
- Keep `per_device_train_batch_size=1`, `generation_batch_size=1`, and
  `diffusion_num_mc=1` for the first smoke test.

Runtime checks before training:

```bash
python - <<'PY'
import torch
import transformers
import trl
import unsloth

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("trl", trl.__version__)
PY
```

If `torch.cuda.is_available()` is false, fix the PyTorch/CUDA environment before
debugging trainer code.

## Run

From the repository root:

```bash
bash DiffGemma/scripts/run_diffgemma_grpo.sh \
  --model_name_or_path unsloth/diffusiongemma-26B-A4B-it \
  --dataset_name sudoku \
  --dataset_path /path/to/4x4_sudoku_unique_puzzles.csv \
  --output_dir DiffGemma/outputs/diffgemma-gdsd-sudoku \
  --rl_loss_type gdsd \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --num_generations 4 \
  --num_iterations 1 \
  --max_prompt_length 1024 \
  --max_completion_length 256 \
  --diffusion_canvas_length 256 \
  --diffusion_steps 48 \
  --diffusion_num_mc 1 \
  --learning_rate 1e-4 \
  --bf16 true \
  --logging_steps 1
```

For the GRPO objective, change:

```bash
--rl_loss_type grpo
```

## Current Scope

This first port intentionally trains one DiffusionGemma canvas at a time. Keep
`max_completion_length <= diffusion_canvas_length` for the first smoke tests.
Multi-canvas training should follow the official JAX SFT logic: cache the prompt
plus previous clean canvases, then denoise the selected current canvas.

The local `code` reward only includes format checking right now. The execution
sandbox from the root-level `gdsd` tree has not been copied into `DiffGemma/`.

This code path requires a runtime with `torch`, `trl`, `transformers`, and
usually `unsloth`. The repository environment used to write this scaffold did
not include `torch`, so only syntax-level checks were run locally.
