# DiffGemma Source Guide

This directory is a local reference mirror for two official DiffusionGemma code
paths:

- `modelling/`: Hugging Face Transformers implementation for inference.
- `fine_tune/`: Google DeepMind JAX/Gemma implementation and Hackable Diffusion
  adapter for SFT/fine-tuning.

The two trees are related conceptually but are not drop-in replacements for one
another. Use `modelling/` through a recent `transformers` install. Use
`fine_tune/` through the official `gemma.diffusion...` Python package layout.

## Inference Path

Primary files:

- `modelling/configuration_diffusion_gemma.py`
  - Defines `DiffusionGemmaConfig` and `DiffusionGemmaTextConfig`.
  - Important defaults: `canvas_length=256`, Gemma4 vocabulary size `262144`,
    image token IDs, and text/vision sub-config wiring.
- `modelling/modeling_diffusion_gemma.py`
  - Defines `DiffusionGemmaForBlockDiffusion`.
  - The encoder prefills prompt context into the KV cache.
  - The decoder refines a fixed-size canvas with bidirectional attention over
    canvas tokens and cross-attention to cached context.
- `modelling/generation_diffusion_gemma.py`
  - Defines the custom `generate()` loop.
  - Outer loop: commit one canvas at a time autoregressively.
  - Inner loop: repeatedly denoise the current canvas.
  - Returns `DiffusionGemmaGenerationOutput`, including `tokens_per_forward`.

Recommended generation defaults from the code:

| Parameter | Default | Meaning |
|---|---:|---|
| `max_denoising_steps` | `48` | Maximum denoising passes per canvas. |
| `sampler_config.entropy_bound` | `0.1` | Entropy-bound token acceptance threshold. |
| `t_max` -> `t_min` | `0.8` -> `0.4` | Linear temperature schedule from early to late denoising. |
| `stability_threshold` | `1` | Stop when canvas argmax is stable across this many steps. |
| `confidence_threshold` | `0.005` | Stop when mean token entropy is sufficiently low. |
| `canvas_length` | `256` | Tokens denoised in parallel per block. |

Annotated example:

- `examples/hf_inference.py`

Run:

```bash
PROMPT="Explain block-autoregressive diffusion in one paragraph." \
  bash scripts/run_diffgemma_inference.sh
```

Multimodal:

```bash
IMAGE="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG" \
PROMPT="What animal is on the candy?" \
  bash scripts/run_diffgemma_inference.sh
```

Thinking mode:

```bash
THINKING=1 SYSTEM_PROMPT="You are concise." PROMPT="Solve: 17 * 23" \
  bash scripts/run_diffgemma_inference.sh
```

## Fine-Tune Path

Primary files:

- `fine_tune/_models.py`
  - Defines `DiffusionGemma_26B_A4B` as Gemma4 26B A4B plus diffusion
    transformer mixin and self-conditioning.
- `fine_tune/_sampler.py`
  - JAX diffusion sampler: random canvas initialization, entropy-bound
    selection, token renoising, and annealed temperature shaping.
- `fine_tune/_chat_sampler.py`
  - Diffusion-aware `Sampler` and `ChatSampler` wrappers around Gemma text
    sampling.
- `fine_tune/_early_stopping.py`
  - Early stopping primitives: token stability, entropy threshold, and chained
    stopping.
- `fine_tune/hackable_diffusion_adapter/hd/sft_model.py`
  - Main `SFTDiffusion` Kauldron model.
  - `sft_encode()` creates causal encoder KV cache over prompt plus clean
    previous canvases.
  - `sft_decode()` denoises the selected canvas with decoder attention over the
    prompt and visible canvases.
  - Training loss combines decoder diffusion loss with an encoder AR loss.
- `fine_tune/hackable_diffusion_adapter/hd/lora.py`
  - LoRA wrappers used by the LoRA configs.
- `fine_tune/hackable_diffusion_adapter/hd/gemma_checkpointer.py`
  - Converts trained parameters back into Gemma-compatible checkpoint format.
- `fine_tune/hackable_diffusion_adapter/eval_main.py`
  - Offline evaluation binary. It restores a checkpoint and injects AR
    diffusion evaluators for Sudoku or PubMedQA.

Training configs:

| Config | Task | Update Type | Key Settings |
|---|---|---|---|
| `configs/sft_sudoku.py` | Sudoku | LoRA rank 8 | 1 canvas x 256 tokens, LR `1.5e-4`, 2000 steps. |
| `configs/sft_sudoku_full.py` | Sudoku | Full weights | 1 canvas x 256 tokens, Adafactor, gradient checkpointing, 2000 steps. |
| `configs/sft_pubmedqa.py` | PubMedQA | LoRA rank 4 | 2 canvases x 128 tokens, prompt length 1024, LR `1e-4`, 2000 steps. |

Data outputs expected by configs:

- Sudoku: `sudoku_train.bagz`, `sudoku_eval.bagz`.
- PubMedQA: `pubmedqa_train.jsonl`, `pubmedqa_test.jsonl`.

Important path note:

The fine-tune code imports modules as `gemma.diffusion...`, and the configs
refer to data paths like `gemma/diffusion/hackable_diffusion_adapter/data/...`.
That means the scripts should be run from the parent directory of an official
`gemma/` source checkout or an environment where the same package layout is
installed. The local `DiffGemma/fine_tune/` folder is useful for reading and
patching, but by itself it is not a complete `gemma` package root.

## Fine-Tune Commands

Prepare data in an official Gemma checkout:

```bash
GEMMA_PARENT=/path/to/parent/of/gemma TASK=sudoku ACTION=prepare-data \
  bash scripts/run_diffgemma_finetune.sh
```

Train Sudoku LoRA:

```bash
GEMMA_PARENT=/path/to/parent/of/gemma TASK=sudoku SFT_MODE=lora \
WORKDIR=/path/to/xp_dir_sudoku_lora \
  bash scripts/run_diffgemma_finetune.sh
```

Train Sudoku full weights:

```bash
GEMMA_PARENT=/path/to/parent/of/gemma TASK=sudoku SFT_MODE=full \
WORKDIR=/path/to/xp_dir_sudoku_full \
  bash scripts/run_diffgemma_finetune.sh
```

Train PubMedQA LoRA:

```bash
GEMMA_PARENT=/path/to/parent/of/gemma TASK=pubmedqa \
WORKDIR=/path/to/xp_dir_pubmedqa_lora \
  bash scripts/run_diffgemma_finetune.sh
```

Evaluate a saved checkpoint:

```bash
GEMMA_PARENT=/path/to/parent/of/gemma ACTION=eval TASK=sudoku \
WORKDIR=/path/to/xp_dir_sudoku_lora STEP=1000 EVAL_NAMES=sample_ar_steps64 \
  bash scripts/run_diffgemma_finetune.sh
```

## Practical Checks

- HF inference requires a Transformers version that exposes
  `DiffusionGemmaForBlockDiffusion`.
- Fine-tuning requires Python 3.12, CUDA/JAX compatibility, `gemma`,
  `kauldron`, `hackable_diffusion`, and task data preparation.
- Official README recommends CUDA 13 for the JAX fine-tune path and warns
  against mixing CUDA 12 JAX/NCCL packages.
- For quick smoke tests, override config fields from the CLI, for example:

```bash
MAX_STEPS=10 EVAL_NUM_BATCHES=1 TASK=sudoku WORKDIR=/tmp/diffgemma_smoke \
  bash scripts/run_diffgemma_finetune.sh
```

