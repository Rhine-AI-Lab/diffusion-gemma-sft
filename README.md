# Fine-Tuning DiffusionGemma with TRL

PyTorch/TRL code for supervised fine-tuning and evaluating
[DiffusionGemma-26B-A4B](https://huggingface.co/google/diffusiongemma-26B-A4B-it),
the open-weight uniform diffusion language model.

This repository accompanies the blog post
[Fine-Tuning DiffusionGemma: What Works, What Breaks](https://www.andreamiele.fr/blog/fine-tuning-diffusiongemma).
It contains the training pipeline, data builders, evaluation harnesses, and
experiment notes behind the Sudoku, tool-calling, mathematical-reasoning, and
terminal-agent results.

The codebase is based on
[GDSD](https://arxiv.org/abs/2605.29398) and retains its reinforcement-learning
infrastructure. The main addition is a TRL/Transformers path adapted to
DiffusionGemma's uniform-state, block-diffusion architecture.

## Highlights

- Four denoising SFT objectives: `base-sft`, `reweighted-ce`, `loo-ce`, and
  `reweighted-loo-ce`.
- Native DiffusionGemma canvas corruption, self-conditioning, and optional
  encoder autoregressive loss.
- LoRA training through PEFT, with optional Unsloth loading.
- Single-canvas and block-structured training for short and long completions.
- Local Hugging Face and OpenAI-compatible vLLM evaluation backends.
- Reproduction utilities for Sudoku, API-Bank/BFCL, math, code, Countdown, and
  TBLite/Terminal-Bench-style experiments.

## SFT objectives

For a clean completion `x₀`, corrupted canvas `xₜ`, and completion mask `M`, the
trainer minimizes a masked denoising cross-entropy. The variants differ in
their diffusion-time weight and whether the logits receive the
leave-one-out-to-denoiser correction.

| Objective | Time weight | Logits scored by cross-entropy |
| --- | --- | --- |
| `base-sft` | `1` | raw denoiser logits |
| `reweighted-ce` | `1 / (1 - t)` | raw denoiser logits |
| `loo-ce` | `1` | leave-one-out corrected logits |
| `reweighted-loo-ce` | `1 / (1 - t)` | leave-one-out corrected logits |

See [the SFT guide](DiffGemma/diffgemma_trl/SFT_README.md) for the objective,
data layout, and parity notes with the original JAX implementation.

## Repository layout

| Path | Purpose |
| --- | --- |
| `DiffGemma/diffgemma_trl/` | TRL/Transformers training, objectives, collators, rewards, and evaluation |
| `DiffGemma/scripts/` | SFT, evaluation, inference, vLLM, and RL launchers |
| `dataset/` | Small committed datasets plus builders for larger training sets |
| `codefix_eval/` | API-Bank, BFCL, code-fix, and structured-generation experiments |
| `tblite_eval/` | Terminal-agent data conversion, training, and Harbor evaluation |
| `local/` and `cscs/` | Local and CSCS experiment launchers |
| `gdsd/` | Original GDSD-derived RL path |
| `eval_out/RESULTS.md` | Curated experiment results |

## Quick start

The model is large: use a recent CUDA GPU with roughly 50 GB or more memory for
LoRA experiments, and start with batch size 1. Python 3.12 is the best-tested
environment for this code.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_sft.txt
```

The base checkpoint may require accepting its Hugging Face license and logging
in before download.

Run a short Sudoku smoke test:

```bash
DATASET=./dataset/sudoku_sft.jsonl \
MODEL=google/diffusiongemma-26B-A4B-it \
VARIANT=loo-ce \
BATCH=1 GRAD_ACCUM=4 MAX_STEPS=10 \
bash DiffGemma/scripts/run_diffgemma_sft_trl.sh
```

For a full run, increase `MAX_STEPS` and point `DATASET` at the generated
training set. Set `VARIANT` to any objective in the table above. To opt into
Unsloth, install it separately and set `USE_UNSLOTH=true`.

Training data is JSONL with one prompt/completion pair per line:

```json
{"prompt": "task input", "completion": "target response"}
```

## Evaluation

Evaluate a Sudoku adapter across the easy, medium, and hard bands:

```bash
PYTHONPATH="$(pwd)" python -m DiffGemma.diffgemma_trl.sft_eval_bands \
  --model_path google/diffusiongemma-26B-A4B-it \
  --adapter_path ./xp_sft_loo_ce_trl/checkpoint-4000 \
  --bands_dir ./dataset \
  --max_denoising_steps 64
```

Run the shared math, Countdown, MBPP, and HumanEval evaluation wrapper:

```bash
ADAPTER_PATH=./xp_sft_loo_ce_trl/checkpoint-4000 \
  bash DiffGemma/scripts/run_diffgemma_eval.sh
```

The wrapper also supports an OpenAI-compatible vLLM endpoint via
`BACKEND=vllm`. Task-specific setup and interpretation live in
[the math guide](MATH_SFT_README.md),
[the terminal-agent guide](tblite_eval/README.md), and
[the experiment summary](SFT_EXPERIMENTS_SUMMARY.md).

## Findings represented by this release

The experiments show that SFT is effective for short, structured outputs such
as Sudoku grids and tool calls, with `loo-ce` strongest overall. Long
chain-of-thought and multi-turn terminal-agent SFT are much more fragile:
mathematical runs lose reliable stopping behavior, while agent runs accumulate
state and action errors. Read the blog post for the full results and analysis.

## Acknowledgements

This project builds on GDSD, Hugging Face Transformers and TRL, PEFT, Unsloth,
and the public DiffusionGemma implementation. The retained GDSD code also
incorporates ideas and utilities from ESPO and SPG.

## Citation

```bibtex
@online{miele2026diffusiongemma,
  title  = {Fine-Tuning DiffusionGemma: What Works, What Breaks},
  author = {Miele, Andrea and Bankes, William and Jiang, Keyue and
            Son, Seongho and Tang, Xiaohang and Bogunovic, Ilija},
  year   = {2026},
  url    = {https://www.andreamiele.fr/blog/fine-tuning-diffusiongemma}
}
```

Released under the [Apache 2.0 License](LICENSE).
