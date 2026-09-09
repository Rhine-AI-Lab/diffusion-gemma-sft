# DiffusionGemma SFT (TRL / transformers)

Supervised fine-tuning of **DiffusionGemma-26B-A4B** with the uniform-state
diffusion objective, implemented on top of `transformers.Trainer` (PyTorch),
as a PyTorch port of the JAX/kauldron `hackable_diffusion_adapter` SFT. This is
the supervised counterpart to the RL trainers in `trainer.py`; it shares the
same HF model and canvas-denoising machinery.

DiffusionGemma is **not** autoregressive: it denoises a fixed-size *canvas* of
response tokens conditioned on the prompt. So SFT is a **masked denoising
cross-entropy** on the completion tokens, not next-token CE — `trl.SFTTrainer`'s
loss does not apply; we subclass `transformers.Trainer` and override
`compute_loss`.

## Objective

One sampled diffusion time `t` per sequence; corrupt the whole completion with
the uniform-state forward process; predict the clean tokens. With completion
mask `M`, RF schedule `α(t)=1−t`, vocab `K`, network logits `f_θ`:

```
L_SFT = - E_{(c,y), t~U(ε,1-ε), x_t~q(·|x0)} [ w(t) · Σ_l M_l · log <x0_l, p_θ_l(x_t,t,c)> ]
```

Four variants (`--sft_variant`), differing only in the time-weight `w(t)` and
the scored distribution `p_θ` (see `sft_loss.py`):

| variant | `w(t)` | `p_θ` |
|---|---|---|
| `base-sft` | 1 | `softmax(f_θ)` |
| `reweighted-ce` | `-α̇/α = 1/(1-t)` | `softmax(f_θ)` |
| `loo-ce` | 1 | `softmax(f_θ + log(1+ Kα/(1-α))·x_t)` (LOO→denoiser shift) |
| `reweighted-loo-ce` | `-α̇/α = 1/(1-t)` | `softmax(f_θ + log(1+ Kα/(1-α))·x_t)` |

Notes:
- `reweighted-ce` uses `-α̇/α` with **no `1/K`** (the `1/K` only belongs to the
  exact ELBO, where it cancels). Its weight explodes near `t=1`; bound it with
  `--diffusion_weight_clip` (e.g. 100).
- `loo-ce` is `base-sft` + the leave-one-out logit shift (Gourevitch et al.,
  arXiv:2605.22765).
- Total loss = `decoder_loss_weight · L_SFT + encoder_loss_weight · L_enc`,
  where `L_enc` is a next-token CE over `[c;y]` on the completion span using the
  encoder hidden states + the (softcapped) LM head.

Other parity features (mirroring the JAX SFT): **self-conditioning** (a second
forward pass fed the detached first-pass logits, gated per-example via the
model's `self_conditioning_mask`), the **uniform corruption** with a
configurable diffusion-time range, and the diffuGemma **prompt template**.

## Files

| File | Role |
|---|---|
| `sft_loss.py` | The four variant losses (weight, LOO shift, optional weight clip). Pure, unit-tested. |
| `sft_trainer.py` | `DiffusionGemmaSFTTrainer(Trainer)`: corruption → self-conditioning → forward → `sft_loss` + encoder AR loss. |
| `sft_configs.py` | `DiffusionGemmaSFTConfig` (variant, schedule, t-range, self-cond, loss weights, weight clip) + script args. |
| `sft_data_utils.py` | Dataset loader + collator (prompt/completion → canvas layout; exact diffuGemma prompt). |
| `sft_train.py` | Entry point (model load via native/bundled `DiffusionGemmaForBlockDiffusion`, PEFT, router freeze, train). |
| `sft_eval_bands.py` | Difficulty-band eval (load adapter, generate, score full-solve + cell-acc). |
| `tests/test_sft_loss.py` | CPU unit tests for the loss variants (no model needed). |
| `../scripts/run_diffgemma_sft_trl.sh` | Convenience launcher. |

## Environment

The root `requirements.txt` includes the broader GDSD/RL stack. For an SFT-only
environment, install the smaller tested set:

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements_sft.txt
```
(`transformers>=5.11` ships `DiffusionGemmaForBlockDiffusion`; `peft>=0.19` is
needed — `peft 0.17` imports `HybridCache`, removed in transformers 5.x.)

## Data

Both train and eval are **JSONL** with two fields per line:

```json
{"prompt": "<space-separated 81-digit puzzle, 0 = blank>", "completion": "<space-separated 81-digit solution>"}
```
(the eval band files use `{"puzzle": ..., "solution": ...}`.)

The collator wraps `prompt` in the exact diffuGemma instruction template
(`DIFFUGEMMA_SUDOKU_PROMPT`) and tokenizes it as a literal string with `add_bos`
(no chat template) — matching the JAX SFT byte-for-byte.

### Training set — `dataset/sudoku_train_full.jsonl` (NOT committed, ~2.7 GB)

The exact data the JAX SFT uses: the Kaggle `rohanrao/sudoku` 9×9 set, 90/10
split, **8.1 M** train puzzles. It lives in the JAX repo as
`sudoku_train.bagz`; export it to JSONL (needs that repo's env with `bagz`+`tf`):

```python
import bagz, tensorflow as tf, json
r = bagz.Reader(".../hackable_diffusion_adapter/data/sudoku/sudoku_train.bagz")
with open("dataset/sudoku_train_full.jsonl", "w") as f:
    for i in range(len(r)):
        ex = tf.train.Example(); ex.ParseFromString(r[i])
        f.write(json.dumps({
            "prompt":     ex.features.feature["puzzle"].bytes_list.value[0].decode(),
            "completion": ex.features.feature["solution"].bytes_list.value[0].decode(),
        }) + "\n")
```
(`dataset/sudoku_sft.jsonl` is a committed 64-row slice for smoke tests.)

### Test set — `dataset/sudoku_eval_{easy,medium,hard}.jsonl` (committed, 100 each)

Difficulty-stratified eval, **100 puzzles per band**, by clue count (matching
the JAX on-device metric and upstream gemma):

| band | clues (givens) |
|---|---|
| easy | ≥ 40 |
| medium | 30–39 |
| hard | < 30 |

These are the exact held-out puzzles used for the JAX↔TRL comparison.

## Train

```bash
bash DiffGemma/scripts/run_diffgemma_sft_trl.sh   # or directly:
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python -m DiffGemma.diffgemma_trl.sft_train \
  --model_name_or_path unsloth/diffusiongemma-26B-A4B-it \
  --dataset_name ./dataset/sudoku_train_full.jsonl --dataset_path ./dataset/sudoku_train_full.jsonl \
  --sft_variant reweighted-ce --encoder_loss_weight 1.0 --decoder_loss_weight 1.0 \
  --diffusion_self_conditioning_prob 0.5 --diffusion_noise_min 0.0001 --diffusion_noise_max 0.9999 \
  --diffusion_weight_clip 100 \
  --use_unsloth false --use_peft true --lora_r 64 --lora_alpha 128 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --max_steps 4000 --per_device_train_batch_size 4 --learning_rate 1.5e-4 \
  --warmup_steps 100 --lr_scheduler_type cosine --max_grad_norm 1.0 \
  --logging_steps 20 --save_steps 1000 --save_strategy steps --report_to none \
  --remove_unused_columns false --bf16 true --max_prompt_length 256 \
  --output_dir ./xp_sft_reweighted_ce
```
`--lora_target_modules q_proj … down_proj` = attention + dense MLP (the MoE
experts are `nn.Parameter` and the router Linear is `proj`, so both stay frozen
— the Unsloth/JAX `attn;mlp2` config). ~1 s/step on one H200 (~1.1 h / 4000).

## Eval

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python -m DiffGemma.diffgemma_trl.sft_eval_bands \
  --adapter_path ./xp_sft_reweighted_ce/checkpoint-4000 --bands_dir ./dataset --max_denoising_steps 64
# -> easy/medium/hard: full_solve + cell_acc (100 puzzles each)
```

## Other benchmarks (Math / Countdown / MBPP / HumanEval)

Per-dataset eval scripts (mirroring `sft_eval_gsm8k.py`) share one backbone,
`sft_eval_common.py`, and run against **either** the local in-process diffusion
`generate()` **or** a vLLM server (`--backend {local,vllm}`):

| script | dataset | metric | scoring reuse |
|---|---|---|---|
| `sft_eval_math.py` | `HuggingFaceH4/MATH-500` | exact-match | `math_verify` (boxed fallback) |
| `sft_eval_countdown.py` | `dataset/countdown_cd3_test.jsonl` | exact-match | `rewards.py` (validate/evaluate_equation) |
| `sft_eval_mbpp.py` | `google-research-datasets/mbpp` | pass@1 | `gdsd/utils/local_sandbox.py` |
| `sft_eval_humaneval.py` | `openai/openai_humaneval` | pass@1 | `gdsd/utils/local_sandbox.py` |

**Base model only** (the "no further adapters" setting) = just omit `--adapter_path`:

```bash
# local diffusion generate(), base model, first 50 examples:
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python -m DiffGemma.diffgemma_trl.sft_eval_math \
  --backend local --limit 50
# a LoRA checkpoint instead of the base model:
... python -m DiffGemma.diffgemma_trl.sft_eval_math --backend local \
  --adapter_path ./xp_sft_math/checkpoint-4000
```

Run all four at once via `DiffGemma/scripts/run_diffgemma_eval.sh`
(`ADAPTER_PATH=` empty → base model).

### vLLM serving

Start the server with the developer-guide config, then point the eval client at it
(`--backend vllm`). The diffusion sampler / canvas length are fixed at **serve**
time; `--max_denoising_steps` is ignored for the vLLM backend.

```bash
bash DiffGemma/scripts/serve_diffgemma_vllm.sh          # vllm serve google/diffusiongemma-26B-A4B-it ...
PYTHONPATH=$(pwd) python -m DiffGemma.diffgemma_trl.sft_eval_humaneval \
  --backend vllm --vllm_base_url http://localhost:8000/v1 \
  --vllm_model google/diffusiongemma-26B-A4B-it --limit 50
```

LoRA via vLLM: serve with `ADAPTER_NAME=ckpt ADAPTER_PATH=... bash ...serve_diffgemma_vllm.sh`
(adds `--enable-lora --lora-modules`), then eval with `--vllm_model ckpt` (requires a
diffusion vLLM build with LoRA support).

> The code sandbox (`local_sandbox.py`) is best-effort (`reliability_guard`), **not** a
> true security sandbox — run MBPP/HumanEval inside the project venv only. `should_execute`
> blocks `os`/`shutil`/threading imports.

> On CSCS Clariden, run every GPU step via `srun -p debug … --environment=<edf.toml>`
> (account `a0112`) — never on the login node.

## Parity notes vs the JAX SFT

Same objective, hyperparameters, prompt, and (held-out) data. Cross-framework
differences that **cannot** be made identical: PyTorch vs JAX numerics + AdamW;
HF has **separate** encoder/decoder modules (JAX shares one network), so the
encoder loss trains a separate adapter; **fused** JAX einsums (`kv`, `gate+up`)
vs **separate** HF Linears → different LoRA factorization; and each framework's
model is best served by its **own** native sampler (HF entropy-bound AR-commit
vs JAX DDIM). Run with `num_workers=0` on the small band sets (the multi-worker
loader under-counts them).
```
