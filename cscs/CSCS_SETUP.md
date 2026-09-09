# CSCS Alps setup (Clariden / GH200) — gdsdv2 DiffusionGemma SFT (PyTorch/TRL)

Thin NGC **PyTorch** container + a Python venv on `$SCRATCH`, following the CSCS
[modified NGC PyTorch container tutorial](https://docs.cscs.ch/tutorials/ml/llm-inference/#build-a-modified-ngc-pytorch-container)
and the same pattern as the sibling `d2` project. The image only provides the
system + a GPU-tested arm64 PyTorch; the project env is installed at runtime as a
venv because the enroot squashfs image is read-only.

> **Platform:** Alps nodes are GH200 (Grace-Hopper, **aarch64**). Account: `a0112`.
> `storage.conf` and `$SCRATCH/ce-images` (LUSTRE-striped) are already configured.
> **`/users` (`$HOME`) is NOT mounted in the container** — keep the repo, venv,
> token, datasets, and HF cache on `$SCRATCH`.

## Quickstart (TL;DR)

```bash
cd $SCRATCH/projects/gdsdv2
# 0. one-time: HF token on scratch (the model is gated)
mkdir -p $SCRATCH/.secrets && printf '%s' "$YOUR_HF_TOKEN" > $SCRATCH/.secrets/hf_token && chmod 600 $SCRATCH/.secrets/hf_token

# 1. build the container image (once) — compute node, ~15-20 min
srun -A a0112 -p debug -t 00:30:00 bash cscs/build_image.sh

# 2. create the venv on scratch (once) — inside the container
srun -A a0112 -p debug -t 00:30:00 --environment=./cscs/gdsdv2-pt.toml bash cscs/setup_env.sh

# 3. (recommended) pre-download the 26B checkpoint into $SCRATCH/huggingface
srun -A a0112 -p debug -t 00:30:00 --environment=./cscs/gdsdv2-pt.toml bash -c '
  source .venv/bin/activate
  HF_TOKEN=$(cat $SCRATCH/.secrets/hf_token) \
  huggingface-cli download google/diffusiongemma-26B-A4B-it'

# 4. smoke test the SFT env on the debug partition
sbatch cscs/sft_smoke.sbatch
tail -f logs/sft_smoke_*.log
```

## Files

| File | Role |
|---|---|
| `cscs/Dockerfile` | NGC PyTorch 25.06 + git (thin). |
| `cscs/build_image.sh` | `podman build` + `enroot import` → `$SCRATCH/ce-images/gdsdv2-pt+25.06.sqsh`. |
| `cscs/gdsdv2-pt.toml` | Container Engine EDF (image, mounts, workdir, NCCL/HF env). |
| `cscs/setup_env.sh` | Create `.venv` (`--system-site-packages`) + install the SFT HF stack. |
| `cscs/sft_smoke.sbatch` | Debug-partition smoke test (loads 26B, runs 20 SFT steps on the 64-row slice). |

## Environment notes

- The venv reuses the **container's** arm64 GPU PyTorch (`--system-site-packages`);
  we never pip-install torch. On top we add `transformers==5.12.1` (ships
  `diffusion_gemma`/`gemma4` natively — required), `peft>=0.19`, `trl`, accelerate,
  datasets, sentencepiece, safetensors, einops, bitsandbytes, huggingface-hub.
- `--use_unsloth false` ⇒ unsloth not needed; the native
  `DiffusionGemmaForBlockDiffusion` path loads the model.
- Image shortcut: the base is identical to `d2+25.06.sqsh`; you can point the EDF
  `image=` at that existing file to skip the build.

### GH200-specific fixes (discovered while bringing this up)

1. **torchao** — the NGC image ships torchao 0.11, but `peft>=0.19`'s
   `is_torchao_available()` *raises* (instead of returning False) when torchao
   < 0.16, killing every `get_peft_model()`. `setup_env.sh` shadows it with
   `torchao>=0.16` (`--no-deps`, so the container torch is untouched; peft only
   reads the version metadata, the cpp kernels stay unloaded).
2. **MoE experts kernel** — transformers auto-selects `grouped_mm` on torch≥2.9 /
   SM90, but `torch._grouped_mm` device-asserts (`GroupMMCommon.cuh`: byte size
   must be a multiple of 16) on the unaligned per-expert token counts.
   `sft_train.py` now passes `experts_implementation=eager` (env-overridable via
   `DIFFGEMMA_EXPERTS_IMPL`). `eager` is memory-safe but slow (~12 s/step on the
   smoke); `batched_mm` is faster but OOMs at batch ≥ ~2 (it replicates
   [experts × tokens], ~60 GB).
3. **Memory / batch size** — the 26B model is ~50 GB of the 96 GB card and
   `DiffusionGemmaForBlockDiffusion` sets `_supports_gradient_checkpointing=False`
   (so activations can't be traded for compute). With `--diffusion_self_conditioning_prob 0.5`
   (two forward passes) the run needs ~11 GB/sample, so **single-GPU LoRA SFT fits
   at batch ≤ 3** (the smoke uses batch 2). For the upstream `batch 4` keep
   effective batch via `--gradient_accumulation_steps 2 --per_device_train_batch_size 2`,
   or shard across the node's 4 GH200 with `accelerate launch`, or use the
   memory-efficient `--use_unsloth true` path.

**Verified smoke** (job ran COMPLETED on `-p debug`): batch 2, 20 steps, loss
1.46 → 0.74, ~12.5 s/step (eager experts), 26B + LoRA on one GH200.

## Data

SFT data is **JSONL** with `{"prompt": ..., "completion": ...}` per line.
- `dataset/sudoku_sft.jsonl` — committed 64-row slice (smoke tests). Used by the
  smoke sbatch.
- `dataset/sudoku_train_full.jsonl` — the real 8.1 M-puzzle set (~2.7 GB, **not
  committed**). Export it from the JAX repo's `sudoku_train.bagz` per
  `DiffGemma/diffgemma_trl/SFT_README.md`, then run the full
  `--max_steps 4000` command on `-p normal`.

## Full training run

Once the smoke test passes and `sudoku_train_full.jsonl` exists, run on a longer
partition (`-p normal`), inside the same EDF. This keeps the upstream **effective
batch of 4** on one GH200 via grad-accum (per-device 2 × accum 2) — see the memory
note above:

```bash
srun -ul -A a0112 -p normal -t 11:55:00 --environment=./cscs/gdsdv2-pt.toml bash -c '
  source .venv/bin/activate
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$SCRATCH/projects/gdsdv2 HF_TOKEN=$(cat $SCRATCH/.secrets/hf_token) \
  python -m DiffGemma.diffgemma_trl.sft_train \
    --model_name_or_path google/diffusiongemma-26B-A4B-it \
    --dataset_name ./dataset/sudoku_train_full.jsonl --dataset_path ./dataset/sudoku_train_full.jsonl \
    --sft_variant base-sft --encoder_loss_weight 1.0 --decoder_loss_weight 1.0 \
    --diffusion_self_conditioning_prob 0.5 --diffusion_noise_min 0.0001 --diffusion_noise_max 0.9999 \
    --use_unsloth false --use_peft true --lora_r 64 --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_steps 4000 --per_device_train_batch_size 2 --gradient_accumulation_steps 2 \
    --learning_rate 1.5e-4 --warmup_steps 100 --lr_scheduler_type cosine \
    --max_grad_norm 1.0 --logging_steps 20 --save_steps 1000 --save_strategy steps \
    --report_to none --remove_unused_columns false --bf16 true --max_prompt_length 256 \
    --output_dir ./xp_sft_base_sft_trl'
```

> To use all 4 GH200 on the node (true `per_device_batch_size 4`), swap `python`
> for `accelerate launch` and drop `CUDA_VISIBLE_DEVICES=0` (the repo's
> `DiffGemma/scripts/run_diffgemma_sft_trl.sh` already uses `accelerate launch`).
> `eager` MoE experts are slow (~12 s/step); for throughput try
> `DIFFGEMMA_EXPERTS_IMPL=batched_mm` with `--per_device_train_batch_size 1`.

## Troubleshooting

- **`torch.cuda.is_available()` is False / CPU torch** — the venv shadowed the
  container torch. Recreate with `--system-site-packages` and never `pip install torch`.
- **`trl` upgraded transformers** — re-pin `transformers==5.12.1` (setup_env.sh does
  this with a torch constraint so the GPU build is preserved).
- **`peft` ImportError on torchao < 0.16** — see GH200 fix #1; `pip install -U
  --no-deps "torchao>=0.16"` into the venv.
- **`GroupMMCommon.cuh` device-side assert in the MoE forward** — see GH200 fix #2;
  set `DIFFGEMMA_EXPERTS_IMPL=eager` (the default now).
- **CUDA OOM in `compute_loss`** — see GH200 fix #3; lower
  `--per_device_train_batch_size` (≤ 3 for single-GPU LoRA), use grad-accum,
  multi-GPU `accelerate launch`, or `--use_unsloth true`.
- **`does not support gradient checkpointing`** — expected; the model class sets
  `_supports_gradient_checkpointing=False`. Don't pass `--gradient_checkpointing`.
- **Model 401 / gated** — `google/diffusiongemma-26B-A4B-it` is currently public;
  if that changes, ensure the token has access or use the
  `unsloth/diffusiongemma-26B-A4B-it` mirror
  (`sbatch --export=ALL,MODEL=unsloth/diffusiongemma-26B-A4B-it cscs/sft_smoke.sbatch`).
- **NGC pull/import exceeds the debug 30-min cap** — run `build_image.sh` on `-p normal`.
