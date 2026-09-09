# Math SFT for DiffusionGemma

Supervised fine-tuning of `google/diffusiongemma-26B-A4B-it` on math reasoning traces,
then re-evaluation on the math benchmark suite (MATH-500, Minerva, OlympiadBench, AMC,
AIME24/25). The base model already scores well under long generation (e.g. MATH-500 92.8%
at the `g2048/8c` config); the goal is to see whether SFT on curated traces improves it,
across several SFT objectives and two dataset designs.

All artifacts live under `dataset/`, `local/`, `DiffGemma/diffgemma_trl/`, and `eval_out/`.

---

## 1. Dataset designs (two response objectives, same problems)

Source: **OpenR1-Math-220k** (`default` config, ~93.7k rows, cached locally). We chose the
canvas / length target from the eval sweep — the model's best math results come at **gen
4096 / canvas 256 (16 canvases)**, so completions are length-scaled to fit a **4096-token
canvas** (see `dataset/analyze_math_sft_lengths.py` and `eval_out/math_sft_length_analysis.md`).

Both designs emit `{"uuid", "prompt", "completion"}` where the completion is:

```
<reasoning>
{reasoning}
</reasoning>
<answer>
\boxed{answer}      # always the ground-truth `answer` column (cleaned)
</answer>
```

| design | reasoning source | correctness filter | rows |
|---|---|---|--:|
| **`solution`** | Numina reference `solution` column (concise, human-written) | — | 30,345 |
| **`r1_cot`** | R1 `<think>` chain-of-thought, **post-`</think>` summary stripped** | strict `math_verify` (drops the ~30% that only pass the LLM judge) | 30,345 |

Key decisions baked into `dataset/make_openr1_math_jsonl.py`:

- **Ground-truth answer box.** The `<answer>\boxed{…}</answer>` uses the dataset's `answer`
  column, not a re-extracted last-box — this fixes partial/wrong answers on multi-answer
  rows. `clean_answer()` strips mangled unit markup (`\mathrm{~}`, `\mathrm{}`) while keeping
  content-bearing units (`\mathrm{Ah}`, `\mathrm{E}`).
- **De-duplicated R1 traces.** OpenR1's R1 message is `<think>{CoT}</think>{restated summary}`
  — i.e. the problem solved twice. `r1_cot` keeps only the `<think>` CoT so the target isn't
  taught to re-derive.
- **Strict correctness (r1_cot).** `--strict_mathverify` keeps only rows whose trained
  generation passed rule-based `math_verify`. A 10-example hand-check showed the reasoning's
  final answer and the `answer` column agree cleanly for `math_verify`-True rows but
  systematically disagree for the judge-only rows (multiple-choice letter-vs-value,
  incomplete multi-part answers).
- **Matched samples.** `solution` is subsampled to exactly `r1_cot`'s problems (aligned on
  `prompt`, since OpenR1 `uuid`s are non-unique junk). Both datasets contain the **same
  30,345 problems**, so the sweep cleanly isolates the reasoning-source variable.

Regenerate (datasets are `.gitignore`d — large):

```bash
python dataset/make_openr1_math_jsonl.py --response_mode r1_cot --strict_mathverify \
    --canvas_length 4096 --out dataset/openr1_math_sft_r1cot_c4096.jsonl
python dataset/make_openr1_math_jsonl.py --response_mode solution --canvas_length 4096 \
    --restrict_prompts dataset/openr1_math_sft_r1cot_c4096.jsonl \
    --out dataset/openr1_math_sft_solution_c4096.jsonl
```

---

## 2. SFT objective sweep

`local/math_sft_sweep.sh` runs **response modes × objectives** — the two datasets above ×
the three DiffusionGemma SFT variants — for **6 runs total**, each 4k steps with LoRA
(r64/α128), checkpoints at 1k/2k/3k/4k, canvas 4096, data-parallel via `accelerate`.

| axis | values |
|---|---|
| response (`RESPONSES`) | `solution`, `r1cot` |
| objective (`--sft_variant`) | `base-sft`, `reweighted-ce`, `loo-ce` |

```bash
bash local/math_sft_sweep.sh                       # full 6-run sweep, GPUs 0,1,2,3
GPUS=1,2 RESPONSES=r1cot VARIANTS=base-sft bash local/math_sft_sweep.sh   # one cell
SMOKE=1 GPUS=1,2 bash local/math_sft_sweep.sh      # fast pipeline smoke
```

**Chat-route parity.** Training builds prompts with `apply_chat_template` (the chat route),
byte-for-byte identical to the math-benchmark eval's default route — same `<bos>`,
`<|channel>thought<channel|>` priming and `<reasoning>\n` prefill. Implemented via a
`chat_mode` branch in `DiffusionGemmaSFTCollator` + the shared `MATH_INSTRUCTION`
(single source of truth in `sft_data_utils.py`, imported by `sft_eval_mathbench.py`).
Flags: `--chat_route true --reasoning_prefill true --enable_thinking false`.

---

## 3. Encoder AR loss — the interleaved forward/backward decision

The DiffusionGemma SFT objective is `decoder_loss_weight·(canvas denoising CE) +
encoder_loss_weight·(encoder AR next-token CE over [prompt; completion])`. Two problems had
to be solved to actually run it:

1. **The encoder loss was silently skipped.** The lookup for `encoder`/`lm_head` assumed the
   bare model, but under multi-GPU `accelerate` the model is a **DDP wrapper** (no
   `get_base_model`/`.model`), so it fell through to a decoder-only degrade. Fixed by
   unwrapping DDP (`accelerator.unwrap_model`) then PEFT before reaching
   `base.model.encoder` / `base.lm_head` (`_encoder_ar_loss`).

2. **With the encoder loss active, canvas 4096 OOMs.** We run **two data-parallel replicas**
   (a full model per GPU — no memory sharing), and `DiffusionGemmaForBlockDiffusion` does not
   support gradient checkpointing. A single combined `loss.backward()` holds *both* the
   decoder and encoder activation graphs simultaneously → OOM.

   **Decision — interleaved forward/backward** (chosen over FSDP sharding as the lower-risk
   first option). `training_step` does decoder forward → `backward` → encoder forward →
   `backward`, so only **one** activation graph is resident at a time (~halves peak memory).
   Gradients accumulate additively (`∂(dec+enc) = ∂dec + ∂enc`) → identical optimizer step.
   The first backward runs under `accelerator.no_sync(model)` so DDP all-reduces once (on the
   second backward), avoiding the "grad ready twice" error.

   Result: fits at ~**131/140 GB** per GPU at canvas 4096, no OOM/DDP errors, encoder loss
   confirmed contributing (train_loss 5.5 → 8.5). Cost: ~**47.5 s/step** vs ~30 s
   decoder-only (~+65%). Fallback if this had failed: FSDP-shard one model across the 2 GPUs.

Implemented in `DiffGemma/diffgemma_trl/sft_trainer.py` (`_encoder_ar_loss`, split
`_decoder_loss`/`_encoder_loss`, `training_step`). Toggle with `ENC_W` (default 1.0;
`ENC_W=0.0` → decoder-only, uses the stock single-backward path) and `--interleaved_backward`
(default off). **This interleaving applies to the legacy single-canvas path only** — the
block-structured objective in §6 shares one encoder pass and uses a plain single backward.

---

## 4. Evaluation

Checkpoints are evaluated with the same math-benchmark harness used for the base model.
The eval is block-autoregressive: `#canvases = ceil(max_new_tokens / canvas_length)`.
The base-model canvas sweep (`local/run_canvas_sweep.sh`, `local/run_g4096_c256.sh`) is in
`eval_out/RESULTS.md`; the added **g4096/16c** column shows longer generation helps the
hardest sets — AIME24 57.5→64.2, AMC 87.5→95.0, AIME25 45.0→49.2 (vs g2048/8c), while
saturated sets (MATH-500, Minerva) are flat.

---

## 5. Key files

| file | role |
|---|---|
| `dataset/analyze_math_sft_lengths.py` | completion length distribution across candidate datasets |
| `dataset/make_openr1_math_jsonl.py` | build the `solution` / `r1_cot` datasets (flags: `--response_mode`, `--strict_mathverify`, `--restrict_prompts`) |
| `local/math_sft_sweep.sh` | the objective × response SFT sweep launcher (`BLOCK=1`, `MAIN_PORT`, `_block` dirs) |
| `DiffGemma/diffgemma_trl/sft_data_utils.py` | collator + chat route + block-diffusion `_call_block` (§6) |
| `DiffGemma/diffgemma_trl/sft_configs.py` | SFT flags (`--chat_route`, `--block_diffusion`, `--interleaved_backward`, …) |
| `DiffGemma/diffgemma_trl/sft_trainer.py` | loss + encoder-loss fix + interleaved `training_step` + block `_block_loss` (§6) |
| `DiffGemma/diffgemma_trl/sft_eval_mathbench.py` | math-benchmark eval (chat/raw routes) |
| `eval_out/RESULTS.md` | curated benchmark results table |

---

## 6. Block-structured (multi-canvas) SFT  `--block_diffusion`

The legacy objective (§2–3) packs the whole completion into **one 4096-token canvas**
conditioned only on the prompt, whereas eval denoises **256-token canvases
block-autoregressively**, each conditioned on the prior committed blocks via the encoder KV
cache. `--block_diffusion` removes that train/eval mismatch with a **stochastic
block-diffusion objective** (default off → legacy path unchanged).

**Objective.** Tokenize the full completion (+EOS, cap `--max_completion_length`) and split
into `canvas_length`-token blocks. Per example draw `u∼U(0,1)`, pick block
`b = min(⌊u·num_blocks⌋, num_blocks−1)`, then:

- **encoder input** = `prompt + clean blocks 0..b−1` (the committed context),
- **denoising canvas** = block `b` (corrupted → denoised) — the decoder target,
- **tail** (blocks `b+1..`) is dropped.

Both terms are per-token means (token-normalized): the decoder denoising CE over block `b`,
and the encoder AR CE over the **prefix-completion tokens** (blocks `0..b−1`, flagged by a new
`prompt_comp_mask`). `b=0` has an empty prefix → the encoder term is a true 0 that step
(nothing committed yet to autoregressively predict); the decoder still learns the
inference-critical first block. The prompt is never a target in either term.

**Positions come for free.** In a single training forward the encoder writes its output into
the decoder's KV cache, so the decoder defaults `decoder_position_ids =
arange(cache_len, cache_len + canvas_length)` (`modeling_diffusion_gemma.py`). With encoder
input `[prompt; blocks 0..b−1]`, block `b`'s canvas lands at positions `[P + b·256 …]` —
byte-identical to inference block `b`. No explicit position bookkeeping needed.

**One shared encoder pass, single backward.** The decoder forward already runs the encoder
over `[prompt; prefix]` and returns `encoder_last_hidden_state`; the encoder AR loss **reuses
it** (`_block_loss`) instead of a second encoder pass. This was chosen over the §3 interleaved
forward/backward after the interleaved path — two ~4096-token encoder graphs held for a
combined backward — both **OOM'd** (137 GB) *and* hit a **DDP reducer error** (the two-backward
+ `no_sync` pattern breaks when the per-step graph varies with `b`). Sharing one pass gives a
single combined backward where **every parameter gets a gradient every step** (DDP-clean) and
**fits at ~120 GB** at ~27 s/step. So `--interleaved_backward` is **off and unused** in block
mode; it remains only for the legacy 4096 single-canvas path.

**Run** (`local/math_sft_sweep.sh`, `BLOCK=1`): sets `--block_diffusion true
--diffusion_canvas_length 256 --max_completion_length 4096 --interleaved_backward false` and
writes to `_block`-suffixed `out/`+wandb names (`DATA_CANVAS` keeps the dataset's build-time
length filter separate from the 256 training block length). Split across 4 GPUs by dataset
with distinct rendezvous ports:

```bash
RESPONSES=solution GPUS=0,1 MAIN_PORT=29500 BLOCK=1 bash local/math_sft_sweep.sh &   # 3 variants
RESPONSES=r1cot    GPUS=2,3 MAIN_PORT=29600 BLOCK=1 bash local/math_sft_sweep.sh &   # 3 variants
```

Implemented in `sft_data_utils.py` (`_call_block`, `_encode_completion_full`,
`prompt_comp_mask`), `sft_trainer.py` (`_block_loss`, `_encoder_ar_loss` hidden-reuse,
`_lm_head_and_cap`), `sft_configs.py` / `sft_train.py` (flags), `local/math_sft_sweep.sh`.
