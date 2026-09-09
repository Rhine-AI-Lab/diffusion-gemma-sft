# DiffusionGemma Terminal-Agent SFT — Session Notes

## Goal

Improve DiffusionGemma-26B-A4B-it's performance on terminal-agent tasks
(OpenThoughts-TBLite via [Harbor](https://github.com/laude-institute/harbor),
using `terminus-2` as the agent — Terminal-Bench's own minimal reference
agent, chosen so the score reflects the model's own agentic ability rather
than a scaffold's). Base model baseline: **16/100 tasks passed**, mean reward
~0.17.

Prior SFT checkpoints trained by others on this base model
(`WillBankes/diffgemma-tmax-sft-nothink-500`,
`Rhine-AI/diffusiongemma-26B-A4B-tmax-sft-ckpt{1000,2000,3000,4000}`) had
**collapsed** (near-0% pass, degenerate output) regardless of training steps.
This session diagnoses why, fixes the pipeline, and iterates on a working
SFT recipe.

## Pipeline (this directory)

```
allenai/tmax-sft (Qwen3.6-27B real successful traces, native tool-calling)
        │  convert_tmax_sft.py
        ▼
terminus-2 JSON-action schema training pairs (messages, completion)
        │  train_agent_sft.py  (LoRA, DiffusionGemmaSFTTrainer)
        ▼
LoRA checkpoint
        │  merge_lora.py  (fp32 merge, decoder-only by default)
        ▼
merged bf16 model  →  served via vLLM  →  evaluated via run_eval.sh (Harbor)
```

- **`convert_tmax_sft.py`** — converts `allenai/tmax-sft` (Qwen's native
  tool-calling protocol: `THOUGHT:` prose + `bash(command=...)` tool call)
  into terminus-2's JSON-action schema (`{analysis, plan,
  commands:[{keystrokes, duration}], task_complete}`), rebuilding the system
  prompt with terminus-2's own template so the *input distribution* matches
  what the model sees at eval time. History is advanced with the **canonical
  JSON** for each turn (not Qwen's raw text) — see Bug 1 below for why this
  matters.
- **`build_sft_data.py`** — equivalent conversion from our own real
  terminus-2 rollout trajectories (`jobs/*/agent/trajectory.json`), for
  self-distillation if revisited (abandoned early this session in favor of
  the much larger/higher-quality tmax-sft source).
- **`agent_sft_data_utils.py`** — `DiffusionGemmaAgentSFTCollator`: runs the
  multi-turn `messages` list through the tokenizer's chat template (instead
  of a flat prompt string), casts `prompt_mask` to bool (transformers'
  `masking_utils` needs a real bool mask, not int64/float).
- **`train_agent_sft.py`** — training entry point. Mirrors
  `DiffGemma/diffgemma_trl/sft_train.py`'s model loading/PEFT/Trainer wiring,
  swaps in the multi-turn collator. **Must be launched with
  `--use_unsloth false`** (see Bug 2).
- **`merge_lora.py`** — merges LoRA into the base model (CPU, fp32, then
  cast to bf16 — bf16 merge accumulates rounding error large enough to tank
  accuracy, per `DiffGemma/diffgemma_trl/diag_merge.py`). **Defaults to
  decoder-only merge** (see Bug 3); pass `--keep_all_adapters` to merge both.
- **`run_eval.sh`** — runs the 100-task OpenThoughts-TBLite benchmark
  against a locally-served model via Harbor + terminus-2.

## Bugs found and fixed this session

### 1. Training-history contamination (root cause of the *original* collapse)

`convert_tmax_sft.py` originally advanced the multi-turn history with each
turn's *raw* Qwen content (`THOUGHT: ...` prose, command dropped — it lived
in `tool_calls`, not `content`). That rationale is valid for
`build_sft_data.py` (our own rollouts, whose raw content already *is*
terminus-2 JSON) but not for Qwen's foreign protocol. Result: **92.9% of
examples** had assistant history turns that were plain prose with no JSON and
no command, while the *target* for the current turn was full terminus-2
JSON — the model never saw the JSON action schema in its own conversation
history, only as a prediction target. At eval, its own history *is* JSON, so
it faced a distribution never seen in training and drifted to prose/
repetition. **Fix:** advance history with the canonical `completion` JSON
string (restores the command too).

### 2. Silent full fine-tune instead of LoRA

`train_agent_sft.py` only calls `get_peft_model` when
`not (use_unsloth and unsloth_lora_rank > 0)`. Both default truthy, so
**without `--use_unsloth false` the LoRA block is skipped entirely** and the
full 25.8B model trains — OOMs a single 140GB GPU (52GB weights + full
gradients + full Adam states), and the failure looked like an "activation
OOM" immune to every canvas/prompt/self-conditioning/gradient-checkpointing
knob, because none of those touch a full-FT footprint. Correct LoRA launch
prints `trainable params: 193,840,128 || trainable%: 0.7450` and fits
~120GB.

### 3. Encoder/decoder weight-tying → LoRA double-merge

DiffusionGemma's encoder and decoder **share the same weight tensors**
(confirmed via `data_ptr`: 627/631 decoder params alias an encoder param —
the model card also confirms the encoder is used at inference, to fold each
completed canvas into the KV cache before the next canvas is generated). But
PEFT creates **separate** LoRA adapters per module object
(`encoder.language_model.*`, `encoder.vision_tower.*`, `decoder.*` — 602
target modules total), so `merge_and_unload()` summed **both** deltas into
the shared tensor: `W + Δ_dec + Δ_enc` instead of `W + Δ_dec`. Verified on
one checkpoint: applied delta norm exactly matched `Δ_dec + Δ_enc` (8.73),
not `Δ_dec` alone (5.77) — and `Δ_enc` (6.53) was *larger* than the intended
`Δ_dec`. Single-turn A/B probes showed the practical effect was milder than
the math implied (the encoder AR loss and decoder diffusion loss train on
the same completion tokens, so both deltas point roughly the same direction
— a ~2x over-application in a coherent direction, not random corruption) —
but it's still wrong. **Fix:** `merge_lora.py` zeros all non-`.decoder.`
LoRA adapters before merging (default; `--keep_all_adapters` reverts).
**Better fix (adopted for later runs):** just don't train an encoder
adapter at all — pass `--encoder_loss_weight 0.0`. The trainer's own config
docstring already documents this as "decoder-only (Unsloth-style)
objective," and the encoder loss's early-return (`if weight <= 0: return
zeros`) means the encoder forward pass is skipped entirely, not merely its
result discarded.

### 4. `task_complete: true` under-represented (root cause of the *v2* collapse)

After fixing bug 1, a full run (`agent_sft_data_v2_filt.jsonl`, 2000 steps)
still scored **0/100, 100% AgentTimeoutError**. Root cause: the model
**never emitted `task_complete: true`** (0 times across 153 probed agent
turns) — terminus-2 only ends a task when the agent declares completion, so
every task ran to the timeout. Cause: `task_complete: true` was only **7.1%**
of training turns; the undertrained model (0.23 epoch) collapsed to the
majority `false` class. **Fix:** oversample `task_complete: true` turns
~7x → **~35%** of the dataset. Validated via targeted probes (clean-shell
vs. task-still-in-progress states): the model correctly declares done when
done and keeps working when not, without becoming trigger-happy.

### 5. Wrong canvas size (model-card mismatch)

Per the official HF model card: native `canvas_length=256`,
`max_new_tokens=256`, EOS-only sequence termination (the "adaptive stopping"
—entropy < 0.005, stable top-1 across 2 denoising steps— only shortens
*per-canvas* denoising, not the overall sequence), and a recommended
**decaying temperature 0.8→0.4** with an entropy-bound sampler (not greedy).
We had trained/evaluated at `diffusion_canvas_length=512`, and vLLM's
startup log explicitly disables the model's own 256-token
`max_new_tokens` safety net ("removes the server-wide max_new_tokens cap"),
so nothing bounded generation except `max_model_len` (16384) if the model
under-emitted EOS. **Fix:** retrain at the native `canvas_length=256`.

### 6. Degenerate / contaminated training targets

8.35% of completions had near-empty `analysis`/`plan` (<5 chars) — training
the model to produce content-free reasoning. Separately, 4 source
trajectories (0.08% of examples) contained the phrase "a previous agent's
attempt" in genuinely low-content turns, which (via the natural per-turn
history fan-out of a trajectory-to-examples conversion) got disproportionate
training exposure relative to its true rarity. **Fix:** drop both classes of
example as training *targets* (kept in history, since that reflects real
trajectory continuity) before oversampling.
**Important caveat:** cross-checking against a **base-model** (no SFT)
eval at `temperature=0.6` showed the *same* "previous agent" self-attribution
pattern occurring natively, correlated strongly with failure (0 passes had
any mention; every failure had 1-37 mentions). So this data-hygiene fix is
reasonable but is **not** expected to be the primary fix — the underlying
behavior (the model loses track of its own authorship after repeated action
failures, e.g. shell/JSON escaping errors in complex commands, and
hallucinates a distinct "previous agent" whose failed work it then discards
and restarts from scratch) looks like a genuine base-model agentic capability
gap, not a training-data artifact.

## Known-working training launch (as of this session)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=<gpu> \
  python3 -m tblite_eval.train_agent_sft \
  --model_name_or_path google/diffusiongemma-26B-A4B-it \
  --dataset_path tblite_eval/agent_sft_data_v4.jsonl \
  --sft_variant base-sft \
  --use_unsloth false \
  --use_peft true --lora_r 64 --lora_alpha 128 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --diffusion_canvas_length 256 --max_prompt_length 2048 \
  --encoder_loss_weight 0.0 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --learning_rate 1.5e-4 --max_steps 2000 --logging_steps 10 --save_steps 250 \
  --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --remove_unused_columns false \
  --prompt_column messages --completion_column completion \
  --output_dir ./tblite_eval/xp_agent_sft_v5
```

Fits ~114GB on one 140GB H200 (LoRA, 193.8M trainable params, 0.745%).
~13s/step, ~7.3h for 2000 steps (0.23 epoch over ~85k examples).

Serving/eval must run at `N_CONCURRENT<=4` / vLLM `--max-num-seqs 4`: at
higher concurrency the model's over-generation (filling context, not
emitting EOS reliably) can blow the KV cache and crash the engine. Harbor's
`run_eval.sh` also defaults to `MAX_INPUT_TOKENS=8192` with harbor's own
`proactive_summarization_threshold=8000` — i.e. proactive context
summarization triggers after only ~192 tokens of history, firing on almost
every task almost immediately. Recommend overriding
`MAX_INPUT_TOKENS=14000` (matching the vLLM server's `--max-model-len
16384` minus headroom for `--max-output-tokens`) so summarization only
fires on genuinely long rollouts.

## Status at end of session

- Data pipeline bugs (1, 2, 3, 4, 6) diagnosed and fixed; canvas-size
  mismatch (5) fixed for the latest run.
- `xp_agent_sft_v5` (cleaned dataset, canvas=256, decoder-only objective) is
  the latest training run.
- Open question, not yet resolved: the base-model "previous agent"
  self-attribution failure mode (recovering from its own repeated action
  errors) — appears to be a genuine capability gap rather than something
  fixable by the current next-action-imitation SFT approach. Next steps to
  consider: training examples that explicitly demonstrate in-place
  debugging/recovery (rather than restart-from-scratch), or a system-prompt
  framing experiment (explicitly stating there is no "previous agent," this
  is a single continuous session) as a cheap non-training test.
- A base-model eval at `temperature=0.6` (vs. the `temperature=0` used
  everywhere else this session) showed a notably higher early pass rate —
  worth confirming at full scale, since greedy decoding may not be well
  matched to this model's entropy-bound/adaptive-stopping design.
