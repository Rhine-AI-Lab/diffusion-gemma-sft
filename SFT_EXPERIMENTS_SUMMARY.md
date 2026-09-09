# Terminal-Agent SFT Summary (TBLite/Harbor)

**Goal:** SFT DiffusionGemma to be a better terminal/coding agent on the
TBLite/Harbor benchmark (100 tasks).

## Bugs found and fixed

1. **History contamination** — `convert_tmax_sft.py` advanced multi-turn
   history with raw foreign-protocol (Qwen) prose instead of the canonical
   terminus-2 JSON completion. 92.9% of examples never showed the model its
   own action schema in history, only as a prediction target. *Fix:* advance
   history with the canonical `completion` JSON string.
2. **Silent full fine-tune instead of LoRA** — `train_agent_sft.py` skips
   `get_peft_model` unless `--use_unsloth false` is passed explicitly (both
   relevant flags default truthy). Without it, the full 25.8B model trains
   and OOMs, with a failure signature indistinguishable from an activation
   OOM. *Fix:* always pass `--use_unsloth false`.
3. **Encoder/decoder weight-tying double-merge** — DiffusionGemma's encoder
   and decoder share weight tensors (confirmed via `data_ptr`), but PEFT
   creates separate LoRA adapters per module, so naive merging summed both
   deltas into the shared weight. *Fix:* `merge_lora.py` zeros non-decoder
   adapters before merging (default); *better fix:* train with
   `--encoder_loss_weight 0.0` so no encoder adapter is trained at all.
4. **`task_complete: true` under-represented** — only 7.1% of training
   turns; an undertrained model collapsed to always predicting `false`,
   causing every task to run to timeout (0/100, 100%
   `AgentTimeoutError`). *Fix:* oversample `task_complete: true` turns ~7x
   to ~35%.
5. **Wrong canvas size** — trained/evaluated at `diffusion_canvas_length=512`
   against the model's native `canvas_length=256`; vLLM's server-wide
   `max_new_tokens` safety net was also disabled, so nothing bounded
   generation. *Fix:* retrain at the native 256.
6. **Degenerate/contaminated training targets** — 8.35% of completions had
   near-empty `analysis`/`plan` fields; a rare "previous agent's attempt"
   phrase got disproportionate exposure via trajectory-to-example fan-out.
   *Fix:* drop both classes as training targets (kept in history). See the
   dedicated subsection below — this turned out to be a genuine base-model
   behavior, not just a training-data artifact.

## Results

All numbers below are `n_completed_trials` / `n_errored_trials` /
pass-rate from each job's `result.json`, read directly from
`tblite_eval/jobs/*/result.json` (`stats.n_completed_trials`,
`stats.n_errored_trials`, `stats.evals.*.metrics[0].mean`). Several jobs
were interrupted before reaching the full 100-task target — those are
flagged explicitly, since a handful of completed trials is a much weaker
signal than a full run.

| Run | Trials completed (of target) | Errored | Pass rate | Note |
|---|---|---|---|---|
| **Base model, full 100 tasks** (`full_a` + `full_b`, verified disjoint 49+51-task split) | 100/100 | 41 | **17.0%** | combined from two batches |
| Base model, temp=0.6, full | 100/100 | 44 | 15.8% | |
| Base model, XML prompt variant | 89/100 | 77 | 2.2% | interrupted before 100 |
| SFT ckpt500 (pre-fix, buggy pipeline) | 100/100 | 100 | 0.0% | full run, 100% errored |
| SFT ckpt1000 (pre-fix, buggy pipeline) | 100/100 | 99 | 0.0% | full run |
| ckpt1000-4000 (early probes) | 11-12/100 each | 11 each | 0.0% | interrupted, small-N, indicative only |
| v2 (history-contamination fix only) | 42/100 | 42 | 0.0% | interrupted; 100% of completed trials errored (timeout) |
| v3 checkpoints (x2) | 9-10/100 each | 9-10 | 0.0% | interrupted, small-N |
| **v5** ckpt1250, `tasks_basepass` subset (14 tasks) | 14/14 | 13 | 14.3% (2/14) | full run on a small, easier subset only — not the full 100 |
| **v5, final checkpoint (step 2000), full benchmark** | 45/100 | 45 | **0.0%** | stopped early — see note below |

**Conclusion: no SFT checkpoint ever beat the base model's ~17% full-benchmark
pass rate.** The v5 recipe (all 6 fixes applied) was evaluated at its final,
completed checkpoint (step 2000, merged and never previously tested) on the
full benchmark; the run was stopped after 45/100 tasks once the pattern
became unambiguous: **all 45 completed trials errored with
`AgentTimeoutError`, 0 passes** (a further 4 trials were mid-flight when the
run was stopped and are excluded as `CancelledError` artifacts of the
shutdown itself, not genuine outcomes). Trial durations clustered into the
harness's timeout tiers (~15min, ~30min, and a hard 60-minute ceiling that
24% of trials hit), meaning the model was not completing tasks or failing
fast — it was running to the timeout on effectively every task attempted.
This directly confirms, at full scale, the same failure mode already visible
in the small 14-task `basepass` check (13/14 errored): the dominant failure
mode is long-horizon compounding drift across many-turn agentic rollouts,
structurally different from what next-action SFT imitation could fix.

## The "previous agent" self-attribution failure

While filtering degenerate training targets (bug 6), a rare training-data
phrase ("a previous agent's attempt") turned out to trace back to a genuine
**base-model behavior**, not just a data artifact: after repeated action
failures (e.g. shell/JSON escaping errors), the base model sometimes
hallucinates that a *different, prior* agent produced the broken state, and
discards/restarts rather than debugging in place.

Reproduced directly from the `base_temp06_t` job (100 full trials, base
model at temperature=0.6), counting literal "previous agent" mentions in
the model's own turns only (`source == "agent"` in `trajectory.json`,
excluding system/user turns):

| | n | trials with >=1 mention | mean mentions | median mentions |
|---|---|---|---|---|
| Passing trials | 15 | 7 (47%) | 16.7 | 0 |
| Failing trials | 85 | 63 (74%) | 23.7 | 4 |

This refines (and is less absolute than) an earlier informal read of this
data — it is **not** "zero passes ever mention it," but the pattern is
still real and directional: failing trials are ~1.6x more likely to contain
the phrase at all, and among trials that do mention it, failures show a
much higher typical count (median 4 vs. 0). The behavior looks like the
model losing track of its own authorship after a run of action errors,
rather than something the training-data fix alone was likely to resolve —
consistent with why this was left as an open question rather than treated
as closed by the data-hygiene fix.

**Not yet tried:** a system-prompt framing experiment (explicitly stating
there is no "previous agent," this is one continuous session) as a cheap,
non-training test of whether this is a framing issue or a deeper capability
gap.
