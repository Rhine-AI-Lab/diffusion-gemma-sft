# TBLite terminal-agent eval for DiffusionGemma

Runs [OpenThoughts-TBLite](https://github.com/open-thoughts/OpenThoughts-TBLite)
(100 terminal-agent tasks, r=0.911 correlated with Terminal-Bench 2 but much
cheaper to run) against DiffusionGemma, via
[Harbor](https://github.com/laude-institute/harbor).

Harbor is a generic agent/model evaluation framework: it spins up a Docker
sandbox per task, runs an **agent** (a program that talks to a **model** and
drives a terminal) inside it, and scores the result against the task's
verifier. We use `terminus-2`, Harbor's own reference agent — a minimal
"shell + model" loop with no extra coding-agent scaffolding — so the eval
measures DiffusionGemma's own tool-use, not e.g. Claude Code's harness.

## 1. Install

```bash
cd tblite_eval
./setup.sh
```

This does two things:
- `uv tool install harbor` — installs the `harbor` CLI in its own isolated
  environment (unrelated to this repo's torch/transformers venv).
- Clones `open-thoughts/OpenThoughts-TBLite` into `tasks/`. It isn't published
  as a named Harbor Hub dataset, so we run against the cloned directory
  directly with `harbor run --path`, which treats any directory of
  `<task>/task.toml` subfolders as an implicit local dataset.

## 2. Serve DiffusionGemma

Harbor talks to the model over an OpenAI-compatible HTTP API (via LiteLLM's
`hosted_vllm/` provider). From the repo root, reuse the existing vLLM launcher:

```bash
VLLM_SERVED_MODEL_NAME=diffgemma \
  bash DiffGemma/diffgemma_trl/run_vllm_openai_gemma4.sh \
  google/diffusiongemma-26B-A4B-it
```

`VLLM_SERVED_MODEL_NAME` must be a bare alias (letters/digits/`.`/`-`/`_`
only, no `/`) — LiteLLM's `hosted_vllm/<name>` routing rejects model strings
with more than one `/`, so the full HF repo id (`google/diffusiongemma-...`)
can't be used as-is.

The launcher uses the model repository's chat template by default and enables
vLLM's Gemma 4 tool-call and reasoning parsers. Set `VLLM_CHAT_TEMPLATE` only
when intentionally testing a custom template.

Also increase `--max-model-len` for this workload: terminal-agent transcripts
run many turns and get long. Match `MAX_INPUT_TOKENS`/`MAX_OUTPUT_TOKENS`
below to whatever you set on the server.

## 3. Run the eval

```bash
cd tblite_eval
VLLM_API_BASE=http://localhost:8000/v1 \
SERVED_MODEL_NAME=diffgemma \
N_CONCURRENT=4 \
./run_eval.sh
```

Under the hood this runs:

```bash
harbor run --path tasks \
  --agent terminus-2 \
  --model hosted_vllm/diffgemma \
  --ak api_base=http://localhost:8000/v1 \
  --ak 'model_info={"max_input_tokens": 8192, "max_output_tokens": 2048, "input_cost_per_token": 0, "output_cost_per_token": 0}' \
  --env docker \
  --n-concurrent 4
```

- `--env docker` (the default) runs each task's sandbox as a local Docker
  container — no cloud provider (Daytona/Modal/etc.) or API key needed.
- `model_info` is required by Harbor for any `hosted_vllm/*` model: LiteLLM
  has no built-in pricing/context-length entry for arbitrary local models, so
  token limits and cost-per-token (0 here) must be supplied explicitly.
- Extra flags after `./run_eval.sh` are passed straight through to
  `harbor run`, e.g. `./run_eval.sh --n-tasks 5` for a quick smoke test, or
  `--include-task-name 'pandas-*'` to filter.

## 4. Results

- Aggregate pass/fail + metrics: `jobs/<job_name>/result.json`.
- Per-task trial directories (including the full agent transcript) live under
  `jobs/<job_name>/`.
- Browse trajectories in a UI: `harbor view jobs`.

`tasks/` and `jobs/` are gitignored — the task set is an external clone and
job outputs are run artifacts, not source.
