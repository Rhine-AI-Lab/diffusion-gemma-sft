# DiffusionGemma eval results

Baseline evaluations of the **base** DiffusionGemma model
(`google/diffusiongemma-26B-A4B-it`, no adapter) produced by the
`DiffGemma/diffgemma_trl/sft_eval_*.py` scripts.

## Base model — 2026-06-26

Backend `local` (in-process diffusion `generate()`) · no LoRA adapter ·
`batch_size=16` · `max_new_tokens=256` · `max_denoising_steps=64` · `seed=0` ·
CSCS Clariden `debug` partition (single GH200), container `cscs/gdsdv2-pt.toml`.

| Dataset | Metric | Score | Correct / Total |
|---|---|---|---|
| Countdown (cd3) | exact-match | **50.4%** | 129 / 256 |
| HumanEval | pass@1 | **49.4%** | 81 / 164 |
| MBPP | pass@1 | **32.6%** | 163 / 500 |
| MATH-500 | exact-match | **10.4%** | 52 / 500 |

Datasets: Countdown = committed `dataset/countdown_cd3_test.jsonl`;
MATH-500 = `HuggingFaceH4/MATH-500` (test); MBPP = `google-research-datasets/mbpp`
(`full`, test); HumanEval = `openai/openai_humaneval` (test).

### Reproduce

```bash
# one debug job per dataset (QOS allows one debug job at a time):
sbatch --job-name=eval-countdown --export=ALL,DS=countdown,BATCH=16 cscs/eval_base.sbatch
sbatch --job-name=eval-humaneval --export=ALL,DS=humaneval,BATCH=16 cscs/eval_base.sbatch
sbatch --job-name=eval-math      --export=ALL,DS=math,BATCH=16      cscs/eval_base.sbatch
sbatch --job-name=eval-mbpp      --export=ALL,DS=mbpp,BATCH=16      cscs/eval_base.sbatch
```

Raw per-dataset logs: `eval_out/<dataset>_base.txt`.

### Notes / caveats
- Generation runs ~6.8–8.0 s/example with the eager MoE kernel
  (`experts_implementation=eager`, required on GH200 — the `grouped_mm` kernel
  device-asserts). Each 500-example run is ~60–65 min, within the 1:30 debug cap.
- MBPP/HumanEval are scored by executing the generated code against the task tests
  via `gdsd/utils/local_sandbox.py` (best-effort, **not** a true security sandbox;
  run in the project venv only). pass@1 = a task counts only if it passes ALL its
  tests. Some misses are formatting (no fenced ```python block) rather than
  capability — these are untuned base-model numbers.
- MATH-500 is scored with `math_verify` (boxed-string fallback); Countdown reuses
  the training reward helpers in `rewards.py` (validate/evaluate the expression).
- These are single-run (`seed=0`), greedy-ish settings; expect a few points of
  run-to-run variance on the stochastic diffusion sampler.

### Math-benchmark sweep (accuracy %; AIME = avg@k mean±std)

| dataset | g256/1c | g512/1c | g512/2c | g1024/1c | g1024/2c | g2048/1c | g2048/2c |
| --- | --- | --- | --- | --- | --- | --- | --- |
| math500 | 16.8 | 61.2 | 63.4 | 63.6 | 85.0 | 45.6 | 64.4 |
| minerva | 15.4 | 37.9 | 38.6 | 35.7 | 45.2 | 27.2 | 36.4 |
| olympiad | 3.7 | 27.6 | 27.2 | 32.0 | 57.7 | 15.6 | 32.3 |
| amc | 2.5 | 37.5 | 37.5 | 37.5 | 70.0 | 17.5 | 32.5 |
| aime24 | 0.4±1.1 | 2.9±2.6 | 2.5±2.2 | 4.2±1.4 | 32.1±3.3 | 1.2±1.6 | 3.8±1.1 |
| aime25 | 1.2±1.6 | 3.3±0.0 | 5.0±1.7 | 4.2±2.2 | 30.4±4.5 | 0.4±1.1 | 5.0±3.3 |

### Math-benchmark sweep (accuracy %; AIME = avg@k mean±std)

| dataset | g256/1c | g512/1c | g512/2c | g1024/1c | g1024/2c | g1024/4c | g2048/1c | g2048/2c | g2048/4c | g2048/8c |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| math500 | 16.8 | 61.2 | 63.4 | 63.6 | 85.0 | 87.6 | 45.6 | 64.4 | 90.2 | 92.8 |
| minerva | 15.4 | 37.9 | 38.6 | 35.7 | 45.2 | 47.4 | 27.2 | 36.4 | 46.0 | 44.1 |
| olympiad | 3.7 | 27.6 | 27.2 | 32.0 | 57.7 | 59.5 | 15.6 | 32.3 | 70.2 | 71.1 |
| amc | 2.5 | 37.5 | 37.5 | 37.5 | 70.0 | 70.0 | 17.5 | 32.5 | 95.0 | 87.5 |
| aime24 | 0.4±1.1 | 2.9±2.6 | 2.5±2.2 | 4.2±1.4 | 32.1±3.3 | 32.1±5.0 | 1.2±1.6 | 3.8±1.1 | 52.5±8.5 | 57.5±4.6 |
| aime25 | 1.2±1.6 | 3.3±0.0 | 5.0±1.7 | 4.2±2.2 | 30.4±4.5 | 30.8±2.2 | 0.4±1.1 | 5.0±3.3 | 44.6±3.3 | 45.0±3.3 |

### Math-benchmark sweep (accuracy %; AIME = avg@k mean±std)

| dataset | g256/1c | g512/1c | g512/2c | g1024/1c | g1024/2c | g1024/4c | g2048/1c | g2048/2c | g2048/4c | g2048/8c | g4096/16c |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| math500 | 16.8 | 61.2 | 63.4 | 63.6 | 85.0 | 87.6 | 45.6 | 64.4 | 90.2 | 92.8 | 92.6 |
| minerva | 15.4 | 37.9 | 38.6 | 35.7 | 45.2 | 47.4 | 27.2 | 36.4 | 46.0 | 44.1 | 44.5 |
| olympiad | 3.7 | 27.6 | 27.2 | 32.0 | 57.7 | 59.5 | 15.6 | 32.3 | 70.2 | 71.1 | 72.1 |
| amc | 2.5 | 37.5 | 37.5 | 37.5 | 70.0 | 70.0 | 17.5 | 32.5 | 95.0 | 87.5 | 95.0 |
| aime24 | 0.4±1.1 | 2.9±2.6 | 2.5±2.2 | 4.2±1.4 | 32.1±3.3 | 32.1±5.0 | 1.2±1.6 | 3.8±1.1 | 52.5±8.5 | 57.5±4.6 | 64.2±5.7 |
| aime25 | 1.2±1.6 | 3.3±0.0 | 5.0±1.7 | 4.2±2.2 | 30.4±4.5 | 30.8±2.2 | 0.4±1.1 | 5.0±3.3 | 44.6±3.3 | 45.0±3.3 | 49.2±4.6 |
