# When does SFT actually help DiffusionGemma? Task-selection criteria + candidates

## The pattern, established empirically this session

| Task | Output space | Base model | + SFT | Result |
|---|---|---|---|---|
| Sudoku (9x9 grid) | closed, 1 valid grid | ~0% (easy/med/hard, even w/ 4x token budget — rambles, self-corrects, never converges) | reportedly high % | **clean win** |
| PubMedQA (official hackable_diffusion recipe) | closed, 3-way label (yes/no/maybe) | — | reportedly works | **clean win** (per official recipe) |
| HumanEvalFix (bug-fix) | open, many valid programs | 93.3% | 76.8% | **regression** (-16.5pt) |
| BigCodeBench (code-gen) | open, many valid programs | 46.0% | 42.7-44.7% | **mild regression** (-1.3 to -3.3pt, near noise) |
| CRUXEval (execution tracing) | closed but structurally rich/varied values | ~70% | 57.3-60.7% | **regression** (-9 to -13pt) |
| Restricted function calling (API-Bank-trained, held-out slice) | closed schema, mostly-extractive args | 45.0% | **72.5%** (ck2574/3ep) | **clean win** (+27.5pt) |
| Restricted function calling (BFCL_v3_simple, never trained on) | closed schema, mostly-extractive args, external domain | 85.0% | **90.0%** (ck2574/3ep) | **win** (+5pt), despite near-ceiling base rate |

## Update: restricted function calling is a clean SFT win (2026-07-17)

Built per spec: API-Bank lv1-train filtered to genuinely single-turn examples
(3178 raw API-Request examples -> 858 clean ones after also dropping ~0.3%
with argument names outside the tool's declared schema), converted to
`{"tool": ..., "arguments": {...}}` targets. 60.5% of examples are fully
extractive (every arg value a literal substring of the user's request),
71.2% mean extractiveness -- the rest are bounded, systematic patterns (ID
placeholders like `"user's ID"`, date normalization, default values for
unspecified optional args), not open-ended invention. BFCL (`BFCL_v3_simple`,
loaded via `HfApi().list_repo_files()` same as API-Bank, since
`datasets.load_dataset` 404s on this repo) held out entirely for eval.

Trained 3 epochs (2574 steps, LoRA r=64/alpha=128, decoder-only merge,
encoder_loss_weight=0). Result: **wins on both the training-matched
held-out slice (45.0% -> 72.5%) and the fully external BFCL benchmark
(85.0% -> 90.0%)** -- the first task this session where SFT improved an
external, never-trained-on benchmark rather than regressing it (contrast
CRUXEval, also "closed" but with a rich/varied value space, which
regressed).

This confirms the refined theory: it's not just "closed output space" that
matters, but closed **+ narrow/low-cardinality + mostly extractive**. A
tool call's argument set is bounded by the tool's own schema and largely
copied from the request text, unlike CRUXEval's arbitrary computed values
or code-gen's open-ended style space.

Caveat: an intermediate checkpoint (ck1806, ~2.1 epochs) scored *below*
base on BFCL (81.7%) despite scoring above base on the training-matched
slice (65.0%) -- a non-monotonic dip on the external benchmark mid-training
that recovered by the final epoch. Single-seed, n=60/40 samples, so partly
attributable to sampling noise and the model's inherent temp=0
non-determinism (unseeded re-noising in vLLM's diffusion sampling step) --
but if real, suggests the transfer benefit isn't guaranteed monotonic in
training dosage the way the training-matched score is.

## Why: output entropy, not "does the base model know how"

Watched directly on sudoku: the base model isn't reasoning-limited so much as
**discipline-limited** — given a puzzle, it drafts an answer, second-guesses
itself ("Wait, let me re-solve carefully..."), restarts, and never converges
to a stable terse output even with a 4x token budget (`finish_reason:
length`, still mid self-correction). The failure mode is "won't commit
cleanly," not "can't solve."

Sudoku and PubMedQA share: a **closed, low-entropy output space** — exactly
one correct grid, or exactly one correct label out of 3. There's no
"legitimate diversity" for SFT to compete with, so training the model to
commit to the one correct format/answer is purely additive.

Code generation is the opposite: an effectively unbounded, high-entropy
space of *equally valid* programs. Training on one `canonical_solution` per
problem doesn't teach "commit to the right answer" (there isn't one) — it
teaches "imitate this one style," which competes with and suppresses the
model's own diverse, valid, pretrained strategies. That's the forgetting
mechanism, and it explains why the regression was much worse on
HumanEvalFix (small, likely near-memorized canonical solutions -- a fragile,
precise retrieval pathway) than on BigCodeBench (larger, more novel problems
requiring more robust, distributed reasoning).

**Working rule for picking the next task:** prefer tasks with a closed,
verifiable, low-entropy output space where the suspected failure mode is
commitment/formatting rather than missing reasoning. Avoid open-ended
generation tasks (there, SFT risks narrowing away valid capability) unless
paired with a much larger, diverse, ideally on-policy training set (per the
sudoku scale argument -- 2.87GB / probably near-zero epoch-repetition).

## Candidate tasks, by category

### Already have plumbing in this repo (cheapest to test)
- **GSM8K** (`GSM8K_PROMPT`, `dataset/make_gsm8k_jsonl.py`) — final numeric
  answer in `<answer>` tags is closed/single-target even though the
  reasoning trace leading there is free-form. Worth a clean base-vs-SFT
  check using the same lens (does the base model fail to *commit* to a
  final boxed answer, or does it get the math wrong?).
- **Countdown** (`COUNTDOWN_PROMPT`) — must reach an exact target value;
  binary pass/fail, closed criterion, some benign flexibility in which
  expression gets there (unlike code, doesn't reward a specific *style*).
- **Sudoku itself** — re-run with a properly-scaled-down training set (per
  the discussion) to isolate "closed output space" from "training set size"
  as the active ingredient.

### Multiple-choice / classification QA (PubMedQA-shaped)
- **BoolQ** — yes/no reading comprehension, structurally identical to
  PubMedQA.
- **MMLU / ARC / HellaSwag / PIQA / WinoGrande** — 4-way multiple choice;
  caveat: strong base models are often already high here (ceiling-effect
  risk like HumanEvalFix) -- worth a quick base-rate probe before committing.
- **NLI (SNLI/MNLI)** — 3-way entailment/contradiction/neutral, same shape
  as PubMedQA.
- Other closed-label domain QA (MedQA, MedMCQA) if domain-specific
  calibration is of interest.

### Other constraint-satisfaction / logic puzzles (sudoku-shaped)
- N-Queens, KenKen, Kakuro, Nonograms — closed, unique-solution puzzles,
  easily and near-infinitely synthesizable (matches sudoku's scale
  advantage).
- Cryptarithmetic (e.g. SEND+MORE=MONEY-style).
- Graph coloring / small CSPs with a unique canonical solution.

### Structured extraction (closed schema, not open generation)
- NER / relation extraction into a fixed schema.
- Information extraction into a fixed JSON schema (closed keys/types --
  contrast with open-ended code, this has a single correct structured
  answer per input).
- Deterministic format canonicalization (date/unit normalization, fixed
  serialization).

### Deterministic transformations (single correct output, likely a
"discipline" failure mode for a token-based model)
- Base64/ROT13/simple cipher encode-decode.
- Exact arithmetic on structured inputs (matrix ops, given a fixed
  procedure).

## Update: CRUXEval (execution tracing) also regressed (2026-07-17)

Tested: base model ~70% (corrected for an int-vs-quoted-string scoring
artifact) vs SFT (trained on 2,326 MBPP-execution-derived examples,
`build_cruxeval_data.py`) at 57.3% (ckpt150, ~0.5 epoch) and 60.7%
(ckpt450, ~1.5 epoch) -- both **regressions**, consistent with BigCodeBench/
HumanEvalFix, not with sudoku/PubMedQA.

**Refines the theory:** "closed" (exactly one correct answer) is necessary
but not sufficient for a clean SFT win. CRUXEval's outputs, while uniquely
determined, are structurally rich/varied (arbitrary ints, strings, nested
lists/dicts -- effectively unbounded value space), unlike sudoku's fixed
81-digit grid or PubMedQA's 3-way label. The revised rule: SFT reliably
helps when the output space is closed **and** structurally simple/low
cardinality; it's still risky when the answer, though unique, is drawn from
a rich/varied structural space -- likely because training on a narrow,
off-policy example set still pulls the model toward that set's specific
value/formatting patterns rather than the full breadth of the true (varied)
target space.

## Update: planted k-coloring pilot collapsed entirely (2026-07-17)

Built a synthetic planted-k-coloring generator per user spec: plant k nonempty
color classes + a k-clique across one representative per class (proves
chi(G)=k exactly, no solver needed), permute vertex IDs, output a bare
digit-string coloring (one char per vertex, matching sudoku's grid
convention) of length n (n in 16-64). Base model: 25.0% valid colorings on a
40-example probe, but **all** successes were n=16 -- 0% at n>=24, a clean
headroom signal (real difficulty scaling with n, not a ceiling effect).

Pilot SFT (3000 examples, 3 epochs, same LoRA recipe as every other
experiment this session) **collapsed completely** rather than improving.
Checked 4 checkpoints across the run (0.4, 0.7, 1.0, 2.5 epochs) plus the
final one: every one failed (0/40 valid), and the failure mode didn't
converge toward correctness -- it drifted through increasingly degenerate
attractors (never-terminates -> terminates instantly -> structured-prefix-
then-repeats -> by epoch 2.5 onward, 39/40 outputs are a single repeated
digit ("5555...") totally independent of the input graph). Final
train_loss=0.71, well above every other successful run this session
(API-Bank 0.23, CRUXEval ~0.32) -- the loss curve itself shows the model
never actually fit the task.

Ruled out as causes: training-data correctness (every one of the 3000
completions independently re-verified as a valid, correctly-lengthed
coloring), tokenization (digit strings are clean 1-char-1-token, no BPE
merging), and the collator's EOS/variable-length handling (structurally
identical to the path that worked fine for API-Bank's variable-length JSON
completions).

**Leading hypothesis:** this task requires a skill none of the other SFT
tasks needed -- a *content-dependent, variable* completion length with *no
syntactic stop cue*. Sudoku's grid is always exactly 81 digits (a fixed
positional pattern, no counting required). JSON/code completions have
syntactic closure (a closing brace, structural end). A bare digit string of
length n has neither: the model must read n stated in natural language near
the *start* of a prompt that can run to ~2000 tokens, then count out exactly
that many digits with no local cue at the generation point. That combination
(long-range dependency + no syntactic terminator + high LR straight through
most of training) looks like what triggered the collapse into a fixed
low-loss-on-average, high-frequency single-token attractor -- a classic
mode-collapse signature, not a data or pipeline bug.

**Not yet tried:** lower peak LR (5e-5 instead of 1.5e-4), and/or repeating
n immediately before the generation point instead of only once near the top
of the prompt (turns the long-range length dependency into a local one).
Neither has been tested -- this is an open task, not a closed negative
result yet, since the specific fix hasn't been attempted.

**Methodological lesson for the output-entropy framework:** "closed +
narrow + low per-token cardinality" (this task: each digit in {0..k-1},
k<=6) is necessary but **not sufficient** -- an orthogonal axis matters too:
whether the completion's *length* is fixed/positional or
variable/content-dependent. Sudoku and PubMedQA both happen to have fixed
output length; function-calling's JSON has variable length but a syntactic
terminator. This is the first task tested with variable length AND no
syntactic terminator, and it's also the first pilot to collapse outright
rather than merely underperform.

## Suggested next step

Quick, cheap triage before committing to a full training run: for each
candidate, run a small (~30-50 example) *base-model-only* probe first (like
we did for sudoku just now) to confirm (a) the base model's accuracy isn't
already near-ceiling, and (b) the actual failure mode looks like
"rambling/non-commitment" rather than genuine reasoning failure -- both are
cheap to check and would have caught the HumanEvalFix ceiling problem and
the sudoku-shape confirmation immediately, before any GPU-hours were spent
training.
