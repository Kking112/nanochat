# Reasoning phase design

Approved scope: add `reason_sft` and `reason_rl` after the existing SFT stage,
starting from `sft/d24`, step 486. The existing model, tokenizer, engine,
pretraining, `chat_sft`, and `chat_rl` remain byte-identical to `ac2aecf`.
No web UI, reasoning-gym, learned process reward, or upstream integration.

## Format and rewards

Assistant traces use ordinary text `<think>...</think>` followed by a short
answer and exactly one `#### answer` line. Think tags are not special tokens.
One nonempty, unnested, closed block earns half of format credit; an unambiguous
task-parsable marker after that block earns the other half. Calculator calls
retain the existing tool tokens and loss masks.

Binary correctness alone chooses the reward band:

```
s = clip01(.15 * format + .45 * answer_partial + .40 * process - penalties)
q = clip01(.50 * format + .50 * process - penalties)
total = .70 + .30 * q if correct else .50 * s
```

Only s and q are clipped. Binary mode returns `float(correct)`. Band boundaries
and normalized, nonnegative in-band weights are configurable. Process weight
anneals linearly to .1 in both bands, transferring weight to partial answer in
the incorrect band and format in the correct band. Exact normalized numeric
matching applies to GSM8K and chains; sorting gets per-position credit, and
Countdown closeness requires a legal expression using the supplied numbers.

Process quality weights verified reference-intermediate coverage .9 and final
result consistency .1, normalized over available terms. Repeated intermediate
values count once. Correct unrelated equations give no coverage. Tasks lacking
reference intermediates report availability and use neutral quality. Inject the
existing calculator rather than importing Torch into the reward library.
Missing-reference neutral quality is .5, so a correct, well-formatted sorting
answer scores .925 at the initial shaped weights; it is not artificially given
verified-work credit. Exact rational normalization is used for answers. Only
calculator-statement verification permits minimal floating-point roundoff.
Penalties: .2 times false checked-statement fraction; up to .1 for missing normal
termination ramped over the final 128 budget tokens; up to .1 for repeated word
four-grams above .2; .1 for text after the final answer line.

## Data and training

Use the repository parquet loader for GSM8K, MetaMathQA and OpenMathInstruct-2
(`train_1M`, numeric answers, GSM8K/MATH source families). Inspect shards and
record schema, examples, revision, and license before adapter implementation.
Retain reference intermediates outside rendered messages. Procedural chains,
sorting and Countdown are deterministic by task, split, seed, index, difficulty.

Render without truncation; reject conversations above 1536 tokens or assistant
turns above 1024 including tools. Cache retained indices with content/conversion,
tokenizer and budget fingerprints, locking, and atomic replacement. Reserve
validation before oversampling; exclude normalized exact evaluation-question
overlap, including original-question fields. This cannot rule out semantic
contamination. Apply caps after conversion and filtering: 25k MetaMath, 25k
OpenMath, four GSM8K passes, 15k procedural. Unmodified SmolTalk:MMLU×3 replay
supplies 25% of final rows.

SFT copies existing packing, masks, validation BPB and ChatCORE hooks, uses a
fresh optimizer at .3 LR fraction, checkpoint context length, explicit conservative
batches, seeds and distinct required output tags. RL copies mean-only advantages
and token-level loss normalization, supporting partial microbatches and skipping
constant-reward groups. Globally inactive steps skip optimization; inactive ranks
participate with zero gradients when others are active. Multi-GPU remains untested.

An adapter around `Engine.generate` preserves real assistant-end targets, tool
masks, completion length and stop reason; ignores post-finish emissions; enforces
prompt-plus-completion context length. Default rollout budget is 768. Half of
prompts are GSM8K, the rest split equally across procedural tasks. Difficulty
starts at 1, stays within 1–5, and changes after each task's 100-prompt window:
advance above .7 binary accuracy, decrease below .2, then clear the window.
Synchronize and checkpoint curriculum state.

## Evaluation and release gates

Report binary greedy pass@1, sampled pass@8, token counts, truncation, task
breakdowns, uncertainty, timing, and configuration. Reward totals are training
diagnostics only. Reasoning evaluation defaults to 1024 tokens across all arms;
preserve old `chat_eval` API defaults. CLI dims streamed think content only in
interactive terminals and keeps raw conversation tokens.

M0 baseline/design; M1 format/rewards; M2 tasks/data; M3 SFT; M4 RL; M5 evaluator,
CLI, runner/docs; M6 approved experiments. Each implementation commit requires
`uv run --no-sync python -m pytest -m "not slow"`; stage explicit files, use scoped
Conventional Commits, and keep checkpoint refactoring separate. Tag completed
milestones and push only the new branch/tags. Open a draft PR to the fork after
M1; do not merge, force-push or rewrite local commits.

Smoke gates: 50 SFT iterations with finite downward-trending loss and a sampled
closed think block; five RL steps × four prompts × eight completions, finite
gradients, component logs, checkpoint and zero-variance checks. Measure throughput,
peak VRAM, evaluation cost and campaign duration before asking for the budget.
Full experiments await the user's budget decision; any run above about six GPU
hours needs separate approval. Stop on unexpected baseline-test failures,
unclear licensing, or measured cold-start regression.

Campaign: A original SFT plus unchanged chat_rl (one seed); B reasoning SFT;
C/D shared B plus binary/shaped reasoning RL (seeds 42,43). Match C/D prompts,
sampling, curriculum rules, seeds and rollout steps. Document A's 256-token
training limit and other differences. Evaluate full GSM8K, procedural held-outs
at difficulties 1/3 and five-task ChatCORE. Report attempted steps separately
from optimizer updates. Inspect 20 deterministic random D completions for
exploitation and publish measured results and limitations, without assuming gains.
