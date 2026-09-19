# Agent brief: add a CoT reasoning phase to nanochat

Repo: `https://github.com/Kking112/nanochat` (fork of `karpathy/nanochat`, default branch `master`).
Owner's hardware: one RTX Pro 6000 Blackwell (96 GB), Ubuntu 24.04, `uv`. Default every command to single GPU; keep `torchrun` multi-GPU paths working but untested is acceptable.

## 1. Mission

Add two new training stages that run after the existing pipeline:

```
base_train -> chat_sft -> reason_sft (cold start on CoT traces) -> reason_rl (shaped-reward RL)
```

`reason_sft` teaches the output format and a prior over step-by-step reasoning. `reason_rl` then optimizes against a multi-component reward that gives credit for format, verified intermediate work, and partially correct answers, not only 1/0 on the final answer.

Non-goals: no changes to the model architecture, the tokenizer, pretraining, or the existing `chat_sft.py` / `chat_rl.py` behavior. No new heavyweight frameworks (no TRL, verl, HF `datasets`, vLLM inside the repo).

## 2. Facts about the repo (verified 2026-09-19 at commit 7c7e49b; re-verify before relying on them)

- `tasks/common.py` defines `Task`, `TaskMixture`, `TaskSequence`, and `load_hub_dataset(repo_id, subset, split)`, a dependency-free loader that pulls the Hub's auto-generated parquet export with `urllib` + `pyarrow`. Use it for every new dataset. Do not add HF `datasets`.
- A Task yields `{"messages": [...]}` conversations. Assistant content is a string or a list of parts (`text`, `python`, `python_output`). Tasks used for RL implement `evaluate(conversation, response) -> int` and `reward(conversation, response) -> float`. `tasks/gsm8k.py` is the reference; its `reward()` is currently just `float(evaluate())`.
- GSM8K gold solutions contain calculator annotations `<<expr=result>>`. `gsm8k.py` converts them to `python` / `python_output` parts. These annotations are the source of verified intermediate values for the process reward (section 5).
- `scripts/chat_rl.py` is hardcoded to GSM8K, loads from source `"sft"`, samples with `Engine.generate_batch`, computes `advantage = reward - mean(reward)` per prompt group (no std normalization, no KL, no clipping, token-level normalization), default `--max-new-tokens 256`.
- `scripts/chat_sft.py` is hardcoded to load from source `"base"` and builds its mixture inline (SmolTalk + MMLU x3 + GSM8K x4). It bin-packs conversations to `max_seq_len` (2048 inherited from pretraining).
- `nanochat/checkpoint_manager.py::load_model` and `load_optimizer_state` each map `base|sft|rl` to a checkpoint dir via a literal dict. `chat_eval.py` and `chat_cli.py` take `-i/--source`.
- Context length is 2048 (`GPTConfig.sequence_len`); `render_conversation` truncates at 2048. Prompt + reasoning + answer must fit.
- `vocab_size` is 32768, already a multiple of 64, so there are no spare padded embedding rows, and the special-token list is baked into the pickled tokenizer. New special tokens would require retraining tokenizer and base model. Therefore the think delimiters are plain text.
- The Engine's calculator tool state machine triggers on `<|python_start|>` anywhere in the assistant turn and returns mask 0 for forced tool-output tokens, so tool calls inside the reasoning block work and are correctly excluded from the loss.
- Tests live in `tests/`, run with `pytest`; a `slow` marker exists.
- The owner regularly merges `karpathy/master`. Prefer new files over edits to existing files so those merges stay clean.

## 3. Fixed design decisions

1. **Output format.** Plain-text tags, final answer marker reused from GSM8K so the existing extractor and `chat_eval` keep working:
   ```
   <think>
   ...reasoning, may include calculator tool calls...
   </think>
   ...short user-facing answer...
   #### 42
   ```
   Put all format constants and parsing in one module (`nanochat/reasoning.py`). Log how `<think>` and `</think>` tokenize; if either splits into more than ~4 tokens, report it but do not change the tokenizer.
2. **New scripts, not flags on old ones.** Create `scripts/reason_sft.py` and `scripts/reason_rl.py` by copy-and-adapt. Leave `chat_sft.py` and `chat_rl.py` byte-identical so the binary-reward baseline stays reproducible.
3. **Checkpoint sources.** Add `"reason_sft": "reasonsft_checkpoints"` and `"reason_rl": "reasonrl_checkpoints"` to both dicts in `checkpoint_manager.py` (refactor into one module-level dict while there). Update the `--source` help strings. This is the only required edit to existing library code.
4. **Anti-forgetting replay.** The cold-start mixture includes a replay slice of the original SFT mixture (start at ~25% of rows from SmolTalk + MMLU) and uses a lower peak LR than `chat_sft` (start at `--init-lr-frac 0.3`, no optimizer warm start from the SFT checkpoint). Replay rows are left unmodified (no think block).
5. **Length budget.** Filter every cold-start example with the nanochat tokenizer: full rendered conversation <= 1536 tokens and assistant turn <= 1024 tokens. Drop, never truncate, reasoning traces. RL default `--max-new-tokens 768`.
6. **Advantage.** Keep mean-only centering. Do not add std normalization: with shaped rewards, z-scoring inflates tiny shaping differences inside all-wrong groups into full-size gradients.
7. **Metrics.** Shaped reward is a training signal only. Every reported result uses binary correctness (`evaluate`), plus response length and truncation rate.

## 4. Git workflow (mandatory)

```bash
git remote add upstream https://github.com/karpathy/nanochat.git   # if missing
git fetch origin && git checkout master && git pull --ff-only
git checkout -b feature/reasoning-phase
```

- Never commit to `master`. Never force-push. Never rewrite pushed history. Do not merge or rebase `upstream` into the feature branch during this work; if upstream changes block you, stop and report.
- One logical change per commit, each leaving `pytest -m "not slow"` green. Conventional prefixes scoped to the area: `feat(reasoning): ...`, `feat(tasks): ...`, `test(reward): ...`, `docs: ...`, `refactor(ckpt): ...`. Body explains why, not what.
- Commit order inside a milestone: tests and pure-Python modules first, scripts second, docs last. A refactor of existing code is always its own commit, separate from new behavior.
- Tag the end of each milestone: `git tag reasoning-m1` etc. Push branch and tags: `git push -u origin feature/reasoning-phase --tags`.
- Never commit checkpoints, datasets, wandb dirs, or anything under `~/.cache/nanochat`. Check `git status` before every commit; extend `.gitignore` if needed (own commit).
- Risky experiments (e.g. the reasoning-gym adapter, PRM reward) go on sub-branches off the feature branch (`feature/reasoning-phase--rgym`) and merge back with `--no-ff` only once tests pass.
- Open a draft PR from `feature/reasoning-phase` into `Kking112/nanochat:master` (not upstream) after M1 and keep its description updated with a milestone checklist. Do not merge it; the owner reviews.
- Keep `dev/REASONING_LOG.md` as a running lab notebook: date, commit hash, command, result, decision. Every training run you launch gets an entry.

## 5. Reward specification (`nanochat/rewards.py`)

The approved implementation plan replaces the original additive reward with
strictly separated correctness bands. The implementation is pure Python with
`RewardConfig` and `RewardBreakdown` dataclasses; arithmetic checks receive the
existing runtime calculator as a callable.

```
correct = task.evaluate(conversation, response) == 1
s = clip01(0.15*r_format + 0.45*r_answer_partial + 0.40*r_process - penalties)
q = clip01(0.50*r_format + 0.50*r_process - penalties)
total = 0.70 + 0.30*q if correct else 0.50*s
```

Binary correctness alone chooses the band, including malformed correct responses.
Only `s` and `q` are clipped. Binary mode returns `float(correct)`. Configurable
band boundaries must remain separated; nonnegative in-band weights must sum to
one. `--process-weight-final=0.1` linearly anneals each band's process weight,
transferring weight to partial answer for incorrect responses and to format for
correct responses. Diagnostics include correctness, band, `s`, `q`, raw terms,
individual penalties, effective weights and total.

- Format: half credit for one nonempty, unnested, closed think block; half for
  one task-parsable answer marker after the block. Malformed tags and ambiguous
  answer markers receive no corresponding credit.
- Partial answer: exact normalized numeric matching for GSM8K/chains, sorting
  per-position accuracy, and Countdown closeness only for legal expressions using
  the supplied numbers. No numeric-distance reward for word problems.
- Process: 90% verified reference-intermediate coverage and 10% final-result
  consistency, normalized over applicable terms. Count each reference value once;
  unrelated arithmetic earns no credit. Without reference intermediates, report
  unavailable process verification and use neutral quality .5.
- Penalties: .2 times the invalid checked-statement fraction; up to .1 for missing
  normal termination during the final 128 effective-budget tokens; up to .1 for
  repeated word four-grams above a .2 fraction; .1 for text after the answer line.
- Invariant: every incorrect response scores at most .5 and every correct
  response at least .7, regardless of formatting or penalties. Randomized tests
  enforce this for partial-credit tasks as well.

The detailed approved workflow and experiment gates are in
`dev/REASONING_DESIGN.md`; those supersede the older milestone details below
(in particular no web UI, d24 step486 input, and full experiments only after
post-smoke budget approval).

## 6. Milestones

**M0 - Baseline and design (no code changes).** Create the branch. Run `pytest -m "not slow"`. Locate or train a small SFT checkpoint for smoke tests (ask the owner which `model_tag` to use; otherwise train a d12 with `runs/speedrun_singlegpu.sh` settings scaled down). Record baseline `chat_eval -i sft` numbers in `dev/REASONING_LOG.md`. Write `dev/REASONING_DESIGN.md` (one page: format, mixture, reward, eval plan). Commit docs.

**M1 - Format + reward library.** `nanochat/reasoning.py` (constants, `split_think_answer`, `extract_intermediate_values`, n-gram repetition), `nanochat/rewards.py`, `tests/test_reasoning.py`, `tests/test_rewards.py`. Tests must cover: malformed/nested/unclosed tags, multiple answer markers, the dominance invariant over randomized inputs, the equation-spam exploit scoring no better than an empty trace, truncation penalty monotonicity. CPU only. Tag `reasoning-m1`, open draft PR.

**M2 - Tasks.** New files under `tasks/`:
- `gsm8k_reasoning.py`: subclass/wrap GSM8K; `get_example` re-renders the gold solution into the think format keeping the tool-call parts inside the think block; `reward` uses `rewards.py` with gold intermediates.
- `metamath.py` (`meta-math/MetaMathQA`), `openmathinstruct2.py` (`nvidia/OpenMathInstruct-2`, use the `train_1M` split, keep only `problem_source` in gsm8k/math families): convert the short CoT solution into the think format with the final answer on the `####` line; skip rows whose final answer cannot be parsed to a number.
- `procedural.py`: 3 zero-dependency, seeded, difficulty-parameterized generators that each emit a gold trace and a partial-credit scorer: multi-step integer arithmetic chains (gold intermediates), number sorting (per-position credit), Countdown-lite (valid-expression + closeness). `num_examples` is a constructor arg; index seeds the RNG so examples are deterministic.
- A `LengthFiltered(task, tokenizer, max_total, max_assistant)` wrapper that precomputes kept indices once and caches them under the base dir.
- Extend routing so a `TaskMixture` can serve RL: each conversation carries `"task_name"` and the mixture exposes `reward`/`evaluate` that dispatch to the owning task. Put this in a new `RoutedTaskMixture` class rather than changing `TaskMixture`.
- Add cases to `tests/test_tasks.py` style: every new task renders through `tokenizer.render_conversation` with a non-empty loss mask; gold example gets `total == 1.0` from its own reward; print token-length histograms to the log file.
Tag `reasoning-m2`.

**M3 - Cold-start SFT.** `scripts/reason_sft.py` from `chat_sft.py`: load from `--source sft` (default), save to `reasonsft_checkpoints`, mixture per section 3.4 with row caps as CLI args (`--metamath-rows`, `--omi2-rows`, `--gsm8k-epochs`, `--procedural-rows`, `--replay-frac`). Keep the bin-packing loader, bpb eval and ChatCORE hooks. Smoke test: 50 iterations on the small checkpoint, confirm loss falls and a sampled GSM8K completion contains a closed think block. Then one full run; log GSM8K pass@1 before/after plus ChatCORE to confirm no regression beyond noise. Tag `reasoning-m3`.

**M4 - Shaped-reward RL.** `scripts/reason_rl.py` from `chat_rl.py`: load `--source reason_sft`, train on a `RoutedTaskMixture` (GSM8K-reasoning + procedural, weights as CLI args), rewards via `task.reward` returning the component dataclass, log every component and the fraction of zero-variance groups to wandb, generalize the eval loop to any generative task with binary `evaluate` and report per-task pass@k. Skip the backward pass for groups whose rewards are all equal. Add simple difficulty curriculum for procedural tasks: raise difficulty when a task's rolling binary pass rate exceeds 0.7, lower below 0.2. Smoke test: 5 steps with `--examples-per-step 4 --num-samples 8`. Tag `reasoning-m4`.

**M5 - Eval and UX.** `chat_eval` accepts the new sources and a `--max-new-tokens` suited to reasoning (1024); `chat_cli`/web UI render the think block dimmed or collapsible. `runs/reasoning_singlegpu.sh` chains the two new stages and evals. README section + `dev/LOG.md` entry. Tag `reasoning-m5`.

**M6 - Ablation (the deliverable that matters).** Same SFT checkpoint, same RL step budget, 2 seeds each if budget allows:
A) `chat_rl` as-is (binary, no CoT phase); B) `reason_sft` only; C) `reason_sft` + `reason_rl --reward-mode binary`; D) `reason_sft` + `reason_rl --reward-mode shaped`.
Report GSM8K test pass@1 (greedy) and pass@8, held-out procedural seeds at two difficulties, ChatCORE, mean response tokens, truncation rate. Write results to `dev/REASONING_RESULTS.md`. Sample and paste 20 random D-arm completions and inspect them for reward hacking before claiming anything.

**M7 - Optional, separate sub-branches.** (a) `tasks/rgym.py` adapter over `reasoning-gym` as an optional extra in `pyproject.toml` (`[project.optional-dependencies] reasoning`), imported lazily; check task names against the library's `GALLERY.md`; most of its scorers are exact-match, so add per-element scorers for the tasks you include. (b) A learned process reward (e.g. a 7B math PRM served out-of-process) behind `--prm-url`; off by default.

## 7. Rules of engagement

- Read before writing: `tasks/common.py`, `tasks/gsm8k.py`, `scripts/chat_sft.py`, `scripts/chat_rl.py`, `nanochat/engine.py`, `nanochat/tokenizer.py`, `nanochat/checkpoint_manager.py`. If anything in section 2 is no longer true, say so in the log and adapt.
- Match the house style: single-file scripts, argparse, `print0`, minimal abstraction, comments that explain why.
- Do not edit `gpt.py`, `tokenizer.py`, `engine.py`, `base_train.py`, `chat_sft.py`, `chat_rl.py`. If you believe an edit is required, stop and ask with a concrete diff proposal.
- Dataset schemas drift. Before writing a loader, fetch one parquet shard and print column names and two rows; record them in the log.
- Stop and ask when: a full training run would exceed ~6 GPU-hours, a dataset license is unclear, tests you did not write start failing, or results contradict the design (e.g. cold start lowers GSM8K).
- Hand-off at each milestone: commit range, test output, commands run, numbers, open questions.
