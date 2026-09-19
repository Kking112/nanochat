# Reasoning phase lab notebook

## 2026-09-19 — intake and M0

- Starting HEAD: `ac2aecf`; local predecessor `35b758f`; remote baseline `7c7e49b`.
- `git fetch origin`; `git rev-list --left-right --count HEAD...origin/master`
  returned `2 0`. Created `feature/reasoning-phase` from the current checkout.
  Existing `.omc` changes are unrelated and remain unstaged.
- `uv sync --extra gpu --group dev` installed the development environment.
- `uv run --no-sync python -m pytest -m "not slow"`: **48 passed, 10 skipped**,
  5.16 seconds; log `/tmp/reasoning-baseline-tests.log`.
- `nvidia-smi`: NVIDIA RTX PRO 6000 Blackwell, 97,887 MiB exposed memory,
  driver 595.84, driver CUDA capability 13.2. This detected hardware supersedes
  the generic PC assumption for this experiment.
- Input checkpoint: `~/.cache/nanochat/chatsft_checkpoints/d24/model_000486.pt`.
  Config: 24 layers, 1536 embedding width, 12 heads, 2048 context, 32768 vocabulary.
  Stored validation BPB: .2721627753655721 (historical, not a fresh measurement).
- Fresh tokenizer check: `<think>` = `[60,15918,62]`; `</think>` =
  `[9444,15918,62]`. Both are three ordinary tokens.
- Fresh bounded baseline command (24 examples per task, not full-test metrics):
  `uv run --no-sync python -m scripts.chat_eval -i sft -g d24 -s 486 -x 24 -m 1024`.
  Raw output: `~/.cache/nanochat/reasoning_runs/m0/chatcore24.log`.
- Bounded baseline results: ARC-Easy 15/24 (.625), ARC-Challenge 11/24 (.45833),
  MMLU 9/24 (.375), GSM8K 0/24, HumanEval 2/24 (.08333), ChatCORE .2056.
  These small samples are a smoke baseline, not a claim about full-test accuracy.
- Approved design is in `dev/REASONING_DESIGN.md`. Full experiments remain gated
  on the measured post-smoke budget decision. No full campaign has been launched.

## Milestone ledger

### Checkpoint refactor (after M0 `ed28e72`)

Centralized the existing three source-directory mappings in
`nanochat.checkpoint_manager.CHECKPOINT_SOURCES`, with no source behavior change.
Verification: full required non-slow suite **48 passed, 10 skipped**, 5.17 s
(`/tmp/reasoning-ckpt-tests.log`). New reasoning source entries are a separate
behavior change. Protected implementation files are untouched.

| Milestone | State | Evidence |
|---|---|---|
| M0 | complete | 48 passed, 10 skipped; fresh bounded five-task baseline above |
| M1 | complete | 73 focused tests; 124 passed, 10 skipped in integrated suite |
| M2 | in progress | Dataset provenance inspection and fixture tests |
| M3 | pending | SFT implementation and 50-step smoke |
| M4 | in progress | Generation adapter, RL implementation and fixture tests |
| M5 | pending | CLI, evaluator, runner and documentation |
| M6 | budget approval pending | No experimental improvement claim |

## M1 — format and reward verification

Parent checkpoint refactor: `689ba9e`. Added Torch-independent parsing, exact
numeric normalization, tool-aware arithmetic verification, repetition counting,
validated reward configuration, separated correctness bands, process annealing,
and full component diagnostics. Updated brief section 5 to replace its old
additive formula; the approved design specifies neutral unavailable-process
quality rather than fabricating intermediate verification.

Commands: `uv run --no-sync python -m pytest tests/test_reasoning.py
tests/test_rewards.py -q` (**73 passed**); required full command
`uv run --no-sync python -m pytest -m "not slow"` (**124 passed, 10 skipped**,
6.10 s; `/tmp/reasoning-m1-tests.log`). Randomized tests cover both correctness
bands even with malformed responses and misleading partial-answer scores;
fixtures cover equation spam, duplicates, false arithmetic, stop-budget ramps,
trailing text and annealing endpoints. No training or improvement claim yet.
