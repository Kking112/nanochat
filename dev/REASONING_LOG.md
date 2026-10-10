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
| M2 | complete | 40 offline task tests; integrated suite 190 passed, 10 skipped |
| M3 | complete (smoke) | 50 steps, downward finite loss, reload and 18 closed think samples |
| M4 | GPU smoke deferred | Runtime/RL implemented; CPU integration verified |
| M5 | complete | Evaluator, streamed CLI, guarded runner, docs and CPU acceptance |
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
# Downloaded dataset inspection (2026-09-19)

## openai/gsm8k

```json
{
  "repo": "openai/gsm8k",
  "subset": "main",
  "split": "train",
  "revision": "740312add88f781978c0658806c59bc2815b9866",
  "license": [
    "mit"
  ],
  "columns": [
    "question",
    "answer"
  ],
  "row_count_first_shard": 7473,
  "shard_url": "https://huggingface.co/api/datasets/openai/gsm8k/parquet/main/train/0.parquet",
  "shard_count": 1,
  "rows": [
    {
      "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
      "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
    },
    {
      "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
      "answer": "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10"
    }
  ]
}
```

## meta-math/MetaMathQA

```json
{
  "repo": "meta-math/MetaMathQA",
  "subset": "default",
  "split": "train",
  "revision": "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18",
  "license": "mit",
  "columns": [
    "type",
    "query",
    "original_question",
    "response"
  ],
  "row_count_first_shard": 395000,
  "shard_url": "https://huggingface.co/api/datasets/meta-math/MetaMathQA/parquet/default/train/0.parquet",
  "shard_count": 1,
  "rows": [
    {
      "type": "MATH_AnsAug",
      "query": "Gracie and Joe are choosing numbers on the complex plane. Joe chooses the point $1+2i$. Gracie chooses $-1+i$. How far apart are Gracie and Joe's points?",
      "original_question": "Gracie and Joe are choosing numbers on the complex plane. Joe chooses the point $1+2i$. Gracie chooses $-1+i$. How far apart are Gracie and Joe's points?",
      "response": "The distance between two points $(x_1,y_1)$ and $(x_2,y_2)$ in the complex plane is given by the formula $\\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}$.\nIn this case, Joe's point is $(1,2)$ and Gracie's point is $(-1,1)$.\nSo the distance between their points is $\\sqrt{((-1)-(1))^2+((1)-(2))^2}=\\sqrt{(-2)^2+(-1)^2}=\\sqrt{4+1}=\\sqrt{5}$.\nTherefore, Gracie and Joe's points are $\\boxed{\\sqrt{5}}$ units apart.\nThe answer is: \\sqrt{5}"
    },
    {
      "type": "GSM_Rephrased",
      "query": "What is the total cost of purchasing equipment for all sixteen players on the football team, considering that each player requires a $25 jersey, a $15.20 pair of shorts, and a pair of socks priced at $6.80?",
      "original_question": "The treasurer of a football team must buy equipment for the sixteen players of his team. Each piece of equipment consists of a $25 jersey, a $15.20 pair of shorts and a pair of socks at $6.80. How much does all the equipment of all the players on the team cost?",
      "response": "Each player requires a $25 jersey, a $15.20 pair of shorts, and a pair of socks priced at $6.80.\nSo the total cost for each player is $25 + $15.20 + $6.80 = $47.\nSince there are sixteen players on the football team, the total cost for all of them is 16 * $47 = $752.\n#### 752\nThe answer is: 752"
    }
  ]
}
```

## nvidia/OpenMathInstruct-2

```json
{
  "repo": "nvidia/OpenMathInstruct-2",
  "subset": "default",
  "split": "train_1M",
  "revision": "469216e3f46f4dacf476b382e192485ea51a143e",
  "license": "cc-by-4.0",
  "columns": [
    "problem",
    "generated_solution",
    "expected_answer",
    "problem_source"
  ],
  "row_count_first_shard": 333334,
  "shard_url": "https://huggingface.co/api/datasets/nvidia/OpenMathInstruct-2/parquet/default/train_1M/0.parquet",
  "shard_count": 3,
  "rows": [
    {
      "problem": "Solve for $y$:\n\n$$\\frac{y^2 - 3y + 2}{y - 2} = y + 1$$",
      "generated_solution": "Start by multiplying both sides by $y - 2$ to eliminate the denominator:\n\\[ (y^2 - 3y + 2) = (y + 1)(y - 2) \\]\n\nExpand both sides:\n\\[ y^2 - 3y + 2 = y^2 - y - 2 \\]\n\nSubtract $y^2$ from both sides to get:\n\\[ -3y + 2 = -y - 2 \\]\n\nAdd $3y$ to both sides:\n\\[ 2 = 2y - 2 \\]\n\nAdd $2$ to both sides:\n\\[ 4 = 2y \\]\n\nDivide by $2$ to solve for $y$:\n\\[ y = \\frac{4}{2} \\]\n\n\\[ y = \\boxed{2} \\]",
      "expected_answer": "2",
      "problem_source": "augmented_math"
    },
    {
      "problem": "Given a circle centered at the origin, a point $A$ is translated along the circle to a new position $A'$ by rotating the circle about its center by $\\frac{\\pi}{4}$ radians. If $A$ has coordinates $(3,4)$, find the coordinates of $A'$.\n\nExpress your answer as $(a,b)$ with $a$ and $b$ integers.",
      "generated_solution": "The rotation matrix for rotating a point $(x, y)$ by an angle $\\theta$ counterclockwise is\n\\[ \\begin{pmatrix} \\cos \\theta & -\\sin \\theta \\\\ \\sin \\theta & \\cos \\theta \\end{pmatrix} \\begin{pmatrix} x \\\\ y \\end{pmatrix} = \\begin{pmatrix} x \\cos \\theta - y \\sin \\theta \\\\ x \\sin \\theta + y \\cos \\theta \\end{pmatrix} \\]\n\nIn this case, $\\theta = \\frac{\\pi}{4}$ and the point $A$ is $(3, 4)$.\n\n\\[ \\begin{pmatrix} 3 \\cos \\frac{\\pi}{4} - 4 \\sin \\frac{\\pi}{4} \\\\ 3 \\sin \\frac{\\pi}{4} + 4 \\cos \\frac{\\pi}{4} \\end{pmatrix} = \\begin{pmatrix} 3 \\cdot \\frac{1}{\\sqrt{2}} - 4 \\cdot \\frac{1}{\\sqrt{2}} \\\\ 3 \\cdot \\frac{1}{\\sqrt{2}} + 4 \\cdot \\frac{1}{\\sqrt{2}} \\end{pmatrix} = \\begin{pmatrix} \\frac{3}{\\sqrt{2}} - \\frac{4}{\\sqrt{2}} \\\\ \\frac{3}{\\sqrt{2}} + \\frac{4}{\\sqrt{2}} \\end{pmatrix} = \\begin{pmatrix} \\frac{-1}{\\sqrt{2}} \\\\ \\frac{7}{\\sqrt{2}} \\end{pmatrix} \\]\n\nTo rationalize the coordinates, we multiply the numerator and denominator by $\\sqrt{2}$:\n\\[ A' = \\begin{pmatrix} \\frac{-1 \\cdot \\sqrt{2}}{\\sqrt{2} \\cdot \\sqrt{2}} \\\\ \\frac{7 \\cdot \\sqrt{2}}{\\sqrt{2} \\cdot \\sqrt{2}} \\end{pmatrix} = \\begin{pmatrix} \\frac{-\\sqrt{2}}{2} \\\\ \\frac{7\\sqrt{2}}{2} \\end{pmatrix} \\]\n\n\\[ \\boxed{(-\\frac{\\sqrt{2}}{2}, \\frac{7\\sqrt{2}}{2})} \\]",
      "expected_answer": "(-\\frac{\\sqrt{2}}{2}, \\frac{7\\sqrt{2}}{2})",
      "problem_source": "augmented_math"
    }
  ]
}
```

## M2 — datasets, filtering and routing

M1 commit `54910a5`, tags `reasoning-m0` and `reasoning-m1` were pushed; draft PR:
https://github.com/Kking112/nanochat/pull/1 (fork master, no merge).
Dataset inspection above preceded adapters; downloaded parquet rows establish
schemas and source cards establish observed license/revision. The existing loader
retrieves current exports; cache fingerprints hash actual content, not only the
observed source revision. Future dataset updates therefore invalidate caches.

Implemented numeric-only MetaMath/OpenMath conversion, GSM calculator-preserving
traces, deterministic chains/sorting/Countdown, task routing and reward dispatch,
no-truncation length limits, locked atomic content/tokenizer/version caches,
postfilter caps, evaluation-question exclusion including original fields,
validation reservation before oversampling and unmodified replay at its original
SmolTalk:MMLU×3 source proportion. Procedural RNG identities and question-hash
partitions separate train/validation/test even when a tiny domain repeats a
question under different RNG seeds. Exact filtering does not rule out semantic
contamination. All generated data and downloaded shards remain outside Git.

`uv run --no-sync python -m pytest tests/test_reasoning_tasks.py -q`:
**40 passed**. Required full suite: **190 passed, 10 skipped**, 6.63 seconds.
Scoped `uvx ruff check` passed. The tests use local fixtures only, including
actual locally trained fixture tokenizers; no dataset download is used by tests.

M1 lint follow-up: imports normalized and frozen reward defaults shared through a
module-level constant; behavior remains covered by the integrated 190-test pass.

## 2026-09-20 — runtime verification and GPU priority

M1 lint follow-up commit: `7472215`; M2 commit: `2c325ad`.
Implemented the new checkpoint source entries, generation adapter retaining true
terminal targets/tool masks, mean-only RL with exact group-token normalization,
partial microbatches, constant-reward skip, collective inactive-rank handling,
synchronized checkpointed curriculum, configurable rewards and JSONL diagnostics.
The evaluator independently samples greedy and pass@8 responses, stores raw
completions/provenance and reports binary task metrics, Wilson intervals, lengths,
termination and timing. No reward score is reported as evaluation accuracy.

Runtime tests include a full tiny CPU RL loop with real gradients, two optimizer
updates, partial sample batches, checkpoint round trips and component logs.
`uv run --no-sync python -m pytest -m "not slow"`: **190 passed, 10 skipped**,
7.20 s immediately before M2 commit. After the user's GPU-priority instruction,
`CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m pytest -m "not slow"`:
**189 passed, 14 skipped**, 4.43 s (includes three new resource-guard tests).
Scoped lint passes for all new runtime files. Multi-GPU behavior is not empirically
validated; collective paths have local fixture coverage only.

The user is running other LLM training and explicitly gives it priority. This
project's evaluation PID 3272133 was terminated with SIGTERM on 2026-09-20;
the other training PID 3256736 was not signaled or modified. Observed total VRAM
before cancellation: 44,758 / 97,887 MiB (45.7%). Cancellation was proactive to
prioritize the other workload, not an out-of-memory failure. This project's GPU
work is deferred while that training is active. The runner's resource guard
polls total-device VRAM and terminates only its own new process group above 80%;
it also cancels its child if monitoring fails. Guard behavior is CPU-fixture tested.

## M3 — 50-iteration SFT smoke

Command (2026-09-19; protected baseline files unchanged):

```bash
OMP_NUM_THREADS=1 uv run --no-sync python -u -m scripts.reason_sft \
  --source sft --model-tag d24 --model-step 486 \
  --output-tag d24-reason-smoke-20260919 --num-iterations 50 \
  --device-batch-size 2 --total-batch-size 16384 \
  --metamath-rows 256 --omi2-rows 256 --gsm8k-epochs 1 \
  --procedural-rows 512 --replay-frac 0.25 --validation-size 32 \
  --eval-tokens 8192 --chatcore-every -1 --no-compile
```

- Input: original `sft/d24` step 486; output `reason_sft/d24-reason-smoke-20260919`
  step 50, separate model/optimizer/metadata files outside Git.
- Filtered mixture: 11,255 train rows, 192 validation rows; sequence length 2048.
- First five raw losses mean .552706; last five mean .435255. All 50 finite.
- Validation BPB .344775 on the new reasoning validation mixture; it cannot be
  compared directly to the original SFT checkpoint's historical .272163.
- Training 69.918 seconds; median warmed step 1.322 seconds, 12,396 packed tokens/s;
  peak allocated VRAM 23,794,256,384 bytes (22.16 GiB). Data preparation, downloads,
  initial model load and checkpoint write are outside the training-loop timer.
- Original smoke log `/tmp/reasoning-sft-smoke.log`; durable training metrics and
  metadata are in the output checkpoint directory.
- SFT CPU tests cover fixed iteration counts, full-epoch stopping, fresh optimizer,
  checkpoint context, tool masking and finite/nonnegative learning rates. Review
  fixed inherited prefetch stopping so tiny datasets cannot checkpoint step 0.
  Progress display now accounts for gradient accumulation.

Completion/reload check on 2026-09-20:

```bash
OMP_NUM_THREADS=1 uv run --no-sync python -u -m scripts.reason_eval \
  --source reason_sft --model-tag d24-reason-smoke-20260919 --model-step 50 \
  --tasks gsm8k --max-examples 4 --max-new-tokens 1024 --device-batch-size 2 \
  --output-dir /home/neo/.cache/nanochat/reasoning_runs/d24-reason-smoke-gsm4
```

Canceled to prioritize the user's other training after two complete prompts.
The saved 18 completions (two greedy plus 16 sampled) all contain valid closed
think blocks and have binary correctness 0. This passes the format smoke gate,
not a correctness improvement test. Partial raw JSONL is retained; no final
summary was written. The chained ChatCORE comparison never started. Do not treat
this interrupted run as a completed evaluation or use its contended timing as
an isolated-GPU throughput measurement. Five-step RL GPU smoke and full campaign
remain pending; no campaign budget estimate is asserted without those measurements.

SFT smoke executed from the implementation working tree atop `54910a5`; source
was committed afterward. Subsequent SFT changes corrected progress display and
style, without changing the explicit 50-step loss/optimizer schedule. Runtime and
evaluation implementation commit: `dee1203`. The SFT commit gate rerun with CUDA
hidden passed **189 tests, 14 skipped** in 4.43s; all new Python files and modified
CLI/evaluator files pass scoped Ruff. The six protected baseline files compare
byte-identical to `ac2aecf`.

### Length sample (CPU-only, 2026-09-20)

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 nice -n 10 uv run --no-sync
/tmp/reasoning_lengths.py` inspected the first 1000 physical rows of each first
downloaded shard and 1000 generated examples per procedural task (seed 42,
difficulty 1). This is a schema/length sanity sample, not the full training mixture.
All counts use untruncated rendering with the actual cached tokenizer.

| Source | Converted | Retained at both limits | ≤256 | 257–512 | 513–768 | 769–1024 | 1025–1536 | >1536 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GSM8K |1000|1000|822|177|1|0|0|0|
| MetaMathQA |950|948|559|353|30|4|3|1|
| OpenMathInstruct-2 |572|567|148|246|103|54|20|1|
| Chains |1000|1000|1000|0|0|0|0|0|
| Sorting |1000|1000|1000|0|0|0|0|0|
| Countdown |1000|1000|1000|0|0|0|0|0|

Maximum full lengths: GSM 517, MetaMath 2034, OpenMath 1544, chains 87, sorting 93,
Countdown 100. Assistant limits reject additional traces even when total length
fits. Raw histogram JSON: `/tmp/reasoning-lengths.json` (copied to durable external
smoke evidence before handoff).

## M5 — CLI, runner, final offline verification

SFT implementation commit `019ab4d`, tagged `reasoning-m3`. Added source-aware
CLI budgets, dimmed streaming think text with all delimiter split positions
covered, unchanged plain redirected text, and the preserved 512-token existing
`run_chat_eval` API default. The runner chains explicit input checkpoint → SFT
→ binary evaluation → RL → evaluation/ChatCORE, using uv and fresh output tags.
It wraps GPU subprocesses with the 80% total-device VRAM guard. README, repository
guidance and provisional results explicitly describe the pending GPU and campaign
gates. No `reasoning-m4` or `reasoning-m6` tag is warranted yet.

Final required suite with CUDA hidden: **190 passed, 14 skipped**, 4.47 s
(`/tmp/reasoning-m5-tests.log`; use the log's measured duration if rerun).
Scoped Ruff passed for all newly added Python files and the modified CLI/evaluator;
`bash -n runs/reasoning_singlegpu.sh` passed. Existing checkpoint-manager lint
warnings predate this work and were not included in scoped clean-code claims.
Protected-file comparison against `ac2aecf` is empty. The monitor's actual
`nvidia-smi` parsing returned total-device usage .3522 without launching a GPU
process. Only the user's preexisting training and desktop compute process remained.

Next GPU command, once the priority training is finished (fresh output tag):

```bash
OMP_NUM_THREADS=1 uv run --no-sync python -m scripts.reason_gpu_guard -- \
  uv run --no-sync python -m scripts.reason_rl \
  --source reason_sft --model-tag d24-reason-smoke-20260919 --model-step 50 \
  --output-tag d24-reason-rl-smoke --num-steps 5 \
  --examples-per-step 4 --num-samples 8 --device-batch-size 2
```

Then reload/evaluate that checkpoint with identical budgets, finish the matched
ChatCORE comparison, measure isolated RL/evaluation costs, and present the full
campaign budget decision. Full A/B/C/D runs and the 20-completion D-arm exploitation
audit remain unrun; `REASONING_RESULTS.md` records that explicitly.
