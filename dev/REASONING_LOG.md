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
