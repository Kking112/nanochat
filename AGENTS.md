# AGENTS.md

This repository is karpathy/nanochat pinned at `92d63d4`, plus a pre-registered study of looped
(weight-shared depth) transformers. Read these before changing anything:

1. `looped_nanochat_proposal.md`: the pre-registration. Section 9 is a normative spec. If a
   requirement conflicts with the code, stop and report it; do not pick an interpretation silently.
2. `dev/LOOPED_LOG.md`: the lab notebook. Every conflict found so far and its approved resolution,
   every deviation from the pinned commit, and measured numbers. Append a dated entry per work
   session. Never rewrite past entries.

## Rules that are easy to break

- **Do not pull upstream.** `origin` is karpathy/nanochat. Never push there.
- **uv: always `--frozen`** (`uv sync --frozen --extra gpu --group dev`, `uv run --frozen ...`, or
  `UV_FROZEN=1`). A plain `uv sync` / `uv run` re-resolved `uv.lock` to the latest version of
  nearly every dependency on this machine. If `git status` shows `uv.lock` modified, restore it.
- **The stock model is the default.** Every new behavior sits behind a flag or config field whose
  default reproduces upstream exactly. `tests/test_looped.py::test_t1_stock_equivalence` compares
  against a frozen copy of upstream `gpt.py` in `tests/fixtures/gpt_stock.py`: never edit that file.
- **Looping is weight sharing and nothing else.** No Post-LN, sandwich norm, residual scaling,
  extra input injection or anything else that differs between looped and baseline arms. If a looped
  arm diverges, record it (the results row has a `diverged` flag); do not patch the architecture.
- **`results/` is append-only.** Never edit or delete rows of `results/looped_results.csv`,
  `results/looped_loopsweep.csv` or files in `results/logs/`, including failed and diverged runs.
  Tests and smoke runs must pass their own `--results-dir` and a scratch `NANOCHAT_BASE_DIR`.
- **`~/.cache/nanochat` is shared** with unrelated checkpoints (`d24`, ...). Never retrain the
  tokenizer there. Study runs use `looped_*` model tags, never `d<number>`.
- **Never resume a study run mid-run.** The dataloader's resume is approximate and changes the
  token stream. Rerun from scratch.
- Attention is SDPA on this GPU (Blackwell has no FA3 kernel; `nanochat/flash_attention.py` gates
  on the compute capability). Layouts require `--window-pattern L`.

## Where things are

- `nanochat/gpt.py`: `GPTConfig.n_prelude/n_core/n_coda/n_loop`, `parse_layout`, `visit_schedule`,
  `set_num_loops` (a plain Python int, so `torch.compile` builds one graph per loop count), FLOP
  accounting per visit. Parameters are indexed by the unique layer `u`; windows, the backout point
  and KV cache slots by the effective index `e`.
- `nanochat/engine.py`: one KV cache slot per effective layer.
- `scripts/base_train.py`: the study flags (`--layout`, `--ref-layout`, `--train-loops`, `--arm`, ...).
- `scripts/base_eval.py --num-loops R`, `scripts/looped_select_lr.py`, `runs/looped_*.sh`.
- Known issue: `scripts/chat_sft.py` writes a `model_config` without the layout keys, so a looped
  model would reload as an unlooped one after SFT. Fix before the optional Phase 3.

## Tests

`python -m pytest tests -m "not slow"` (seconds), `python -m pytest tests` (adds T8/T9, which run
the training script on CPU, ~90 s). The equivalence tests re-randomise all parameters first,
because `init_weights` zeroes every `c_proj` and an untrained block is a no-op.
