# Repository guidance

## Environment and verification

Use `uv` for Python execution and dependencies. Install the development GPU
environment with `uv sync --extra gpu --group dev`; run tests with
`uv run --no-sync python -m pytest -m "not slow"`. Use type annotations for new
Python interfaces; simple typed dataclasses and TypedDicts suffice for reasoning
metadata and rewards. Add a validation framework only when its benefits justify
the dependency.

Assume local Ubuntu hardware unless instructed otherwise. The owner's generic
PC profile is Ryzen 9, RTX 5090 (32 GB), 64 GB RAM, 2 TB SSD. The reasoning
experiment machine was actually detected as an RTX PRO 6000 Blackwell (96 GB),
driver 595.84 on 2026-09-19. Recheck `nvidia-smi` before sizing GPU runs; do not
assume cloud resources or that large batches fit the generic PC.

The user's other LLM training takes priority. Do not interrupt or change its
process. Defer this project's GPU work while that training is active. When GPU
checks resume, use `scripts.reason_gpu_guard` with a ceiling of 80% total-device
VRAM; crossing the ceiling must cancel only this project's owned process group.
For CPU verification while the other training runs, set `CUDA_VISIBLE_DEVICES=''`.

## Changes and commits

Create a feature branch before implementing a feature. Preserve unrelated local
work. Test, fix, and document each completed task before committing; stage explicit
files only and use scoped Conventional Commits. Never commit incomplete or untested
changes. Update README and relevant design/lab documentation with behavior,
commands, measured results and limitations. Keep refactoring existing checkpoint
code separate from new behavior.

## Reasoning phase

Read `dev/REASONING_DESIGN.md` and `dev/REASONING_LOG.md` before changing the
reasoning stages. The approved design supersedes older milestone prose in
`REASONING_PHASE_AGENT_BRIEF.md`. Keep `nanochat/gpt.py`, `nanochat/tokenizer.py`,
`nanochat/engine.py`, `scripts/base_train.py`, `scripts/chat_sft.py` and
`scripts/chat_rl.py` byte-identical to the reasoning branch baseline `ac2aecf` so
the original binary-reward experiment remains reproducible.

Use the existing parquet loader for external datasets; inspect source schemas,
examples, revisions and licenses before conversion. Keep checkpoints, datasets
and raw results outside Git under `NANOCHAT_BASE_DIR`. New training runs require
fresh output tags so they cannot overwrite parents or other experimental arms.
New tests must use local fixtures, not dataset downloads.

Report binary correctness separately from reward diagnostics. Full experiments
await the user's post-smoke budget decision; separately confirm any run expected
to exceed approximately six GPU-hours. Stop and discuss unexpected failures in
baseline tests, unclear dataset licensing, or measured cold-start regression.
Do not merge the draft PR, force-push, rewrite history, or integrate upstream
changes during the reasoning campaign. Tag only completed milestone gates.
