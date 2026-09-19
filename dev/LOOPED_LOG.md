# Looped nanochat: lab notebook

Running log for the study pre-registered in `looped_nanochat_proposal.md`. One dated entry per
work session. Entries are appended, never rewritten. Codebase pinned at `92d63d4`.

---

## 2026-09-19: Planning, environment survey, spec conflicts

### Scope decisions (confirmed with the author)

- The proposal is authoritative. DeepLoop (arXiv:2607.13491) is related work only. No Post-LN
  block, no sandwich norm, no alpha/beta residual scaling. The only difference between looped and
  baseline arms is the visit schedule (weight sharing).
- The section 4 diagnostics (per-loop residual RMS, core vs non-core grad norm) are implemented.
  They decide whether a DeepLoop-style arm is added in a later phase.
- If a looped arm diverges in Phase 1 the architecture is not patched. The divergence is recorded
  and reported.

### Environment

- NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887 MiB, driver 595.84. System CUDA
  toolkit 13.3 is not used by training: `pyproject.toml` pins torch 2.9.1 with cu128 wheels.
- `~/.cache/nanochat` (shared `NANOCHAT_BASE_DIR`) already holds a tokenizer, 170 train shards
  plus the val shard of climbmix, the eval bundle, and unrelated checkpoints `d24`, `d6`, `d3`.
  These are reused. Every run of this study uses a `looped_*` model tag.
- `nanochat/flash_attention.py` loads FA3 on non-Hopper GPUs whenever the
  `kernels-community/flash-attn3` kernel reports a usable build. "Blackwell uses SDPA" is
  therefore an assumption to be checked at run time, not a guarantee; run scripts record the
  backend that was actually selected.
- `COMPUTE_DTYPE` is resolved once at import and is bf16 on this machine even for CPU tensors, so
  the fp32 equivalence tests force it explicitly.
- Only `GPT.init_weights` consumes the global torch RNG. The optimizer, dataloader, tokenizer and
  evals do not, so `--seed` changes parameter init only. Dataloader resume is approximate (it
  skips a row group), so Phase 1 runs are never resumed mid-run; an interrupted run is rerun from
  scratch.
- `get_peak_flops` has no entry for this GPU, so MFU prints as 0. Cosmetic; left alone.
- **uv must always run with `--frozen` (or `UV_FROZEN=1`).** The committed `uv.lock` carries an
  `exclude-newer-span` option that uv 0.9.28 on this machine does not honour: a plain
  `uv sync` / `uv run` treated the lock as stale and re-resolved ~all dependencies to their
  latest versions (e.g. tiktoken 0.11.0 -> 0.14.0, wandb 0.21.3 -> 0.30.0; torch stayed at 2.9.1
  only because `pyproject.toml` pins it). This happened once during setup; `uv.lock` was restored
  from git and the environment reinstalled with `uv sync --frozen --extra gpu --group dev`.
  Installed and verified against the lock: python 3.10.17, torch 2.9.1+cu128 (arch list includes
  sm_120), numpy 1.26.4, tiktoken 0.11.0, kernels 0.11.7, wandb 0.21.3, pyarrow 21.0.0.

### OPEN BLOCKER for all GPU runs: FA3 is selected on this GPU and then aborts

Proposal 3.2 states "On Blackwell nanochat falls back from FlashAttention-3 to SDPA". On this
machine that is false at `92d63d4`. Measured:

- `nanochat.flash_attention`: `HAS_FA3 = True`, `USE_FA3 = True`, `COMPUTE_DTYPE = bfloat16`.
  `_load_flash_attention_3` gates only on `has_kernel("kernels-community/flash-attn3")`, which
  reports a usable build, although the comment in that function says Blackwell needs the SDPA
  fallback.
- The first FA3 call aborts the whole process:
  `CUDA error (.../flash-attn/flash_fwd_launch_template.h:192): no kernel image is available for
  execution on the device`. The unmodified stock test suite dies this way in
  `tests/test_attention_fallback.py::TestFA3VsSDPA::test_basic_causal` (pytest exit 1, no report),
  and stock `scripts.base_train` would die on its first forward pass.
- With `_override_impl = 'sdpa'` a GPT forward + backward on the GPU works (sm_120).

So every arm, including the stock d12 reference, needs SDPA forced. How to force it is NOT yet
decided: `nanochat/flash_attention.py` is outside the spec 9.1 file list and the choice defines
what "stock" means in the Phase 0 gate. Awaiting the author's decision. CPU-side work (model,
tests T1-T7) does not depend on it.

### Conflicts and ambiguities found in spec section 9, with the resolution adopted

Spec 9.0 requires conflicts to be reported rather than silently resolved. These were reported to
and approved by the author before implementation.

1. **`--horizon-frac` (9.5 vs 3.2).** Read literally, a 0.4x horizon re-derives the batch size
   (2^19 -> 2^18), the LR batch scale and the weight decay (x2.5), so the LR sweep would tune
   `matrix_lr` under a different recipe than the main runs. Adopted: batch size, LR batch scale
   and weight-decay scale come from the FULL reference horizon; `--horizon-frac` only shortens
   `num_iterations`. Side effects to state in the paper: the Muon momentum warm-up is a fixed
   400 steps (about 40% of a ~1008-step sweep run) and the weight-decay cosine spans the short
   horizon.
2. **Tokenizer setup "as in speedrun.sh" (9.8).** `scripts/tok_train.py` overwrites the tokenizer
   unconditionally, and existing checkpoints depend on it. Adopted: train only if missing.
3. **T4 "through Engine" (9.7).** `Engine.generate` yields tokens, not logits. Adopted: logits are
   compared by driving `model.forward` with a `KVCache` sized the way `Engine` sizes it, and
   `Engine` itself is covered by a greedy token-match check against `GPT.generate`.
4. **"Pass-through of inference loop count" (9.1).** Implemented as model state
   (`set_num_loops`) that `Engine` reads, which satisfies 9.6. It is not an `Engine` argument.
5. **`ve_gate` in 9.4 / T6.** 9.4 lists `ve_gate` beside `lm_head` as if counted once; T6 says the
   L8 vs B12 FLOPs difference is "ve_gate only". Stock already counts `ve_gate` as a matmul.
   Counted per visit, `2,4x2,2` has 6 value-embedding visits, the same as B12, so the difference
   is exactly 0. Adopted: per visit. Either reading passes the 0.5% bound.
6. **"Stock settings" for evals (4.2, 4.3).** `base_train` evaluates 80x524288 val tokens and
   500 CORE examples per task; `base_eval` defaults to 40x524288 tokens and the full CORE set.
   Adopted: the loop sweep is pinned to the training-time values so its numbers are comparable
   with `results/looped_results.csv`.
7. **`scripts/chat_sft.py` is not in the 9.1 file list** but writes a hand-built `model_config`
   without the new layout keys, so a looped model would reload as an unlooped U-layer model after
   SFT. KNOWN ISSUE, not fixed now; must be fixed before optional Phase 3.
8. **Additions beyond 9.1 / 9.5:** `--arm`, `--results-dir`, a switch to skip saving optimizer
   state, a required `--model-tag` whenever `--layout` is given (the stock default tag is
   `d{depth}`, which would make every arm overwrite one directory), `runs/looped_common.sh`, and
   `scripts/looped_select_lr.py`.
