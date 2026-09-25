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

**Resolution (same day, authorised by the author: "you can modify the files to force sdpa").**
`_load_flash_attention_3` now returns `None` when the compute capability major is >= 10, which
is what the comment already in that function says Blackwell needs. After the change:
`HAS_FA3 = False`, `USE_FA3 = False` on this GPU, and `tests/test_attention_fallback.py` +
`tests/test_engine.py` pass with the GPU visible (13 passed, 10 skipped; the skips are the
FA3-only comparisons). This is a DEVIATION from pinned `92d63d4` in a file outside spec 9.1, in
its own commit. It applies identically to every arm, including the stock d12 reference, so
"stock" in the Phase 0 gate means `92d63d4` + this gate. To be stated in the paper.

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

### Implementation (same session)

Built on branch `looped` (feature branches `looped-model`, `looped-train`, `looped-runs`; git cannot
hold `looped` and `looped/model` at once, hence the dashes).

- `nanochat/gpt.py`: `GPTConfig.n_prelude/n_core/n_coda/n_loop`, `parse_layout`, `visit_schedule`,
  `set_num_loops`; forward iterates the schedule (parameters by unique layer `u`; window, backout at
  `E // 2` and KV cache slot by effective index `e`); `diag` list for per-layer residual RMS;
  per-visit FLOP accounting, E-based KV accounting.
- `nanochat/engine.py`: one KV cache slot per effective layer. `nanochat/checkpoint_manager.py`:
  stock defaults for the new config keys.
- `scripts/base_train.py`: `--layout --width-depth --ref-layout --horizon-frac --matrix-lr-mult
  --seed --train-loops --arm --results-dir --no-save-optimizer`. `scripts/base_eval.py`:
  `--num-loops`, outputs keyed by model tag and R.
- Tests T1-T9 in `tests/test_looped.py` (68 passed, 10 skipped FA3-only, full suite). The model
  tests were mutation-checked: cache slot by `u` instead of `e` (T4 fails), backout at
  `n_layer // 2` (T3, T7 fail), FLOPs counted once per unique block (T6 fails).
- T8/T9 run `scripts.base_train` on CPU with `TORCHDYNAMO_DISABLE=1`: CPU inductor needs
  `setuptools`, which only nanochat's `cpu` extra installs. Compiled behaviour is covered by the
  GPU smoke below instead.
- T8 caught a real bug during development: the data hash depended on the micro-batch size
  (inputs-then-targets per micro-batch, and the loader's one-batch prefetch past the last step).
  The hash is now row-wise and capped at the rows the run trains on.

### GPU smoke (30-60 steps per arm, scratch base dir and results dir, nothing in `results/`)

All arms: identical data hash, identical full-horizon `num_iterations` = 2520 (1.32B tokens),
batch 524,288, weight decay 0.28, LR batch scale 1. No recompile-limit failures, including the
random-R arm (4 training graphs, compiled before the timed loop).

| arm | layout | U / E | FLOPs/token | block params | value-embed params | tok/s (steady) | peak VRAM |
|---|---|---|---|---|---|---|---|
| stock d12 | `--depth 12` | 12 / 12 | 8.871e8 | 84.9M | 151.0M | ~268k | 28.8 GB |
| B12 | `12` | 12 / 12 | 8.871e8 | 84.9M | 151.0M | ~266k | 28.8 GB |
| B8 | `8` | 8 / 8 | 6.417e8 | 56.6M | 100.7M | ~360k | 20.9 GB |
| B6 | `6` | 6 / 6 | 5.191e8 | 42.5M | 75.5M | ~446k | 16.9 GB |
| L8 | `2,4x2,2` | 8 / 12 | 8.871e8 | 56.6M | 100.7M | ~263k | 28.2 GB |
| L6s | `2,2x4,2` | 6 / 12 | 8.871e8 | 42.5M | 75.5M | ~264k | 27.9 GB |
| L6p | `0,6x2,0` | 6 / 12 | 8.871e8 | 42.5M | 75.5M | ~264k | 27.9 GB |
| LR | `2,4xR,2`, R~U{1..4} | 8 / 8-20 | 6.4e8-1.38e9 | 56.6M | 100.7M | ~239k mean | 42.8 GB |

- L8, L6s, L6p have exactly B12's FLOPs per token (T6's "ve_gate only" difference is 0: each has
  6 value-embedding visits) and B12's throughput. LR step time scales with R: ~1.4 / 2.0 / 2.5 /
  3.0 s for R = 1..4.
- **Revised compute estimate.** A B12-equivalent run is ~2520 x 1.97 s = ~83 min of training, plus
  11 val evals. The proposal guessed 3-5 h. Phase 1 (21 sweep runs at 40% + 21 main runs) is
  therefore on the order of 2 GPU-days, not 5. To be replaced by the Phase 0 measurement.
- **Training on this GPU is not run-to-run deterministic**, on unmodified stock code too: two
  identical stock runs log the same loss for ~5 steps and then differ in the 5th-6th digit (final
  loss of 30 steps: 6.450897 vs 6.451034). Stock vs `--layout 12` differ by the same amount
  (6.450897 vs 6.450961). Consequences: (a) the Phase 0 "identical losses" check under
  torch.compile is a tolerance check (1e-3), exact equality is proven in fp32 by T1/T2;
  (b) "a seed changes parameter initialization only" holds for the inputs of a run, but seed
  variance as measured will include this nondeterminism. To be stated in the paper.
- One L8 smoke run showed 17 consecutive steps at 2x the step time, with normal speed before and
  after: something else used the GPU (this is a desktop workstation). The results row therefore
  also records `tokens_per_sec_median`, which is robust to that; the table above is steady state.
- `base_eval --num-loops` on the LR smoke checkpoint at R = 1, 2, 3, 5 (5 is beyond the trained
  range) runs, R = 2 reproduces the in-training val_bpb, sampling through the Engine KV cache works
  at every R, and a non-looped model is rejected. `infer_bench` on L8 vs B8: 2.62 vs 1.81 ms per
  decoded token (~ 12/8 layers), KV capacity ~ 8/12.
- CORE with a small `--max-per-task` (8) crashes in stock `core_eval` on 10-shot tasks (few-shot
  examples are sampled from the truncated set). Not touched; the study uses 500.

### Run scripts, end-to-end check, independent review (same session)

- `runs/looped_common.sh` (arm table, tags `looped_<arm>_lr<mult>_h<frac>_s<seed>`, skip-if-done),
  `looped_phase0.sh`, `looped_sweep.sh`, `looped_main.sh`, `looped_loopsweep.sh`,
  `scripts/looped_select_lr.py` (refuses to write a selection while an arm's optimum is on a grid
  edge and prints the command that extends the grid; a diverged run loses but still counts as tried).
- **End-to-end check of one real sweep cell** through `runs/looped_sweep.sh` (scratch base dir and
  results dir): B6, 1x, 40% horizon: 1008 steps, 528M tokens, val_bpb 3.171 -> 0.9327, 443k tok/s
  (mean and median agree), 16.9 GB, 1435 s wall-clock of which ~1190 s training; a second invocation
  skipped it. So evals and compile cost ~4 min of a sweep run: not negligible next to 40% of the
  training time, which is why the Phase 0 gate costs training and the rest separately.
- **Independent review** (separate reviewer agent, read-only): no critical/high findings; it
  re-derived the weight-sharing equivalence (logit diff 0.0, gradients = sum over visits) and the
  FLOP parity of B12/L8/L6s/L6p on its own. Fixed from its findings:
  - `--layout` now REQUIRES `--ref-layout`. Without it an arm silently derives its own recipe and
    still completes: B8/L8 would get 1872 steps at weight decay 0.377, B6/L6 1548 steps at 0.456,
    instead of 2520 at 0.28. T9 now checks the horizon is the reference model's and not the arm's.
  - Stopping at a non-finite loss is now limited to study runs (`--arm`); without it the loop
    trains on as upstream does. Under DDP the decision is all-reduced so ranks leave together.
  - The data hash and its GPU sync are limited to study runs; a resumed run records no hash rather
    than the hash of nothing.
  - `git_hash` is marked `-dirty` for untracked files too, so a run launched from uncommitted
    scripts can not carry a clean hash.
  - `looped_loopsweep.sh` skips models without a checkpoint (a diverged run stops before its final
    checkpoint) instead of aborting the whole sweep. The run scripts only create the environment
    when `.venv` is missing, so starting one never touches the `.venv` of a run in progress.
  - A results row of a run without a final eval says `not evaluated`; throughput of a diverged
    run counts its last step; `set_num_loops` rejects bools; new test that the stock sliding-window
    (`SSSL`) path still equals the frozen stock model.
- **For the write-up (no code change, mandated by spec 9.3):** the backout tap is the residual after
  effective layer `E // 2`, so in the test-time loop sweep (H4) it moves with R (e = 4, 6, 8, 10, 12,
  14, 18 for R = 1..6, 8 on `2,4xR,2`), and for the LR arm it moves from step to step in training.
  The H4 curve therefore varies the backout point together with the loop count. The loop-sweep CSV
  records `backout_layer`. The main matrix is unaffected (E // 2 = 6 for every 12-effective-layer arm).
- Full suite: 69 passed, 10 skipped (FA3-only).

### Left for the author

1. `bash runs/looped_phase0.sh` (two full d12 runs, ~1.5 h each, plus the L8 throughput run), record
   its gate report here, then `git tag prereg-v1` if the gates and all tests pass (spec 9.9).
2. Before optional Phase 3: `scripts/chat_sft.py` drops the layout keys from `model_config`.

---

## 2026-09-20: Phase 0 results. Both gates PASS

`bash runs/looped_phase0.sh`, run by the author 2026-09-19 21:36 to 2026-09-20 00:59 at commit
`95dc84e`. Raw outputs are committed under `results/` (`looped_results.csv`, `logs/*.jsonl`,
`stdout/*.log`, `phase0_identity_20260919_213652/`). All runs: SDPA, bf16, seed 0, device batch 32,
data hash `a1413887432d...` (identical in all three rows), 0 diverged.

### Identity check under torch.compile (20 steps)

Stock `--depth 12` vs refactored `--layout 12`: loss at step 0 identical (difference 0.0), worst
difference over the 20 steps 1.3e-4 (tolerance 1e-3). PASS. This is the size of this GPU's
run-to-run drift (see the 2026-09-19 entry), not a difference between the models.

### Full d12 runs (2520 steps, 1,321,205,760 tokens, 1.172e18 training FLOPs)

| run | final val_bpb | min val_bpb | val_bpb @500 | CORE | tok/s mean / median | peak VRAM | wall-clock |
|---|---|---|---|---|---|---|---|
| stock `--depth 12` (`stock12`) | 0.847663 | 0.847663 | 1.005543 | 0.1543 | 260,395 / 260,550 | 28,771 MiB | 5835 s (1.62 h) |
| refactored `--layout 12` (`B12_phase0`) | 0.847749 | 0.847749 | 1.005739 | 0.1494 | 260,521 / 260,701 | 28,771 MiB | 5831 s (1.62 h) |

- **GATE 1: |val_bpb difference| = 0.000086 < 0.003. PASS** (35x inside the gate).
- The val_bpb curves track each other at every one of the 12 evals; the largest gap is 7.2e-4 at
  step 250, shrinking to 1e-4 by step 2500. Learned scalars agree too: `backout_lambda` 0.348 vs
  0.344, `resid_lambdas` within 0.007 everywhere, final residual RMS per layer within 0.2.
- Throughput and VRAM are the same: the refactor costs nothing.

### L8 throughput (`2,4x2,2`, 200 steps, no eval)

261,757 tok/s mean / 261,012 median, 28,194 MiB peak. Same as B12, as equal FLOPs per token
(8.871e8) predict. Weight sharing is free in wall-clock at this scale; it saves ~0.6 GB.

### GATE 2: Phase 1 compute

A B12 run = 5831 s = 1.408 h training (tokens / median tok/s) + 0.212 h compile, 12 val evals and
CORE. A 40%-horizon sweep run = 0.4 x 1.408 + 0.5 x 0.212 = 0.669 h. Phase 1 = 21 main runs + 21
sweep runs = 21 x (1.620 + 0.669) h = **2.0 GPU-days <= 10. PASS.** (Proposal 3.6 guessed ~5.)
Costing every arm like B12 is on the safe side: B8/B6 are much cheaper, only the LR arm is dearer
(~1.17x). Not included: `runs/looped_loopsweep.sh` (42 evaluations). No arm needs to be dropped.

### What these two runs say about noise (useful for reading Phase 1)

`stock12` and `B12_phase0` are the same model, same seed, same data, and differ only by this GPU's
nondeterminism. So this pair is a direct measurement of the noise floor *below* seed variance:

- final val_bpb: 8.6e-5. Any Phase 1 difference has to be read against seed SD, which can not be
  smaller than this.
- **CORE: 0.0049** (0.1543 vs 0.1494) between two runs that agree to 1e-4 in val_bpb. At this scale
  CORE differences below ~0.005 are not distinguishable from rerunning the same model. This supports
  the pre-registered choice of val_bpb as primary and CORE/H5 as descriptive only.

### Correction to the provenance of these rows

All three rows carry `git_hash = 95dc84e...-dirty`. No code was modified: tracked files are
byte-identical to `95dc84e` (`git diff --quiet HEAD -- nanochat scripts runs tests pyproject.toml
uv.lock`). The `-dirty` is a flaw in the 2026-09-19 review fix that made the dirty check count
untracked files repo-wide: the untracked `results/` directory that the first run creates (and an
unrelated untracked `.agents/`) then marks every later run dirty. The rows are left as written
(append-only). Fixed for future runs: the check now covers the code and its environment only
(`nanochat scripts runs tasks tests pyproject.toml uv.lock`), still including untracked files there.
Limit of this statement: it is established now, after the runs, not recorded at run time.

### Definition of done for Phase 0 (spec 9.9)

All tests pass (69 passed, 10 skipped FA3-only, rerun today); both gates met; tokens/s for B12 and
L8 and the revised compute estimate are recorded above. Remaining: the author places
`git tag prereg-v1` on the commit that contains the proposal and the passing tests.

---

## 2026-09-20: Phase 1a, matrix LR sweep. Gate PASS, 2x selected for every arm

`bash runs/looped_sweep.sh` (01:07 to 13:40), then the grid extension
`LR_MULTS="4" bash runs/looped_sweep.sh` (13:41 to 17:55). 28 runs, 16.7 GPU-hours, all at commit
`55961c2` with a clean hash, all with data hash `a1413887432d...`, 0 diverged. Seed 0, 40% horizon
(1008 steps, 528M tokens); batch size, LR batch scale and weight decay at their full-horizon values.

Final val_bpb (`*` = selected):

| arm | layout | 0.5x | 1x | 2x | 4x |
|---|---|---|---|---|---|
| B12 | `12` | 0.91583 | 0.90148 | 0.89974* | 0.90668 |
| B8 | `8` | 0.92869 | 0.92087 | 0.91726* | 0.92158 |
| B6 | `6` | 0.94710 | 0.93285 | 0.93131* | 0.93746 |
| L8 | `2,4x2,2` | 0.92722 | 0.91010 | 0.90891* | 0.91405 |
| L6s | `2,2x4,2` | 0.93882 | 0.91943 | 0.91904* | 0.92352 |
| L6p | `0,6x2,0` | 0.93608 | 0.92238 | 0.91722* | 0.92166 |
| LR | `2,4xR,2` | 0.92940 | 0.91730 | 0.91586* | 0.92194 |

- After the first grid {0.5, 1, 2} the optimum of EVERY arm was at the upper edge (2x), so by
  section 3.3 step 2 the grid was extended one step for every arm. 4x is worse than 2x for every
  arm, so every optimum is now interior. **Gate "no arm has its optimum at a grid edge after
  extension": PASS.** `results/selected_lr.json`: 2x for all seven arms, written by
  `scripts.looped_select_lr` (which refused to write anything before the extension).
- The motivation for the sweep (section 3.3) was that a shared weight sums its gradient over R
  visits, so stock `matrix_lr` might be mis-tuned for looped arms specifically. At this horizon
  that is not what the sweep shows: the optimum is the same (2x) for untied and looped arms alike,
  so no arm gets a different multiplier than its controls, and the main-matrix comparisons are all
  at one common `matrix_lr` = 0.04. That 2x also wins for the stock B12 is most plausibly a property
  of the 40% horizon (a shorter run tolerates a higher LR), i.e. the stated limitation of tuning at
  a reduced horizon, not evidence that stock nanochat is mis-tuned at its full horizon.
- Caveats for reading the table, all single-seed: the 1x vs 2x gap is small for several arms
  (L6s 0.0004, L8 0.0012, LR 0.0014, B6 0.0015, B12 0.0017; larger for B8 0.0036 and L6p 0.0052).
  The same-seed noise floor from Phase 0 is ~1e-4 and the seed SD is not known yet, so for L6s in
  particular "2x beats 1x" is not established; the pre-registered rule selects by lowest val_bpb
  regardless and was followed. Descriptively, too low an LR (0.5x vs 1x) costs the looped arms more
  (L6s +0.019, L8 +0.017, L6p +0.014, LR +0.012) than B8 (+0.008), with B12 and B6 (+0.014) in between.
- No arm-ordering conclusions are drawn from sweep runs: they are one seed at 40% horizon and are
  not part of the confirmatory analysis.
- Throughput (median tok/s, stable across the 4 runs of each arm): B12 ~263k, L8 ~267k, L6s ~268k,
  L6p ~268k, B8 ~363k, B6 ~449k, LR ~210k (mean E = 14, 42.8 GB peak). A sweep run costs 38-39 min
  for the E = 12 arms, 44 min for LR.

Next: main matrix, `bash runs/looped_main.sh` (7 arms x seeds 0,1,2 at 2x, full horizon), ~1.6 h
per E = 12 run, ~1.4 GPU-days in total. `scripts/looped_analyze.py` (decision rule, rho, figures)
was written and tested on synthetic data on 2026-09-20, before any main-matrix run existed, and is
merged before the main matrix starts.

---

## 2026-09-22: Phase 1b main matrix and loop sweep. Pre-registered verdicts

`bash runs/looped_main.sh` (2026-09-20 17:56 to 2026-09-22 00:30, 21 runs, 32.4 GPU-hours) and
`bash runs/looped_loopsweep.sh` (00:33 to 09:54, 42 evaluations). All at `967caca`, clean hashes,
data hash `a1413887432d...` in every run, **0 diverged of 21**. `python -m scripts.looped_analyze`
wrote `results/analysis.md` and `results/figures/F1-F6`; the numbers below are copied from it.

| arm | layout | U/E | block params | VE params | val_bpb (s0, s1, s2) | mean | SD | CORE | tok/s | VRAM |
|---|---|---|---|---|---|---|---|---|---|---|
| B12 | `12` | 12/12 | 84.9M | 151.0M | 0.85060, 0.85094, 0.85108 | **0.85087** | 0.00025 | 0.149 | 266k | 28.8 GB |
| L8 | `2,4x2,2` | 8/12 | 56.6M | 100.7M | 0.86001, 0.85901, 0.86005 | **0.85969** | 0.00059 | 0.149 | 268k | 28.2 GB |
| LR | `2,4xR,2` | 8/8-20 | 56.6M | 100.7M | 0.86493, 0.86523, 0.86529 | **0.86515** | 0.00019 | 0.139 | 265k | 42.8 GB |
| B8 | `8` | 8/8 | 56.6M | 100.7M | 0.87034, 0.86979, 0.86921 | **0.86978** | 0.00057 | 0.137 | 367k | 20.9 GB |
| L6p | `0,6x2,0` | 6/12 | 42.5M | 75.5M | 0.86979, 0.87083, 0.87116 | **0.87059** | 0.00071 | 0.148 | 270k | 27.9 GB |
| L6s | `2,2x4,2` | 6/12 | 42.5M | 75.5M | 0.87129, 0.87066, 0.87057 | **0.87084** | 0.00039 | 0.140 | 270k | 27.9 GB |
| B6 | `6` | 6/6 | 42.5M | 75.5M | 0.88592, 0.88614, 0.88525 | **0.88577** | 0.00046 | 0.129 | 452k | 16.9 GB |

Decision rule (3.4): supported iff difference > 2 x pooled SD AND all seeds of one arm beat all of the other.

- **H1 SUPPORTED** (all three): L8 < B8 by 0.0101 (2 x pooled SD 0.0012); L6s < B6 by 0.0149
  (0.0009); L6p < B6 by 0.0152 (0.0012). Every seed of every looped arm beats every seed of its
  equal-parameter control. Looping helps at equal parameters, in this recipe, by 10-30 seed SDs.
- **H2 SUPPORTED**: B12 < L8 by 0.0088 (0.0009). **rho(L8) = 0.534** (seed combinations 0.49 to
  0.59), inside the predicted [0.3, 0.8]. rho(L6s) = 0.43, rho(L6p) = 0.44: looping recovers about
  half of the loss that removing 4 layers costs, and a bit over 40% of the loss of removing 6.
- **H3 INCONCLUSIVE**: L6s vs L6p differ by 0.0002 (threshold 0.0012); the seeds interleave. The
  weak prediction "sandwich not worse than pure loop" is neither supported nor contradicted.
- **H4, split verdict.** (a) Monotone decrease over R = 1..4: **NOT supported**, narrowly: for all
  three seeds val_bpb falls 0.881 -> 0.865 -> 0.8625 from R = 1 to 3, then rises by 0.0002-0.0004 at
  R = 4. The optimum of a model trained with R ~ U{1..4} is R = 3, not the largest trained R.
  (b) R in {5, 6, 8} no worse than R = 4 by more than 0.005: **supported** for all seeds (worst
  +0.0038 at R = 8). No improvement beyond R = 4 (hoped for, not predicted). The randomized-R model
  is robust to the loop count: 0.8625-0.8670 across R = 2..8, all within 0.005 of each other.
- **L8 control (trained at fixed R = 2):** collapses at any other R: 1.10 at R = 1, 0.99 at 3, 1.07
  at 4, 1.3-1.4 at 5-8 (CORE 0.03-0.07). Without randomized-R training there is no test-time loop
  scaling at all. F4.
- **Price of loop-count robustness:** LR at its evaluation R = 2 is 0.0055 worse than L8 (same
  parameters, same layout), and 0.0026 worse at its best R = 3; it also costs 13% more training
  FLOPs (mean E = 14) and 42.8 GB VRAM.
- CORE (secondary, descriptive): tracks val_bpb loosely. L8 = B12 = 0.149 despite 0.009 bpb between
  them; L6p 0.148 vs L6s 0.140 with equal bpb. Seed SD of CORE is 0.003-0.009 per arm, consistent
  with the 0.005 same-seed floor from Phase 0. H5 (reasoning-like vs recall tasks) is not analysed
  here: at this floor the per-task differences are not interpretable.
- Efficiency (tertiary): the equal-compute arms match B12's throughput (266-270k tok/s) and VRAM
  within 3%; B8 is 1.38x and B6 1.70x faster. Weight sharing buys parameters, not wall-clock.
- F6 (residual RMS through depth): the looped arms grow more slowly through their second core pass
  than B12 does through layers 7-12 (end at 27-29 vs 33); no instability, consistent with 0
  divergences. L6p's first layer (RMS 10) differs from every other arm (15-19): with no prelude the
  first core visit sees the raw embedding.

**Phase 2 gate (3.5): H1 supported => proceed to the larger-width confirmation.**

Compute: Phase 1 total 32.4 + 16.7 + ~9.4 (loop sweep) = ~58 GPU-hours, 2.4 GPU-days.

---

## 2026-09-22: Phase 2 design (decided before any Phase 2 run)

Author's decisions: **d20 width** (the proposal allowed d16 or d20; d16 would have been ~16 GPU-h,
d20 is ~58 GPU-h), and the matrix LR by a **mini-sweep on B_U** at 40% horizon, seed 0, selecting by
val_bpb with 1x preferred when the two are within 0.003 bpb; the grid depends on a stability check
of the Phase 1b runs at 2x. Implemented in `runs/looped_phase2.sh`.

- Stability check of all 21 full-horizon 2x runs: largest single-step rise of the smoothed loss
  0.020-0.032, at the same steps (233, 878, 994, 1664) across arms, i.e. data batches, not
  instability; val_bpb monotone at every one of the 12 evals in every run; grad norm never above
  2.0x its median. **No instability => grid {1x, 2x}.**
- Arms: `d20_B20` = layout `20` (B_E, 393M block params), `d20_B12` = `12` (B_U), `d20_L12` =
  `2,8x2,2` (U = 12, E = 20, same sharing ratio as L8). dim 1280, 10 heads. Reference horizon:
  12 x 435M = 5.22B tokens, auto batch 1,048,576, 4980 steps. Data: Phase 1 consumed 29 of the 170
  shards for 1.32B tokens, so 5.22B needs ~115 shards: single epoch.
- Smoke (15 steps, scratch dirs): B20 and L12 both 77.6k tok/s, 73.6 / 70.8 GB at device batch
  32, FLOPs/token 3.240e9 for both (equal, as at d12). Estimates: 18.7 h per E = 20 run, ~11.5 h
  for B_U, 4.6 h per mini-sweep run: ~58 GPU-hours total.

---

## 2026-09-24: Phase 2 mini-sweep done (1x selected); B_E run lost to a power outage, restarted

- Mini-sweep on `d20_B12` (B_U), 40% horizon (1992 steps, 2.09B tokens), seed 0, at `2349a33`:
  **1x: 0.79705, 2x: 0.79730**. The difference (0.00025) is within the 0.003 band, so by the
  pre-set rule **1x is selected** for all three Phase 2 arms (`results/selected_lr_phase2.json`).
  At d20 and this horizon the stock `matrix_lr` is as good as 2x, unlike at d12 / 40% in Phase 1a.
  Both runs 4.9 h, 125.6k tok/s (12 layers at dim 1280), 46.4 GB, 0 diverged.
- **Power outage** at ~16:12 on 2026-09-23 killed `looped_d20_B20_lr1.0_h1.0_s0` at step 2720 of
  4980 (10.1 h in). The machine was down until 22:45 on 2026-09-24. No results row and no checkpoint
  were written (checkpoints are written at the last step only), so nothing of the run is usable and,
  per the no-mid-run-resume rule, it is **rerun from scratch**. Its partial logs
  (`results/logs/looped_d20_B20_lr1.0_h1.0_s0_20260923_053920.jsonl`, 51 complete lines to step
  2700, and the matching stdout log) are kept as written and are not part of any analysis; the rerun
  gets a new run id. Integrity checked before restarting: `looped_results.csv` has 54 complete rows
  and an intact last line; the two sweep checkpoints were written 10+ h before the outage and have
  identical sizes; `git fsck` clean; the code tree identical to `2349a33`; CUDA works after reboot.
- Restart: `bash runs/looped_phase2.sh` skips the two sweep runs, recomputes the same selection and
  runs `d20_B20`, `d20_B12`, `d20_L12` at 1x: ~49 GPU-hours remaining.

---

## 2026-09-24 (later): intermediate checkpoints with exact resume; Phase 2 relaunched

After the outage the author asked for periodic checkpointing (20 per run, each replacing the last).
Checkpoints are only useful with a resume, and the stock resume is approximate (it skips a row
group, which changes the token stream and would break the identical-data-order protocol of 3.2).
So resume was made exact instead:

- `--num-checkpoints 20`: every `num_iterations // 20` steps (249 for the d20 runs, ~56 min) a
  checkpoint with optimizer state is written and the previous intermediate one is deleted after
  the new one is complete. Files are written to a temporary name and renamed into place, and the
  meta file last, so a crash mid-save can never leave a truncated checkpoint that looks complete.
- `--auto-resume`: picks the latest complete intermediate checkpoint of the same run
  configuration (asserted against the saved `user_config`), restores model, optimizer and the
  study's loop state (val_bpb history, step times, wall-clock of earlier segments, run id, so the
  JSONL log is continued and the results row is that of one run), and then **replays the data
  stream** from the start up to the interruption point. The stream is deterministic and seed
  independent, so the resumed run trains on exactly the batches an uninterrupted run would have.
  The leading-rows hash is rebuilt during the replay, so the data check holds for resumed runs.
  Replay speed measured at d20: 320 micro-batches (21M tokens) in 6 s, ~3.5M tok/s: a worst-case
  replay of a whole d20 stream is ~25 min. Cost of an interruption is therefore at most ~56 min of
  training plus the replay.
- `--exit-after-step N` (testing only) exits right after the checkpoint of step N, to simulate an
  interruption. T10 (`tests/test_looped.py`) runs an uninterrupted 20-step CPU run against one cut
  at step 10 and resumed: same losses step for step (within 2e-3, fp32 CPU summation order), same
  data hash, one results row, one continued log, only the final checkpoint left. A d20-scale smoke
  (40 steps, cut at 20) behaved the same: smooth loss across the resume, study data hash reproduced.
- `runs/looped_common.sh` now passes `--num-checkpoints 20 --auto-resume` to every study run. The
  stock path (no flags) is unchanged. Full suite: 71 passed, 10 skipped.
- Caveat for the paper: a resumed run is not bit-identical to an uninterrupted one (bf16 GPU
  training is not run-to-run deterministic anyway, ~1e-4 in loss by step 30, see 2026-09-19), and
  its throughput row mixes segments; `tokens_per_sec_median` is robust to that. Whether a Phase 2
  row was resumed is visible in its JSONL log (`resume` events).
- The `d20_B20` rerun that started at 22:58 was stopped at step 5 for this change and relaunched.
