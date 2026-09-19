# Weight-Shared Depth in a Modern Small-LM Recipe: A Controlled Comparison of Looped and Standard Transformers in nanochat

**Status:** Research proposal and implementation specification (pre-registration draft)
**Author:** Neo
**Date:** 2026-09-19
**Target codebase:** `karpathy/nanochat`, commit `92d63d4` (2026-07-03). Pin this commit; do not track master during the study.
**Hardware:** 1x NVIDIA RTX Pro 6000 Blackwell (96 GB), 128 GB DDR5, Ryzen 9, Ubuntu 24.04, CUDA 13.

This document has two audiences. Sections 1-8 are the research proposal. Section 9 is an implementation specification written for an AI coding agent that will modify the nanochat repository. Section 9 is normative: where it says MUST, the agent must comply or stop and report.

---

## Abstract

Looped (depth-recurrent) transformers reuse a block of layers several times per token, which decouples effective depth from stored parameters. Published evidence that this helps comes mostly from either large-budget training runs or from plain GPT-2-style baselines. It is unclear how much of the benefit survives in a modern, heavily tuned small-model recipe that already contains several depth-related tricks (per-layer residual scalars, embedding re-injection, value embeddings, Muon). We propose a controlled, single-GPU study inside nanochat that compares standard and looped models sharing one width, one dataset, one token budget, and one optimizer recipe. A single looped model is compared against two untied baselines at once: one with the same compute per token (more parameters) and one with the same parameters (less compute per token). This yields a direct estimate of what fraction of the "missing layers" loss is recovered by looping. We additionally compare loop layouts at a fixed parameter count and test whether a model trained with a randomized loop count can convert extra inference-time loops into lower loss. All hypotheses, metrics, and decision thresholds are fixed in this document before any run.

---

## 1. Background and motivation

A standard depth-N transformer applies N distinct blocks once. A looped transformer stores K distinct blocks and applies some of them R times, so effective depth exceeds unique depth. The idea dates to Universal Transformers and ALBERT and has returned as a route to latent test-time compute: recurrent-depth models (Geiping et al., 2025), looped models for reasoning (Saunshi et al., 2025), large-scale looped LMs (Ouro), parameter-shared conversions of pretrained models (Relaxed Recursive Transformers), and loop-aware residual scaling (DeepLoop).

Two gaps motivate this study.

1. **Baseline strength.** Most small-scale looped results use GPT-2-style baselines. nanochat's GPT is not that. At the pinned commit it includes rotary embeddings, QK-norm, ReLU^2 MLPs, untied embeddings, logit soft-capping, learned per-layer residual scalars (`resid_lambdas`), learned per-layer re-injection of the input embedding (`x0_lambdas`), alternating-layer value embeddings, a "smear" bigram gate, a mid-layer "backout" subtraction, and Muon for matrix parameters. Several of these interact with weight sharing. Notably, `x0_lambdas` already provides the input re-injection that recurrent-depth designs add by hand.
2. **Accessibility.** Reported looped-LM results typically consume hundreds to thousands of datacenter GPU-hours. A clean result reproducible on one workstation GPU, in an open harness many people already use, is independently useful.

This is a controlled comparison and partial replication. It does not claim a new architecture.

## 2. Research questions and hypotheses

All models share width d=768 (nanochat d12 width), the tokenizer, the data stream, the token budget, the batch size, and the learning-rate schedule. Notation `P,KxR,C` means P unique prelude layers, K unique core layers applied R times, C unique coda layers. Unique depth U = P+K+C. Effective depth E = P+K*R+C.

**Primary metric:** final validation bits-per-byte (`val_bpb`). **Secondary:** DCLM CORE score. **Tertiary:** throughput, peak VRAM, decode speed.

Define the *recovery fraction* for a looped model L with U unique and E effective layers:

    rho(L) = (bpb(B_U) - bpb(L)) / (bpb(B_U) - bpb(B_E))

where B_n is the untied n-layer baseline. rho = 0 means looping adds nothing over the small model; rho = 1 means looping fully matches the larger untied model.

- **H1 (looping helps at equal parameters).** bpb(L8) < bpb(B8) and bpb(L6*) < bpb(B6), by more than 2 pooled seed standard deviations.
- **H2 (sharing has a cost at equal compute).** bpb(L8) > bpb(B12). Prediction: 0.3 <= rho(L8) <= 0.8. A value of rho >= 1 or rho <= 0 falsifies the prediction in an interesting direction and will be reported as such.
- **H3 (layout matters at fixed parameters and compute).** Between `2,2x4,2` and `0,6x2,0` (both U=6, E=12), we predict the sandwich layout is not worse than the pure loop. This is a weak directional prediction; the question is open.
- **H4 (test-time loops).** For a model trained with R sampled uniformly from {1,2,3,4}, val_bpb decreases monotonically in R over the trained range, and at R in {5,6,8} is no worse than at R=4 by more than 0.005 bpb. Monotone improvement beyond R=4 is hoped for but not predicted.
- **H5 (exploratory, not confirmatory).** Looping's gain over the equal-parameter baseline is larger on CORE tasks that are reasoning-like than on knowledge-recall tasks. CORE is noisy at this scale; this is reported descriptively only.

## 3. Experimental design

### 3.1 Arms

| ID | Layout | Unique U | Effective E | Role |
|---|---|---|---|---|
| B12 | `12` | 12 | 12 | Reference baseline (stock nanochat d12) |
| B8 | `8` | 8 | 8 | Equal-parameter control for L8 |
| B6 | `6` | 6 | 6 | Equal-parameter control for L6s, L6p |
| L8 | `2,4x2,2` | 8 | 12 | Main looped model |
| L6s | `2,2x4,2` | 6 | 12 | Sandwich layout, heavy sharing |
| L6p | `0,6x2,0` | 6 | 12 | Pure loop layout |
| LR | `2,4xR,2`, R~U{1..4} in training | 8 | 8-20 | Test-time loop scaling (H4) |

L8 vs B12 is equal compute per token. L8 vs B8 is equal parameters. Both come from one looped training run. Approximate non-embedding block parameters at d=768 are 7.08M per unique block (B12 ~84.9M, U=8 ~56.6M, U=6 ~42.5M). `lm_head` and `wte` are ~25.2M each and identical in all arms. Value-embedding tables are ~25.2M each and scale with the number of unique layers that carry one (see 9.3); they are lookups with no FLOP cost and MUST be reported as a separate line in every parameter table.

### 3.2 Matching protocol

- **Tokens:** every arm trains on the identical token stream for the identical number of steps. The horizon is the stock nanochat compute-optimal horizon for B12 (`target_param_data_ratio=12` applied to B12's scaling parameters; roughly 1.3B tokens).
- **Hyperparameters derived from size** (batch size, LR batch scaling, weight-decay scaling) are computed once from B12 and reused for every arm. They MUST NOT be re-derived per arm (see 9.5).
- **Attention:** `--window-pattern L` for all arms. On Blackwell nanochat falls back from FlashAttention-3 to SDPA, where sliding windows require an explicit mask and are slow. Full context everywhere also removes a per-layer confound in looped models.
- **Precision:** bf16, no fp8.
- **What a seed changes:** parameter initialization only. Data order is held fixed across seeds and arms. This is stated in the paper.

### 3.3 Learning-rate fairness

A shared weight receives a gradient summed over its R visits, so the stock Muon `matrix_lr` may be mis-tuned for looped arms. To avoid measuring hyperparameter mismatch:

1. For every arm, sweep `matrix_lr` in {0.5x, 1x, 2x} of the stock value, one seed, at a 40% token horizon.
2. Select the best multiplier per arm by val_bpb. If the best value is at an edge of the grid, extend the grid one step in that direction.
3. Run 3 seeds per arm at the full horizon with the selected multiplier.

The reduced-horizon sweep is a stated limitation. All sweep results are reported in an appendix, including losing settings.

### 3.4 Statistics

Three seeds per arm. Report every individual run, the mean, and the sample SD. With n=3, formal tests are weak; the decision rule is pre-specified: a difference is called "supported" if it exceeds 2x the pooled SD of the two arms and all three seeds of one arm beat all three of the other. Anything else is reported as "inconclusive". rho is reported with a bootstrap interval over seed combinations. No run is excluded unless it diverges (loss NaN or final bpb above the step-500 value); divergences are counted and reported per arm, because instability is itself a result.

### 3.5 Phases and go/no-go gates

| Phase | Work | Gate to proceed |
|---|---|---|
| 0 | Environment; stock d12 run; refactor; equivalence tests; throughput measurement | Refactored `12` layout reproduces stock forward to numerical tolerance; stock-vs-refactored final val_bpb differ by < 0.003; measured runtime makes Phase 1 fit in <= 10 GPU-days |
| 1a | LR sweep (21 short runs) | No arm has its optimum at a grid edge after extension |
| 1b | Main matrix (7 arms x 3 seeds) | - |
| 2 | Confirmation at larger width (one seed): B_E, B_U, and one looped layout at d16 or d20 width, chosen by measured throughput | Phase 1 shows H1 supported. If H1 fails, stop and write up the negative result |
| 3 (optional) | SFT with stock `chat_sft.py`, chat evals; synthetic multi-hop probes | - |

### 3.6 Compute estimate

B12 costs roughly 1.2e18 training FLOPs. A rough guess for this GPU under SDPA and bf16 is 3-5 hours per B12-equivalent run; this number is unverified and Phase 0 replaces it with a measurement. On that guess: LR sweep ~35 GPU-hours, main matrix ~80 GPU-hours, total about 5 GPU-days for Phase 1. If the Phase 0 measurement is more than 2x worse, drop L6s first, then reduce the sweep to B12, B8, L8.

## 4. Evaluation

1. **val_bpb** on the stock nanochat validation split, stock `--eval-tokens`, at the final step and every 250 steps for curves.
2. **CORE** at the final step with stock settings.
3. **Loop sweep (LR arm and, as a control, L8):** val_bpb and CORE at inference R in {1,2,3,4,5,6,8}. L8 was trained at fixed R=2, so its curve shows what happens without randomized-R training.
4. **Efficiency:** training tokens/s, peak VRAM, and `scripts/infer_bench.py` decode throughput. Reported per arm, since equal FLOPs does not imply equal wall-clock.
5. **Diagnostics logged during training:** residual-stream RMS at the end of each loop iteration; gradient norm of core vs prelude/coda matrices; learned values of `resid_lambdas`, `x0_lambdas`, `backout_lambda` at the end of training.

Figures planned: (F1) val_bpb vs unique parameters, all arms; (F2) val_bpb vs training FLOPs; (F3) recovery fraction bar chart; (F4) val_bpb vs inference R; (F5) training curves; (F6) loop-iteration residual RMS.

## 5. Threats to validity

- **Scale.** Results at ~50-110M non-embedding parameters and ~1.3B tokens may not transfer. Phase 2 is a single-seed check, not proof.
- **Recipe inheritance.** Every nanochat hyperparameter was tuned for untied models. Only `matrix_lr` is re-tuned. This biases against looped arms; it is the conservative direction for H1.
- **Value embeddings.** They hold more parameters than the blocks at this scale and are tied to unique layers, so arms with fewer unique layers lose lookup capacity as well as block capacity. An ablation with value embeddings disabled in B12, B8, L8 (one seed each) is a recommended addition if budget allows.
- **Token budget.** The horizon is compute-optimal for B12, so smaller-parameter arms are relatively over-trained. Reported explicitly; F2 shows the FLOP view.
- **Benchmark floor.** CORE at this scale is only slightly above chance on many tasks; val_bpb is primary for that reason.
- **Single dataset, single tokenizer.**
- **n=3 seeds** with fixed data order understates total run-to-run variance.

## 6. Related work (to be verified and expanded before submission)

The author must read and verify each of these; IDs are from memory and some may be wrong.

- Dehghani et al., Universal Transformers, arXiv:1807.03819.
- Lan et al., ALBERT, arXiv:1909.11942.
- Geiping et al., Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach, arXiv:2502.05171. Source of the prelude/core/coda layout and randomized-R training.
- Saunshi et al., Reasoning with Latent Thoughts: On the Power of Looped Transformers, arXiv:2502.17416.
- Bae et al., Relaxed Recursive Transformers, arXiv:2410.20672 (verify ID).
- Bae et al., Mixture-of-Recursions, arXiv:2507.10524 (verify ID).
- Zhu et al., Scaling Latent Reasoning via Looped Language Models (Ouro), 2025 (find ID).
- Li et al., DeepLoop: Depth Scaling for Looped Transformers, arXiv:2607.13491. Loop-aware residual scaling for Post-LN; relevant to stability diagnostics here, though nanochat is Pre-norm.
- Hao et al., Coconut, arXiv:2412.06769 (contrast: token-level latent feedback, not depth recurrence).
- Zhang, Recurrent Looped Transformer, 2026 technical report (contrast: temporal recurrence, not used here).
- Karpathy, nanochat, 2025.

## 7. Deliverables

1. Public fork of nanochat with the looped model behind config flags, tests, and run scripts.
2. All run configs, logs, and a results table (CSV) including failed and diverged runs.
3. A paper-style write-up (8 pages + appendix) and a short README summary with F1 and F3.
4. This proposal, committed to the repository **before Phase 1 begins**, as the pre-registration record.

## 8. Timeline (elapsed, part-time)

Week 1: Phase 0. Week 2: Phase 1a. Weeks 3-4: Phase 1b and analysis. Weeks 5-6: Phase 2 and writing. Week 7: external feedback, revisions, release.

---

## 9. Implementation specification for the coding agent

### 9.0 Ground rules

- Work on a branch of nanochat pinned at commit `92d63d4`. Do not pull upstream.
- The stock model MUST remain the default. Every new behavior sits behind config fields whose defaults reproduce stock behavior exactly.
- Do not change tokenizer, dataloader, optimizer internals, loss, or eval code paths except where listed below.
- Do not tune anything against validation data beyond the sweep defined in 3.3.
- Never delete or overwrite logs of failed runs. Append to the results file; do not edit past rows.
- If a requirement here conflicts with what you find in the code, stop and report the conflict. Do not silently pick an interpretation.
- Write tests first for 9.7 items T1-T3, then implement.

### 9.1 Files expected to change

| File | Change |
|---|---|
| `nanochat/gpt.py` | New config fields; layout-driven forward; FLOP and parameter accounting; KV cache indexing |
| `nanochat/engine.py` | KV cache sized by effective depth; pass-through of inference loop count |
| `nanochat/checkpoint_manager.py` | Default-fill new config keys for old checkpoints |
| `scripts/base_train.py` | New CLI flags; reference-derived hyperparameters; seed; random-R training; extra logging |
| `scripts/base_eval.py` | `--num-loops` override for loop-sweep evaluation |
| `tests/test_looped.py` | New |
| `runs/looped_phase0.sh`, `runs/looped_sweep.sh`, `runs/looped_main.sh`, `runs/looped_loopsweep.sh` | New |
| `dev/LOOPED_LOG.md` | New; running lab notebook, one dated entry per work session |

### 9.2 Config

Add to `GPTConfig` (all with defaults that reproduce stock behavior):

```python
n_prelude: int = -1      # -1 => stock behavior: all n_layer blocks applied once
n_core: int = 0
n_coda: int = 0
n_loop: int = 1          # default loop count R used when no override is given
```

Invariant when `n_prelude >= 0`: `n_prelude + n_core + n_coda == n_layer`. `n_layer` continues to mean the number of **unique** blocks and sizes `transformer.h` exactly as today. Blocks are stored in order prelude, core, coda.

Add a helper that returns the visit schedule for a given R as a list of unique-layer indices, for example `2,4x2,2` gives `[0,1, 2,3,4,5, 2,3,4,5, 6,7]`. Effective depth E is its length. With `n_prelude == -1` the schedule is `range(n_layer)`.

Add `model.set_num_loops(R: int)` which sets a plain Python int attribute read by `forward`. It must be a Python int, not a tensor, so `torch.compile(dynamic=False)` specializes one graph per R. Raise the dynamo recompile/cache limit enough for the R values used (at most 8).

### 9.3 Forward pass and treatment of per-layer components

Replace the trunk loop in `GPT.forward` with iteration over the visit schedule. Let `e` be the effective index (0..E-1) and `u` the unique-layer index at that visit.

| Component (stock) | Rule in looped model | Reason |
|---|---|---|
| `transformer.h[u]` weights | Shared across all visits of `u` | The point of the study |
| `resid_lambdas`, `x0_lambdas` | Size `n_layer` (unique). Indexed by `u`, shared across loops. Init formulas unchanged, using `u` and `n_layer` | Allows inference R beyond training R. `x0_lambdas` on core layers serves as the recurrent-depth input injection; do not add a separate injection module |
| Value embeddings | `has_ve(u, n_layer)` on unique layers; table indexed by `u`; the same table is reused on every visit | Keeps lookup parameters tied to unique layers; must be reported separately |
| `window_sizes` | Length E, all full-context under `--window-pattern L`. Assert pattern is `L` when `n_prelude >= 0` | Avoids defining window tiling across loops |
| Backout | Cache the residual after effective index `E // 2` (computed from the current R) | Matches stock `n_layer // 2` when there is no looping |
| Smear, final norm, soft-cap, loss | Unchanged | - |
| KV cache | One cache slot per **effective** layer. `CausalSelfAttention.forward` takes a `cache_idx` argument (= `e`) used for `get_layer_cache` and for the "advance after last layer" check, in place of `self.layer_idx` | The same unique layer sees different inputs on each visit, so its K/V differ per visit |

Gradients to shared weights accumulate through autograd with no special handling. Use full backpropagation through all loops; do not truncate.

### 9.4 Accounting (critical for the paper's claims)

- `estimate_flops()` currently counts each `Linear` once. It MUST count block matmul parameters once **per visit**: sum over the visit schedule of that block's `Linear` parameter count, plus `lm_head`, `ve_gate`, `smear_gate`. Attention FLOPs sum over E effective layers. Apply the same correction to `estimate_decode_flops`, `estimate_prefill_flops`, `kv_bytes_per_token`, and `kv_read_bytes` (these use E).
- For the random-R arm, accumulate training FLOPs per step using the R actually used on that step.
- `num_scaling_params()` keeps returning unique-parameter counts and adds keys `effective_layers` and `unique_layers`. The existing total-parameter assertion must still pass.
- At startup print a table: unique block params, `lm_head`, `wte`, value-embed params, scalars, E, U, FLOPs/token.

### 9.5 Training script

New flags for `scripts/base_train.py`:

- `--layout "P,KxR,C"` or `--layout "N"` (plain N-layer model). Parsed into the config fields.
- `--width-depth 12`: the integer used only to set `model_dim = width_depth * aspect_ratio` (replaces the use of `--depth` for width when `--layout` is given). All Phase 1 arms use 12.
- `--ref-layout "12"`: horizon, auto batch size, LR batch scaling, and weight-decay scaling are computed from a meta-device model built with this layout, not from the arm's own model. `num_iterations` therefore comes out identical for every arm. Assert and print this. (At the pinned commit, `target_tokens` and `weight_decay_scaled` are derived from the arm's own parameter count; that is what this flag overrides.)
- `--horizon-frac 0.4`: multiplies the reference horizon, for the LR sweep. The LR schedule is defined over the shortened run.
- `--matrix-lr-mult 1.0`.
- `--seed`: seeds parameter init only. Data order MUST be identical across seeds and arms; verify by hashing the first 10 batches and logging the hash.
- `--train-loops "1,2,3,4"`: if given, sample R uniformly from this set each optimizer step (same R for all micro-batches within the step), using an RNG seeded independently of `--seed` so every seed sees the same R sequence. Call `model.set_num_loops(R)` before the step's forward passes. Evaluation during training uses `n_loop` from the config (set to 2 for the LR arm).

Additional logging (wandb and a local JSONL): resolved hyperparameters, git hash, layout, U, E, FLOPs/token, tokens/s, peak VRAM, cumulative FLOPs, per-loop residual RMS (mean over batch and positions, measured at the end of each core iteration, every `--eval-every` steps), grad-norm of core vs non-core matrices.

At the end of each run append one row to `results/looped_results.csv`: arm, layout, seed, lr_mult, horizon_frac, steps, tokens, train FLOPs, final val_bpb, CORE, tokens/s, peak VRAM, wall-clock, diverged flag, git hash.

### 9.6 Inference and evaluation

- `Engine` builds `KVCache` with `num_layers = E` for the active R.
- `scripts/base_eval.py --num-loops R` calls `set_num_loops(R)` before evaluation. `runs/looped_loopsweep.sh` evaluates R in {1,2,3,4,5,6,8}.
- `checkpoint_manager.py`: fill missing new config keys with stock defaults so existing checkpoints load.

### 9.7 Tests (`tests/test_looped.py`)

- **T1 Stock equivalence.** With default config, the refactored model and the original `gpt.py` (keep a frozen copy under `tests/fixtures/`) produce identical state-dict keys and, given identical weights and inputs, logits equal to within 1e-5 in fp32 on CPU.
- **T2 Layout equivalence.** Layout `"12"` expressed through the new fields (`n_prelude=12`) equals stock under the same conditions.
- **T3 Untied equivalence.** A looped model `1,2x2,1` equals a stock 6-layer model whose layers 1-2 and 3-4 are loaded with the same weights and whose per-layer scalars and value embeddings are set correspondingly (fp32, 1e-5). This proves looping is exactly weight sharing and nothing else.
- **T4 KV cache consistency.** For a looped model, prefill-then-decode logits through `Engine` match full-sequence forward logits at every position (1e-4, fp32), for R in {1,2,3}.
- **T5 Causality.** Changing token t+1 leaves logits at positions <= t unchanged.
- **T6 Accounting.** `estimate_flops` for `2,4x2,2` equals that of layout `12` at the same width within 0.5% (the difference being `ve_gate` only). Parameter totals match a hand computation.
- **T7 Gradient sharing.** After one backward pass, a core block's weight gradient equals the sum of gradients obtained from the T3 untied copy's two corresponding layers.
- **T8 Determinism of data.** Batch hashes are identical across two different `--seed` values.
- **T9 Smoke.** A 20-step CPU training run for layouts `4`, `1,2x2,1`, and `--train-loops 1,2,3` completes and loss decreases.

All stock tests must continue to pass.

### 9.8 Run scripts

- `looped_phase0.sh`: tokenizer and data setup as in `speedrun.sh`; stock d12 with `--window-pattern L`; refactored `--layout 12`; prints throughput and the val_bpb difference for the Phase 0 gate.
- `looped_sweep.sh`: 7 arms x 3 LR multipliers, `--horizon-frac 0.4`, seed 0.
- `looped_main.sh`: reads selected multipliers from `results/selected_lr.json`; 7 arms x seeds {0,1,2}. Resumable: skips runs already present in the results CSV.
- `looped_loopsweep.sh`: loop-count evaluation for LR and L8 checkpoints.

Single-GPU only: invoke with `python -m scripts.base_train` (no `torchrun`). Start with `--device-batch-size 32` and raise it if VRAM allows; total batch size stays at the reference value through gradient accumulation.

### 9.9 Definition of done for Phase 0

All tests pass; the Phase 0 gate in 3.5 is met; `dev/LOOPED_LOG.md` records measured tokens/s for B12 and L8 and a revised compute estimate for Phase 1; a git tag `prereg-v1` is placed on the commit that contains this document and the passing tests.
