# Coding Agent Task: Add a Looped Mixture-of-Experts architecture to nanochat

## Mission

Modify Andrej Karpathy's **nanochat** repo (https://github.com/karpathy/nanochat) so it can train a **looped Mixture-of-Experts (MoE) language model**, inspired by the *LoopMoE* paper (Chen et al., 2026, arXiv:2606.04438). The result must be a **working, trainable model I can study and experiment with** — not a research-grade controlled baseline comparison. Preserve nanochat's minimal, hackable, single-file-model philosophy: no heavy config frameworks, no unnecessary abstraction layers.

The implementation must:
- Support **both single-GPU and multi-GPU (DDP) training**, and remain runnable on a single 96 GB GPU for the small config.
- Provide **two working configs**: a **small model (<500M active params)** and a **large model (≥10B total params)**.
- **Keep nanochat's existing attention** (RoPE + GQA/MHA + QK-norm + Flash Attention). **Do NOT implement Multi-head Latent Attention (MLA).**

## Before you write any code

1. Clone the repo and **read the actual current source**, especially `nanochat/gpt.py` (contains `GPTConfig`, `CausalSelfAttention`, `MLP`, `Block`, `GPT`), the training entry point (`scripts/base_train.py` or equivalent), the optimizer setup (`setup_optimizer` / the Muon+AdamW hybrid), and `speedrun.sh`. Line numbers and details drift over time — inspect, don't assume.
2. Identify the exact seams: the dense `MLP` class (the ReLU² FFN) is the **MoE injection point**; `Block.forward` and the block loop inside `GPT.forward` are the **looping injection points**; `setup_optimizer` is where new parameters must be assigned to param groups.
3. Confirm how nanochat currently handles: config/CLI args, DDP init, mixed precision, gradient checkpointing (if any), loss return path, and logging. Match those patterns.

## What NOT to do (explicit non-goals)

- Do **not** port MLA or the paper's LoRA-rank capacity-balancing math. We keep nanochat's attention.
- Do **not** build the paper's strict "matched total params / per-token FLOPs / active sublayer ratio" comparison harness. We want a working model, not a controlled A/B.
- Do **not** introduce new dependencies or config frameworks. Stay within nanochat's style.
- Do **not** rewrite the tokenizer, dataloader, eval harness, or web UI. Leave them intact.

## Architecture to implement

Three additive changes on top of stock nanochat. Implement and validate them **in the order below**, each as an independently testable milestone.

### Change 1 — MoE feed-forward layer (replace the dense MLP)

Create an `MoE` module that replaces the dense ReLU² `MLP` inside the block. Follow the DeepSeekMoE pattern that LoopMoE uses:

- **Routed experts:** `n_routed_experts` independent FFNs, each a smaller-hidden-dim copy of nanochat's ReLU² MLP (`c_fc` → `relu().square()` → `c_proj`, no bias).
- **Router:** a single `nn.Linear(n_embd, n_routed_experts, bias=False)` producing logits; softmax; select **top-k** experts per token; combine expert outputs weighted by (normalized) routing weights.
- **Shared experts:** `n_shared_experts` (1–2) always-on FFNs whose outputs are added unconditionally, in addition to the routed top-k.
- **Load balancing:** implement the **auxiliary-loss-free bias-update strategy** from DeepSeek-V3 (per-expert bias added to routing logits, nudged each step toward balanced load) as the default, and *also* support a classic **auxiliary load-balancing loss** as a config flag. Return any aux loss up through `Block.forward` → `GPT.forward` so it can be added to the training loss.
- **Instrumentation:** log per-expert token counts / routing entropy each N steps so router collapse is visible.
- **Implementation note:** a simple `for`-loop over experts is acceptable for a first correct version; leave a clearly-marked TODO for a batched/grouped-GEMM expert path for speed. Correctness first.

**Milestone 1 verification:** a small dense-MoE (loop disabled) trains end-to-end, loss decreases, and no single expert receives a runaway share of tokens over a few hundred steps.

### Change 2 — Sandwich loop layout

Reorganize the stack into three regions:
- `n_prefix_layers` (Lp) non-shared prefix blocks,
- `n_loop_layers` (Lb) **shared** blocks that form the loop body,
- `n_suffix_layers` (Ls) non-shared suffix blocks.

Behavior (per the paper's Eq. 1–2, **no prefix re-injection**):
```
h = prefix_blocks(x)          # run once
h_0 = h                        # loop body starts from prefix output, no re-injection
for k in 1..K:
    h = loop_body_blocks(h; cond=k)   # SAME shared weights each iteration
h = suffix_blocks(h)
```
- Effective depth = `Lp + K * Lb + Ls`; unique layers = `Lp + Lb + Ls`.
- Build `Lp + Lb + Ls` unique `Block`s; in `forward`, reuse the loop-body modules `K` times.
- **Backward pass:** running tied weights K times is BPTT through shared parameters — autograd handles gradient accumulation across iterations correctly; do not hand-code it.
- **Memory:** loop-body activations scale with K. Add **gradient checkpointing on the loop body** (config-toggizable), needed for the large config.
- Make `K` (loop iterations) a config field. Setting `Lp/Ls/Lb` such that there's no loop (K=1) should reduce to a standard stack, useful for debugging.

**Milestone 2 verification:** looped model (MoE from Change 1 inside the loop body) trains end-to-end; increasing K increases effective depth and compute without adding unique parameters; gradient-checkpointed and non-checkpointed runs produce matching losses (within numerical tolerance).

### Change 3 — IterAdaLN (breaks weight-sharing symmetry)

Inside the **loop body only**, replace each pre-normalization RMSNorm with **IterAdaLN**: affine-free RMSNorm followed by token- and iteration-conditioned modulation.

For token `t` at iteration `k`, build a branch-specific condition vector by fusing a token-stream and an iteration-stream (broadcast add):
```
c = Linear_branch(hidden_state_t) + MLP_iter(sinusoidal_PE(k))
```
- Token stream: linear projection of the current per-token hidden state (for the attention branch use the block's input state; for the MoE branch use the post-attention intermediate state).
- Iteration stream: DiT-style fixed sinusoidal positional encoding of the iteration index `k`, passed through a small learnable MLP (`MLP_iter`).

Then modulate:
```
IterAdaLN(h; c) = (1 + gamma(c)) * RMSNorm_affine_free(h) + beta(c)
```
- `gamma`, `beta` (and a residual gate `alpha` for the **attention branch only**) are produced by small, **zero-initialized** output-projection MLPs (AdaLN-Zero style), so the loop starts as near-identity and differentiates during training.
- Apply the residual gate `alpha` to the **attention branch only** (per the paper's Appendix A.3: a symmetric gate on the MoE branch gets absorbed into routing weights and interferes with load balancing). Attention update: `m = h + alpha * Attn(IterAdaLN(h; c_attn))`; MoE update: `h_out = m + MoE(IterAdaLN(m; c_moe))`.
- Omit RMSNorm's static learnable scale where IterAdaLN supplies a dynamic scale.
- Prefix and suffix blocks keep nanochat's standard normalization (they are not shared/looped).

**Milestone 3 verification:** full looped-MoE with IterAdaLN trains stably; modulation MLP outputs start near zero and grow during training; loss is competitive with or better than the Milestone-2 model at equal compute.

## Configuration system

Expose all new hyperparameters through nanochat's existing config/CLI mechanism (extend `GPTConfig` and the training script args). New fields at minimum: `n_routed_experts`, `n_shared_experts`, `top_k`, `expert_hidden_dim` (or an expansion ratio), `load_balance_mode` (`bias_free` | `aux_loss`), `aux_loss_coef`, `n_prefix_layers`, `n_loop_layers`, `n_suffix_layers`, `loop_iters` (K), `use_iter_adaln` (bool), `grad_checkpoint_loop` (bool).

Provide two ready-to-run configs (as scripts or documented arg sets):

**Small (<500M active params), single-GPU friendly:**
- Target ~1B total params, ~300–450M active. Example starting point: `n_embd≈1024`, GQA heads matching nanochat's ratio, `n_prefix=2 / n_loop=2 / n_suffix=2`, `K=4`, `n_routed_experts≈32`, `top_k=6`, `n_shared_experts=1`, small expert hidden dim. Tune to hit the active-param target; document the resulting active/total counts.

**Large (≥10B total params), multi-GPU:**
- Decouple total from active: keep active modest (compute-bound) while scaling the routed-expert pool to reach ≥10B total. Example direction: larger `n_embd`, more routed experts (e.g. 64–128+), same `top_k`, `K=4`, gradient checkpointing on the loop body **on**. This config **requires** the sharded-training path below.
- Print/log the active-param count, physical-active count (shared attention counts once but K× toward FLOPs), and total-param count at startup for both configs.

## Distributed / multi-GPU training

- **Single-GPU:** must work out of the box for the small config.
- **Multi-GPU:** preserve nanochat's DDP path for the small/medium case (replicated model + optimizer).
- **Large (10B+):** DDP with fully replicated optimizer state will not fit. Add a **sharded path** (PyTorch **FSDP** or ZeRO-style optimizer sharding) gated behind a config flag, and shard along sensible boundaries (per-block / per-expert). Keep the DDP path as the default for smaller runs so nothing regresses. Ensure MoE routing and the aux-loss/bias-update reductions are correct across ranks (all-reduce expert load statistics).
- Verify a short multi-GPU run produces the same loss trajectory as single-GPU (within tolerance) for the small config.

## Optimizer wiring (important)

nanochat uses a **Muon + AdamW hybrid**, with Muon on the 2D hidden matmul weights and AdamW on the rest, plus per-group LR scaling. When you add parameters:
- **Router linear, IterAdaLN modulation MLPs, `MLP_iter`, and any embedding-like/1D params** → assign to the **AdamW** group, not Muon.
- **Expert `c_fc` / `c_proj` weight matrices** → these are 2D matmul weights and should go to the **Muon** group like the original MLP weights (verify this empirically; if Muon destabilizes many small experts, fall back to AdamW for experts and note it).
- Preserve zero-init on the AdaLN-Zero output projections after optimizer setup.
- Make sure every new parameter lands in exactly one group (add an assertion that param-group membership partitions all model parameters).

## Known pitfalls to handle

- **Router collapse:** default to the bias-free balancing strategy; log per-expert load; expose the aux-loss fallback.
- **Activation memory across K:** gradient-checkpoint the loop body for the large config.
- **Shared-weight gradients:** accumulate across K iterations automatically via autograd — don't detach the loop.
- **Aux-loss plumbing:** thread any aux loss cleanly from `MoE` → `Block` → `GPT` and add it to the main loss with its coefficient; make it zero-cost when disabled.
- **Numerical stability:** MoE runs are prone to loss spikes — ensure checkpointing is frequent and resumable; keep nanochat's existing grad-clipping / precision settings.
- **FLOPs accounting:** update any FLOP/param-count utility so it reflects looped compute (active N × implied by K) and MoE active vs total; otherwise logging/MFU numbers will be wrong.

## Deliverables

1. Modified `nanochat/gpt.py` (or a clearly separated new module imported by it) implementing `MoE`, the sandwich-loop `GPT.forward`, and `IterAdaLN`, with the dense path still selectable via config so nothing is lost.
2. Updated optimizer setup with correct param grouping and a partition assertion.
3. The FSDP/ZeRO sharded training path for the large config, flag-gated, DDP default preserved.
4. Two runnable config scripts (small + large) mirroring `speedrun.sh` style, printing param-count breakdowns at startup.
5. Minimal tests / assertions for: forward-pass shape correctness, aux-loss propagation, checkpointed-vs-non-checkpointed loss match, single-vs-multi-GPU loss match (small config), and near-zero initial modulation output.
6. A short `LOOPMOE.md` documenting the new config fields, the two example configs, how to switch dense↔MoE and loop on↔off, and the known-pitfall notes above.

## Working style

Deliver **incrementally by milestone** (dense MoE → sandwich loop → IterAdaLN → large-config sharding), running a short training smoke test after each so regressions are caught early. Keep changes minimal and readable in nanochat's spirit; comment the non-obvious parts (routing, the no-re-injection loop, AdaLN-Zero, param grouping). Prefer correctness and clarity over premature optimization, but leave clearly-marked TODOs where a faster expert kernel would later help.