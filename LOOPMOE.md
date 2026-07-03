# LoopMoE: a looped Mixture-of-Experts architecture for nanochat

This adds a **looped Mixture-of-Experts (MoE)** language model to nanochat, inspired by
*LoopMoE*. It is three additive, independently-toggleable changes on top of stock nanochat,
all living in `nanochat/gpt.py`:

1. **MoE feed-forward** — replace the dense ReLU² MLP with routed + shared experts.
2. **Sandwich loop** — prefix / shared-loop-body / suffix layout with `K` loop iterations.
3. **IterAdaLN** — token- and iteration-conditioned modulation inside the loop body.

Everything is off by default: with the default config you get **exactly stock nanochat**
(dense MLP, standard stack, standard RMSNorm). Turn features on via `GPTConfig` fields (and
the matching `scripts/base_train.py` CLI flags).

> Status: **Changes 1–3 (MoE, sandwich loop, IterAdaLN) implemented and tested.** The large-config
> FSDP sharding path and the two ready-to-run config scripts are documented below.

---

## Configuration fields (added to `GPTConfig`)

| Field | Default | Meaning |
|---|---|---|
| `n_routed_experts` | `0` | Number of routed experts. **`0` = dense MLP** (MoE disabled). |
| `n_shared_experts` | `0` | Always-on experts added unconditionally to every token. |
| `top_k` | `1` | Experts selected per token by the router. |
| `expert_hidden_dim` | `0` | Per-expert FFN hidden dim. **`0` = auto**: `4*n_embd/(top_k+n_shared)` rounded down to a multiple of 128, so per-token FFN FLOPs roughly match the dense MLP. |
| `load_balance_mode` | `"bias_free"` | `"bias_free"` (DeepSeek-V3 aux-loss-free bias) or `"aux_loss"` (classic Switch aux loss). |
| `aux_loss_coef` | `0.01` | Coefficient on the aux load-balancing loss (only used in `aux_loss` mode). |
| `router_bias_update_rate` | `0.001` | Per-step nudge size for the bias-free balancing bias. |
| `n_prefix_layers` | `0` | Lp: non-shared prefix blocks (loop layout, Change 2). |
| `n_loop_layers` | `0` | Lb: shared loop-body blocks. **`0` = no loop** (standard stack). |
| `n_suffix_layers` | `0` | Ls: non-shared suffix blocks. |
| `loop_iters` | `1` | K: how many times the shared loop body runs. |
| `use_iter_adaln` | `False` | Enable IterAdaLN in the loop body (Change 3). |
| `grad_checkpoint_loop` | `False` | Gradient-checkpoint loop-body executions (large config). |

When `n_loop_layers > 0`, you must set `n_layer == n_prefix_layers + n_loop_layers +
n_suffix_layers` (the unique block count); the model asserts this.

---

## Change 1 — MoE feed-forward layer

`MoE` (in `nanochat/gpt.py`) replaces the dense `MLP` inside a `Block` when
`n_routed_experts > 0`, following the DeepSeekMoE pattern:

- **Routed experts:** `n_routed_experts` independent ReLU² FFNs (`Expert`), each a
  smaller-hidden-dim copy of nanochat's MLP (`c_fc` → `relu().square()` → `c_proj`, no bias).
- **Router:** `nn.Linear(n_embd, n_routed_experts, bias=False)` → softmax affinities → top-k
  selected → outputs combined by the (renormalized) softmax weights of the chosen experts.
- **Shared experts:** `n_shared_experts` always-on FFNs added unconditionally.
- **Load balancing:**
  - *Default — bias-free (DeepSeek-V3):* a per-expert bias buffer is added to the **selection**
    score only (the combine weights still use the raw affinity). After each forward the bias is
    nudged by the sign of each expert's load error (underloaded → up, overloaded → down). No aux
    loss, no extra gradient. Token counts are all-reduced across ranks so the bias reflects
    global load.
  - *Optional — aux loss:* set `load_balance_mode="aux_loss"`. Adds the classic
    Switch-Transformer load-balancing loss `E · Σ fᵢ·Pᵢ`, threaded up through
    `MoE → Block → GPT.forward` and added to the training loss as `aux_loss_coef · aux`.
- **Instrumentation:** each `MoE` stores `last_load` (per-expert token fraction) and
  `last_entropy` (routing entropy) each training step, so router collapse is visible.

**Implementation notes / caveats**

- The expert dispatch is a **correctness-first Python for-loop over experts**. Every expert is
  always invoked (on a possibly-empty token slice), because nanochat's `MuonAdamW` optimizer
  stacks per-parameter gradients and a `None` grad (an expert that received no tokens) would
  crash it. An empty slice still produces a defined (zero) gradient. There is a clearly-marked
  `TODO` for a batched/grouped-GEMM expert path for speed.
- `MoE.forward` is wrapped in `@torch.compiler.disable` because top-k routing is
  data-dependent (dynamic shapes); the rest of the model still compiles.

**Optimizer wiring**

- Expert `c_fc` / `c_proj` (2D matmul weights) → **Muon** group (like the original MLP weights).
- Router weight (classifier-like) → **AdamW** group.
- `setup_optimizer` ends with a **partition assertion**: every model parameter lands in exactly
  one optimizer group.
- `expert_bias` is a **buffer, not a parameter** — it is nudged manually, never optimized.
  It is persistent, so it survives checkpoint save/resume.

**Switching dense ↔ MoE:** set `n_routed_experts` to `0` (dense) or `>0` (MoE). Nothing else
is required; the dense path is byte-for-byte the original nanochat.

---

---

## Change 2 — sandwich loop layout

The stack is reorganized into three regions (all sizes are config fields):

```
h = prefix_blocks(x)            # Lp non-shared blocks, run once
h_0 = h                         # loop starts from the prefix output; NO prefix re-injection
for k in 1..K:                  # loop_iters = K
    h = loop_body_blocks(h)     # Lb SHARED blocks, same weights every iteration
h = suffix_blocks(h)            # Ls non-shared blocks, run once
```

- **Effective depth** = `Lp + K·Lb + Ls`; **unique layers** = `Lp + Lb + Ls == n_layer`.
- The model builds `n_layer` unique `Block`s and **reuses** the loop-body modules `K` times, so
  the unique parameter count is **independent of K** (verified by a test). More `K` buys more
  depth/compute at zero extra parameters.
- **BPTT:** running tied weights `K` times is backprop-through-time over shared parameters;
  autograd accumulates the gradients across iterations — the loop is never detached.
- **No prefix re-injection:** the loop iterates `h ← f(h)`, it does not add `h_0` back each step.
  (nanochat's existing `x0_lambdas` feature blends the *token embedding*, not the prefix output,
  so it does not constitute prefix re-injection.)
- **`K=1` reduces exactly to a standard stack** — a looped config with `K=1` is bitwise-identical
  to a plain stack of the same blocks (a test asserts this), which is handy for debugging.

**Implementation details**

- A single **execution schedule** `[(unique_block_idx, iter_k), ...]` drives both `forward` and
  the FLOPs/param accounting. Per-block machinery (window sizes, resid/x0 scalars, value
  embeddings) is indexed by the **unique** block index; the **KV-cache slot** and the backout
  midpoint use the **execution position**, so each loop execution gets its own cache slot. The
  inference `Engine` sizes the KV cache by effective depth (`model.n_exec`).
- **Gradient checkpointing** (`grad_checkpoint_loop=True`): loop-body executions are wrapped in
  non-reentrant `torch.utils.checkpoint`, recomputing their forward in backward to bound the
  activation memory that otherwise scales with `K` (needed for the large config). Checkpointed
  and non-checkpointed runs produce matching loss **and** gradients (tested).
- **MoE + checkpointing interaction (subtle):** the aux-loss-free per-expert bias is **not**
  mutated inside `forward`. Because a shared loop block is checkpointed multiple times per step
  and recompute runs in reverse order, mutating the bias in `forward` would make an earlier
  execution's recompute diverge from its saved activations. Instead each `MoE` stashes its
  (all-reduced) token counts, and `GPT.update_router_bias()` applies the nudge once per optimizer
  step, after `backward()` — this is also how DeepSeek-V3 does it. The training loop
  (`scripts/base_train.py`) calls it right after `optimizer.step()`.

**Turning the loop on/off:** set `n_loop_layers` to `0` (no loop, standard stack) or `>0`. When
looping you must set `n_layer = n_prefix_layers + n_loop_layers + n_suffix_layers`; the model
asserts this. `loop_iters` (K) controls the number of loop-body applications.

**Startup logging** (`base_train.py`) prints total params, physical-active params (loop body
counted once), FLOPs-active params (loop body counted `K×`), and effective vs unique depth.

---

## Change 3 — IterAdaLN (breaks weight-sharing symmetry)

Inside the **loop body only**, each pre-normalization RMSNorm is replaced with **IterAdaLN**:
an affine-free RMSNorm followed by token- and iteration-conditioned modulation. This lets each
loop iteration behave differently despite sharing weights.

For token `t` at iteration `k`, a branch-specific condition vector fuses a token stream and an
iteration stream (broadcast add):

```
c = adaln_token_proj(hidden_state_t) + adaln_iter_mlp(sinusoidal_PE(k))
IterAdaLN(h; c) = (1 + gamma(c)) * RMSNorm_affine_free(h) + beta(c)
```

- **Token stream:** a linear projection of the current per-token hidden state. The attention
  branch conditions on the block **input** state `h`; the MoE branch conditions on the
  **post-attention** state `m`.
- **Iteration stream:** a DiT-style fixed sinusoidal PE of the iteration index `k`, passed
  through a small learnable MLP (`adaln_iter_mlp`). The PE table is **precomputed once** as a
  buffer and indexed by `k` (rather than built from a Python int inside `forward`), so
  `torch.compile` does not recompile per iteration.
- **AdaLN-Zero:** `gamma`, `beta`, and (attention branch only) a residual gate `alpha` come from
  **zero-initialized** output heads, so the loop starts as near-identity and differentiates
  during training. Updates:
  ```
  m     = h + alpha * Attn(IterAdaLN(h; c_attn))   # attention branch: GATED
  h_out = m +         MoE (IterAdaLN(m; c_moe))     # MoE branch: NO gate
  ```
  The gate is on the **attention branch only** — a symmetric gate on the MoE branch would be
  absorbed into the routing weights and interfere with load balancing.
- Affine-free RMSNorm (nanochat's `norm`) supplies no static scale; IterAdaLN supplies the
  dynamic one. Prefix and suffix blocks keep standard normalization.

**Implementation details / gotchas**

- nanochat's custom `Linear` is **bias-free by design** (it overrides `forward` and never applies
  a bias), so all IterAdaLN layers are `bias=False`. AdaLN-Zero still works by zeroing the head
  **weights** (zero output at init).
- **Dead-branch avoidance:** the attention `c_proj` in an adaln loop-body block is **not**
  zero-initialized (it uses a small 0.4× init). If both `c_proj` and the `alpha` gate were zero,
  neither would ever receive a gradient (the branch would be permanently dead). A small non-zero
  `c_proj` keeps the branch alive while the zero `alpha` keeps it near-identity at init.
- **AdaLN needs a lower LR (important):** the modulation `(1+gamma)`/`beta` and the `alpha` gate
  are unbounded; at nanochat's matrix LR they diverge in bf16 within a few steps. IterAdaLN
  params therefore get their **own AdamW group at `0.1×` the matrix LR** (`ADALN_LR_SCALE` in
  `setup_optimizer`). MoE routers keep the full matrix LR (stable). This was necessary to make
  bf16 training on the GPU stable — verified empirically.
- All IterAdaLN params (token proj, iter MLP, gamma/beta/alpha heads) → **AdamW**, never Muon.

**Turning it on/off:** `use_iter_adaln=True` (requires a loop, i.e. `n_loop_layers>0`).

---

## Optimizer wiring summary

| Params | Optimizer group | LR |
|---|---|---|
| Attention + MoE **expert** matrices (`c_fc`/`c_proj`) | Muon | `matrix_lr` |
| MoE **router** | AdamW | `matrix_lr · dscale` |
| **IterAdaLN** (token proj, iter MLP, gamma/beta/alpha) | AdamW | `matrix_lr · dscale · 0.1` |
| Embeddings, lm_head, value embeds, scalars | AdamW | (unchanged from stock) |

`setup_optimizer` asserts the groups **partition** all model parameters (each param in exactly
one group). The `MoE.expert_bias` is a **buffer, not a parameter**, so it is never in any group.

---

## Tests

`tests/test_loopmoe.py` (CPU, fp32 — no GPU needed):

- dense path forward/backward/inference unchanged;
- MoE forward shapes; auto hidden-dim sizing;
- **every expert receives a gradient** each step (both balancing modes);
- aux loss propagates to the router and changes the loss value;
- bias-free balancing moves the bias and produces no aux/grad on the bias;
- optimizer partition (router→AdamW, experts→Muon);
- `param_report` active < total for MoE;
- MoE **overfits a fixed batch** (loss decreases) and **shows no expert collapse** over 200 steps;
- loop schedule & effective depth; all loop params get grads;
- **K-independence** of unique params (more `K` ⇒ more FLOPs, same params);
- **K=1 loop == standard stack** (bitwise);
- **checkpointed == non-checkpointed** loss and grads (dense + MoE loop bodies);
- bias nudge applies exactly once under checkpointing;
- **looped KV-cache inference == full forward**;
- looped-MoE overfits a fixed batch;
- IterAdaLN only in the loop body; **near-zero modulation at init** (identity + closed gate);
- loop body near-identity at init; adaln params → AdamW; no None grads;
- MoE branch has no alpha gate; checkpoint match; modulation grows during training;
- optimizer partition holds with adaln present.

Run: `NANOCHAT_DTYPE=float32 python -m pytest tests/test_loopmoe.py -v`
