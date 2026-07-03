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

> Status of this document: **Change 1 (MoE) implemented and tested.** Changes 2–3 and the
> large-config sharding are added in subsequent milestones and documented as they land.

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

## Tests

`tests/test_loopmoe.py` (CPU, fp32 — no GPU needed):

- dense path forward/backward/inference unchanged;
- MoE forward shapes; auto hidden-dim sizing;
- **every expert receives a gradient** each step (both balancing modes);
- aux loss propagates to the router and changes the loss value;
- bias-free balancing moves the bias and produces no aux/grad on the bias;
- optimizer partition (router→AdamW, experts→Muon);
- `param_report` active < total for MoE;
- MoE **overfits a fixed batch** (loss decreases) and **shows no expert collapse** over 200 steps.

Run: `NANOCHAT_DTYPE=float32 python -m pytest tests/test_loopmoe.py -v`
