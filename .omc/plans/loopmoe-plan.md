# LoopMoE implementation plan (branch: loopmoe)

Mission: add a trainable looped-MoE architecture to nanochat per INSTRUCTIONS.md,
preserving the dense path and nanochat's single-file style.

## Facts about this fork that shape the design
- No DDP wrapper: `MuonAdamW` all-reduces/reduce-scatters grads internally and shards
  optimizer state (ZeRO-2). "DDP path" == this optimizer path. Every param MUST get a
  grad every step (no find_unused_parameters) -> experts must always execute (empty
  token sets produce zero grads, not None).
- Model built on meta device; ALL data init (params + buffers) happens in `init_weights()`.
- `torch.compile(dynamic=False)` wraps model -> MoE top-k dispatch (data-dependent shapes)
  gets `@torch.compiler.disable`; TODO grouped GEMM later.
- Per-layer machinery (`window_sizes`, `resid_lambdas`, `x0_lambdas`, value_embeds,
  backout at midpoint, KV cache `layer_idx`) -> indexed by UNIQUE layer; KV cache gets
  one slot per layer *execution* (schedule), engine sized via effective depth.
- Muon groups params by shape -> expert matrices stack naturally.
- AdamW reduce_scatter needs shape[0] % world_size == 0 for params >= 1024 numel
  (router (E, n_embd) and adaln matrices satisfy this for typical world sizes).

## Milestones (each: implement -> test -> commit)
1. MoE FFN: `MoE` module (routed ReLU^2 experts, softmax router, top-k, shared experts,
   DeepSeek-V3 bias-free balancing default + aux-loss flag, load/entropy instrumentation).
   Aux loss threaded MoE -> Block -> GPT. Optimizer: experts -> Muon, router -> AdamW,
   strict partition assertion. `expert_hidden_dim=0` auto-sizes to ~match dense FLOPs:
   4*n_embd // (top_k + n_shared).
2. Sandwich loop: n_prefix/n_loop/n_suffix + loop_iters K, execution schedule
   [(block_idx, iter_k)], no prefix re-injection, BPTT via autograd,
   grad_checkpoint_loop (non-reentrant), KV-slot mapping for inference,
   estimate_flops + param_report (total / active / flops-active with K).
3. IterAdaLN (loop body only): affine-free RMSNorm modulated by
   c = token_proj(h) + iter_mlp(sinusoidal_PE(k)); zero-init out heads (AdaLN-Zero);
   alpha gate on attention branch ONLY. NOTE: loop-body attn.c_proj must NOT be
   zero-init when alpha is zero-init (both zero -> dead branch); use standard init there.
   All adaln params -> AdamW.
4. FSDP2 sharded path (`--fsdp`, per-Block fully_shard, plain AdamW groups), flag-gated,
   default MuonAdamW path untouched. Configs runs/loopmoe_small.sh (<500M active,
   ~1B total) + runs/loopmoe_large.sh (>=10B total). LOOPMOE.md. Tests.

## Config additions (GPTConfig + base_train CLI)
n_routed_experts (0=dense), n_shared_experts, top_k, expert_hidden_dim (0=auto),
load_balance_mode (bias_free|aux_loss), aux_loss_coef, router_bias_update_rate,
n_prefix_layers, n_loop_layers, n_suffix_layers, loop_iters, use_iter_adaln,
grad_checkpoint_loop.

## Verification
- CPU tests (fp32): forward shapes, aux-loss propagation, K-independence of param count,
  checkpointed-vs-not loss match, adaln zero-init, optimizer partition, bias-update direction,
  kv-cache prefill-vs-full-forward logits match for looped model.
- GPU (RTX PRO 6000 96GB, single): short synthetic + real smoke training runs per milestone;
  expert-load balance observed over a few hundred steps.
- Multi-GPU loss match: only 1 GPU here -> 2-rank gloo/CPU run if compute_init allows
  (may add gloo init for cpu), else documented as untested-on-this-box.
