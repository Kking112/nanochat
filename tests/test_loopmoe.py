"""
Tests for the LoopMoE additions (MoE feed-forward, sandwich loop, IterAdaLN).

These run on CPU in fp32 so they need no GPU and stay CI-friendly. Set the compute dtype
to float32 at import time (the model reads COMPUTE_DTYPE at construction for some buffers).

Run:  NANOCHAT_DTYPE=float32 python -m pytest tests/test_loopmoe.py -v
The conftest-free way: we set the env var here before importing nanochat.
"""

import os
os.environ.setdefault("NANOCHAT_DTYPE", "float32")

import torch
import pytest

from nanochat.gpt import GPT, GPTConfig, MoE, MLP
from nanochat.engine import KVCache


def build(**kw):
    """Small CPU model. Defaults keep it tiny and fast."""
    kw.setdefault("n_layer", 4)
    cfg = GPTConfig(
        sequence_len=64, vocab_size=128, n_head=4, n_kv_head=2,
        n_embd=128, window_pattern="L", **kw
    )
    m = GPT(cfg)          # not meta: build directly on CPU for tests
    m.to_empty(device="cpu")
    m.init_weights()
    return m, cfg


# looped defaults: Lp=1, Lb=2, Ls=1 => n_layer=4 unique blocks
LOOP_KW = dict(n_layer=4, n_prefix_layers=1, n_loop_layers=2, n_suffix_layers=1)


def a_batch(cfg, B=2, T=32):
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    tgt = torch.randint(0, cfg.vocab_size, (B, T))
    return idx, tgt


# --------------------------------------------------------------------------------------
# Dense path must be untouched
# --------------------------------------------------------------------------------------
def test_dense_forward_and_backward():
    m, cfg = build()
    assert isinstance(m.transformer.h[0].mlp, MLP)
    idx, tgt = a_batch(cfg)
    loss = m(idx, tgt)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    # inference path returns logits of the right shape
    m.eval()
    with torch.no_grad():
        logits = m(idx)
    assert logits.shape == (idx.size(0), idx.size(1), cfg.vocab_size)


# --------------------------------------------------------------------------------------
# MoE forward-pass correctness
# --------------------------------------------------------------------------------------
def test_moe_forward_shapes():
    m, cfg = build(n_routed_experts=8, n_shared_experts=1, top_k=2)
    assert isinstance(m.transformer.h[0].mlp, MoE)
    idx, tgt = a_batch(cfg)
    loss = m(idx, tgt)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_moe_auto_hidden_dim_matches_dense_flops():
    # top_k + shared = 3 active experts; auto hidden ~ 4*n_embd/3 rounded to mult of 128
    m, cfg = build(n_routed_experts=8, n_shared_experts=1, top_k=2, expert_hidden_dim=0)
    moe = m.transformer.h[0].mlp
    assert moe.expert_hidden_dim % 128 == 0
    assert moe.expert_hidden_dim > 0


@pytest.mark.parametrize("mode", ["bias_free", "aux_loss"])
def test_moe_all_experts_receive_grad(mode):
    """Critical: nanochat's optimizer stacks per-param grads, so a None grad (an expert that
    got no tokens) would crash it. Every expert weight must have a defined grad each step."""
    m, cfg = build(n_routed_experts=16, n_shared_experts=1, top_k=2, load_balance_mode=mode)
    idx, tgt = a_batch(cfg, B=1, T=8)  # few tokens => many experts get zero tokens
    m(idx, tgt).backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert missing == [], f"params with None grad: {missing}"


def test_aux_loss_propagates_to_router():
    m, cfg = build(n_routed_experts=8, n_shared_experts=0, top_k=2, load_balance_mode="aux_loss", aux_loss_coef=0.1)
    idx, tgt = a_batch(cfg)
    m(idx, tgt).backward()
    g = m.transformer.h[0].mlp.router.weight.grad
    assert g is not None and g.norm() > 0, "router should get gradient from the aux loss"


def test_aux_loss_changes_loss_value():
    """With aux_loss mode the reported loss includes coef*aux; bias_free returns no aux."""
    torch.manual_seed(0)
    m_bias, cfg = build(n_routed_experts=8, top_k=2, load_balance_mode="bias_free")
    torch.manual_seed(0)
    m_aux, _ = build(n_routed_experts=8, top_k=2, load_balance_mode="aux_loss", aux_loss_coef=1.0)
    idx, tgt = a_batch(cfg)
    # same init (same seed) => CE identical; aux model's loss should be strictly larger
    lb = m_bias(idx, tgt).item()
    la = m_aux(idx, tgt).item()
    assert la > lb, f"aux_loss ({la}) should exceed bias_free ({lb}) by the aux term"


def test_bias_free_updates_bias_toward_balance():
    """The auxiliary-loss-free strategy nudges per-expert bias; underloaded experts rise."""
    m, cfg = build(n_routed_experts=8, top_k=2, load_balance_mode="bias_free", router_bias_update_rate=0.01)
    moe = m.transformer.h[0].mlp
    assert torch.count_nonzero(moe.expert_bias) == 0  # starts at zero
    idx, tgt = a_batch(cfg, B=4, T=32)
    m(idx, tgt)                 # forward stashes the load counts
    m.update_router_bias()      # deferred nudge (called once per optimizer step)
    assert torch.count_nonzero(moe.expert_bias) > 0, "expert_bias should move after a step"
    # magnitude of each nudge is exactly the update rate
    assert torch.allclose(moe.expert_bias.abs().unique(), torch.tensor([0.01]), atol=1e-6) or \
           moe.expert_bias.abs().max() <= 0.01 + 1e-6


def test_bias_free_produces_no_aux_and_no_grad_to_bias():
    m, cfg = build(n_routed_experts=8, top_k=2, load_balance_mode="bias_free")
    idx, tgt = a_batch(cfg)
    m(idx, tgt).backward()
    # expert_bias is a buffer, not a parameter -> never has a grad, never in optimizer
    assert not m.transformer.h[0].mlp.expert_bias.requires_grad


# --------------------------------------------------------------------------------------
# Optimizer wiring
# --------------------------------------------------------------------------------------
def test_optimizer_partition_dense():
    m, cfg = build()
    opt = m.setup_optimizer()  # construction runs the partition assertion
    grouped = [id(p) for g in opt.param_groups for p in g["params"]]
    assert set(grouped) == {id(p) for p in m.parameters()}
    assert len(grouped) == len(set(grouped))


def test_optimizer_partition_moe_router_in_adamw():
    m, cfg = build(n_routed_experts=8, n_shared_experts=1, top_k=2)
    opt = m.setup_optimizer()
    router_w = m.transformer.h[0].mlp.router.weight
    expert_fc = m.transformer.h[0].mlp.experts[0].c_fc.weight
    # router must be in an adamw group, expert matrices in a muon group
    def group_kind_of(param):
        for g in opt.param_groups:
            if any(p is param for p in g["params"]):
                return g["kind"]
        return None
    assert group_kind_of(router_w) == "adamw"
    assert group_kind_of(expert_fc) == "muon"


# --------------------------------------------------------------------------------------
# Param/flops accounting
# --------------------------------------------------------------------------------------
def test_param_report_active_less_than_total_for_moe():
    m, cfg = build(n_routed_experts=32, n_shared_experts=1, top_k=2)
    rep = m.param_report()
    assert rep["active_params_flops"] < rep["total_params"], "MoE active < total"
    assert rep["active_params_physical"] <= rep["active_params_flops"]
    assert rep["effective_depth"] == cfg.n_layer  # no loop yet


# --------------------------------------------------------------------------------------
# The model actually learns (loss decreases) — Milestone 1 verification
# --------------------------------------------------------------------------------------
def test_moe_overfits_fixed_batch():
    """A small dense-MoE should drive the loss down on a fixed batch (Adam, plain, CPU)."""
    torch.manual_seed(0)
    m, cfg = build(n_routed_experts=8, n_shared_experts=1, top_k=2)
    m.train()
    idx, tgt = a_batch(cfg, B=2, T=32)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for step in range(60):
        opt.zero_grad(set_to_none=True)
        loss = m(idx, tgt)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
    last = loss.item()
    assert last < first * 0.6, f"loss did not decrease enough: {first:.3f} -> {last:.3f}"


def test_no_expert_collapse_over_steps():
    """Over a few hundred steps of bias_free routing on varied data, no single expert should
    capture a runaway share of tokens (router-collapse guard)."""
    torch.manual_seed(0)
    m, cfg = build(n_routed_experts=8, n_shared_experts=1, top_k=2, load_balance_mode="bias_free")
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for step in range(200):
        idx, tgt = a_batch(cfg, B=2, T=32)
        opt.zero_grad(set_to_none=True)
        m(idx, tgt).backward()
        opt.step()
        m.update_router_bias()  # aux-loss-free balancing nudge, once per step
    load = m.transformer.h[0].mlp.last_load
    # uniform share is 1/8 = 0.125; require no expert above ~3x uniform
    assert load.max() < 0.375, f"an expert captured a runaway share: {load.tolist()}"


# ======================================================================================
# Change 2 — sandwich loop layout
# ======================================================================================
def test_loop_schedule_and_effective_depth():
    m, cfg = build(loop_iters=3, **LOOP_KW)
    # prefix once + body(2) x K(3) + suffix once = 1 + 6 + 1 = 8 executions
    assert m.n_exec == 8
    assert m.exec_schedule[0] == (0, 0)          # prefix
    assert m.exec_schedule[-1] == (3, 0)         # suffix
    # loop body blocks are 1 and 2, appearing with iteration indices 0,1,2
    body = [(b, k) for (b, k) in m.exec_schedule if b in (1, 2)]
    assert body == [(1, 0), (2, 0), (1, 1), (2, 1), (1, 2), (2, 2)]


def test_loop_forward_backward_all_grads_defined():
    m, cfg = build(loop_iters=3, n_routed_experts=8, top_k=2, **LOOP_KW)
    idx, tgt = a_batch(cfg)
    m(idx, tgt).backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert missing == []


def test_K_independent_unique_params_but_more_flops():
    m3, cfg = build(loop_iters=3, **LOOP_KW)
    m6, _ = build(loop_iters=6, **LOOP_KW)
    r3, r6 = m3.param_report(), m6.param_report()
    assert r3["total_params"] == r6["total_params"], "K must not change unique param count"
    assert r6["effective_depth"] > r3["effective_depth"]
    assert m6.estimate_flops() > m3.estimate_flops(), "more K => more compute"


def test_K1_loop_reduces_to_standard_stack():
    """A looped config with K=1 in the same block order must be bitwise-identical to a plain
    stack with the same weights (useful for debugging)."""
    torch.manual_seed(1)
    ml, cfg = build(loop_iters=1, **LOOP_KW)
    md, _ = build()  # plain 4-layer stack, same architecture
    md.load_state_dict(ml.state_dict())
    ml.eval()
    md.eval()
    idx, _ = a_batch(cfg)
    with torch.no_grad():
        assert torch.allclose(ml(idx), md(idx), atol=1e-5)


@pytest.mark.parametrize("moe", [False, True])
def test_grad_checkpoint_matches_non_checkpoint(moe):
    """Checkpointed and non-checkpointed loop bodies must produce matching loss and grads."""
    extra = dict(n_routed_experts=8, top_k=2, load_balance_mode="aux_loss") if moe else {}
    def run(flag):
        torch.manual_seed(2)
        m, cfg = build(loop_iters=3, grad_checkpoint_loop=flag, **LOOP_KW, **extra)
        m.train()
        idx, tgt = a_batch(cfg)
        loss = m(idx, tgt)
        loss.backward()
        g = m.transformer.h[1].attn.c_q.weight.grad.clone()  # loop-body grad
        return loss.item(), g
    l0, g0 = run(False)
    l1, g1 = run(True)
    assert abs(l0 - l1) < 1e-5, f"loss mismatch: {l0} vs {l1}"
    assert torch.allclose(g0, g1, atol=1e-5), f"grad mismatch max {(g0-g1).abs().max().item()}"


def test_checkpoint_bias_free_updates_bias_once():
    """Checkpointing recomputes the loop-body forward; the bias-free nudge must still apply
    exactly once per step (guarded by torch.is_grad_enabled)."""
    def bias_after_one_step(flag):
        torch.manual_seed(4)
        m, cfg = build(loop_iters=2, grad_checkpoint_loop=flag,
                       n_routed_experts=8, top_k=2, load_balance_mode="bias_free",
                       router_bias_update_rate=0.01, **LOOP_KW)
        m.train()
        idx, tgt = a_batch(cfg)
        m(idx, tgt).backward()
        m.update_router_bias()  # deferred nudge, after backward
        # a loop-body MoE (block index 1)
        return m.transformer.h[1].mlp.expert_bias.abs().max().item()
    assert abs(bias_after_one_step(False) - bias_after_one_step(True)) < 1e-9


def test_loop_kv_cache_matches_full_forward():
    """Incremental (KV-cache) inference on a looped model must match a full forward, which
    checks that each loop execution uses its own cache slot (kv_layer_idx == exec position)."""
    torch.manual_seed(3)
    m, cfg = build(loop_iters=3, **LOOP_KW)
    m.eval()
    seq = torch.randint(0, cfg.vocab_size, (1, 20))
    with torch.no_grad():
        full = m(seq)[:, -1, :]
        kv = KVCache(batch_size=1, num_heads=cfg.n_kv_head, seq_len=20,
                     head_dim=cfg.n_embd // cfg.n_head, num_layers=m.n_exec,
                     device="cpu", dtype=torch.float32)
        cached = m(seq, kv_cache=kv)[:, -1, :]
    assert torch.allclose(full, cached, atol=1e-4), f"max diff {(full-cached).abs().max().item()}"


def test_loop_config_requires_consistent_n_layer():
    with pytest.raises(AssertionError):
        # n_layer=4 but prefix+loop+suffix = 1+2+2 = 5 -> must fail
        cfg = GPTConfig(sequence_len=64, vocab_size=128, n_layer=4, n_head=4, n_kv_head=2,
                        n_embd=128, window_pattern="L",
                        n_prefix_layers=1, n_loop_layers=2, n_suffix_layers=2, loop_iters=2)
        GPT(cfg)


def test_looped_moe_overfits_fixed_batch():
    """End-to-end Milestone 2 check: a looped MoE learns (loss decreases) on a fixed batch."""
    torch.manual_seed(0)
    m, cfg = build(loop_iters=3, n_routed_experts=8, n_shared_experts=1, top_k=2, **LOOP_KW)
    m.train()
    idx, tgt = a_batch(cfg, B=2, T=32)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for step in range(60):
        opt.zero_grad(set_to_none=True)
        loss = m(idx, tgt)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
    assert loss.item() < first * 0.6, f"looped MoE did not learn: {first:.3f} -> {loss.item():.3f}"


# ======================================================================================
# Change 3 — IterAdaLN
# ======================================================================================
def test_adaln_only_in_loop_body():
    m, cfg = build(use_iter_adaln=True, **LOOP_KW)
    flags = [b.use_adaln for b in m.transformer.h]
    # LOOP_KW: Lp=1, Lb=2, Ls=1 -> only blocks 1,2 (the loop body) are adaln
    assert flags == [False, True, True, False]
    # prefix/suffix blocks have no adaln submodules
    assert not hasattr(m.transformer.h[0], "adaln_attn")
    assert not hasattr(m.transformer.h[3], "adaln_attn")


def test_adaln_near_zero_modulation_at_init():
    """AdaLN-Zero: gamma/beta/alpha heads are zero at init, so modulation is identity and the
    alpha gate is closed (modulation MLP outputs start near zero and grow during training)."""
    from nanochat.gpt import norm
    m, cfg = build(use_iter_adaln=True, loop_iters=3, **LOOP_KW)
    blk = m.transformer.h[1]
    h = torch.randn(2, 8, cfg.n_embd)
    pe = m.iter_pe[1]
    out, alpha = blk.adaln_attn(h, pe)
    assert alpha.abs().max().item() == 0.0, "alpha gate must start closed (zero)"
    assert torch.allclose(out, norm(h), atol=1e-6), "modulation must start as plain RMSNorm"
    # MoE-branch adaln (no gate) also starts as identity
    assert torch.allclose(m.transformer.h[1].adaln_moe(h, pe), norm(h), atol=1e-6)


def test_adaln_loop_body_near_identity_at_init():
    """With zero alpha gate and (adaln) zero heads, a loop-body block starts near-identity."""
    m, cfg = build(use_iter_adaln=True, loop_iters=3, **LOOP_KW)
    m.eval()
    x = torch.randn(2, 8, cfg.n_embd)
    cos_sin = (m.cos[:, :8], m.sin[:, :8])
    with torch.no_grad():
        y, _ = m.transformer.h[1](x, None, cos_sin, (-1, 0), None, 1, m.iter_pe[2])
    assert (y - x).abs().max().item() < 1e-5


def test_adaln_params_go_to_adamw_not_muon():
    m, cfg = build(use_iter_adaln=True, **LOOP_KW)
    opt = m.setup_optimizer()
    def kind_of(p):
        for g in opt.param_groups:
            if any(pp is p for pp in g["params"]):
                return g["kind"]
    blk = m.transformer.h[1]
    for p in (blk.adaln_attn.adaln_gamma.weight, blk.adaln_attn.adaln_token_proj.weight,
              blk.adaln_attn.adaln_alpha.weight, blk.adaln_moe.adaln_beta.weight):
        assert kind_of(p) == "adamw"


def test_adaln_no_none_grads_and_all_params_defined():
    m, cfg = build(use_iter_adaln=True, n_routed_experts=8, top_k=2, **LOOP_KW)
    idx, tgt = a_batch(cfg)
    m(idx, tgt).backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert missing == [], f"None grads: {missing}"


def test_adaln_moe_branch_has_no_alpha_gate():
    """Per the paper, only the attention branch gets a residual gate; the MoE branch does not."""
    m, cfg = build(use_iter_adaln=True, **LOOP_KW)
    assert m.transformer.h[1].adaln_attn.adaln_alpha is not None   # attn: gated
    assert m.transformer.h[1].adaln_moe.adaln_alpha is None        # moe: no gate


def test_adaln_grad_checkpoint_matches():
    def run(flag):
        torch.manual_seed(2)
        m, cfg = build(use_iter_adaln=True, grad_checkpoint_loop=flag, **LOOP_KW)
        m.train()
        idx, tgt = a_batch(cfg)
        loss = m(idx, tgt)
        loss.backward()
        g = m.transformer.h[1].adaln_attn.adaln_token_proj.weight.grad.clone()
        return loss.item(), g
    l0, g0 = run(False)
    l1, g1 = run(True)
    assert abs(l0 - l1) < 1e-5
    assert torch.allclose(g0, g1, atol=1e-5)


def test_adaln_modulation_grows_during_training():
    """The AdaLN-Zero heads start at zero and should move away from zero as training proceeds."""
    torch.manual_seed(0)
    m, cfg = build(use_iter_adaln=True, n_routed_experts=8, top_k=2, **LOOP_KW)
    m.train()
    gamma0 = m.transformer.h[1].adaln_attn.adaln_gamma.weight.abs().sum().item()
    assert gamma0 == 0.0
    idx, tgt = a_batch(cfg)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        m(idx, tgt).backward()
        opt.step()
    gamma1 = m.transformer.h[1].adaln_attn.adaln_gamma.weight.abs().sum().item()
    assert gamma1 > 0.0, "modulation heads should differentiate from zero during training"


def test_optimizer_partition_with_adaln():
    m, cfg = build(use_iter_adaln=True, n_routed_experts=8, top_k=2, **LOOP_KW)
    opt = m.setup_optimizer()  # runs the partition assertion internally
    grouped = [id(p) for g in opt.param_groups for p in g["params"]]
    assert set(grouped) == {id(p) for p in m.parameters()}
    assert len(grouped) == len(set(grouped))
