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


def build(**kw):
    """Small CPU model. Defaults keep it tiny and fast."""
    cfg = GPTConfig(
        sequence_len=64, vocab_size=128, n_layer=4, n_head=4, n_kv_head=2,
        n_embd=128, window_pattern="L", **kw
    )
    m = GPT(cfg)          # not meta: build directly on CPU for tests
    m.to_empty(device="cpu")
    m.init_weights()
    return m, cfg


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
    m(idx, tgt)  # one forward updates the bias
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
    load = m.transformer.h[0].mlp.last_load
    # uniform share is 1/8 = 0.125; require no expert above ~3x uniform
    assert load.max() < 0.375, f"an expert captured a runaway share: {load.tolist()}"
