"""
Tests for the looped (weight-shared depth) layouts, spec section 9.7 of looped_nanochat_proposal.md.

python -m pytest tests/test_looped.py -v

Notation: layout "P,KxR,C" = P prelude blocks, K core blocks applied R times, C coda blocks.

All model tests run on CPU in fp32 with the SDPA attention path. Both have to be forced:
COMPUTE_DTYPE is resolved once at import and is bf16 on any machine with a recent GPU, even
for CPU tensors, and FA3 may be selected on GPUs it has no kernel for.

GPT.init_weights() zeroes every c_proj, so at init each block is a no-op on the residual
stream and any equivalence test would pass vacuously. Every test below therefore re-randomises
ALL parameters first (see randomize_).
"""

import importlib.util
from pathlib import Path

import pytest
import torch

import nanochat.engine as engine_module
import nanochat.flash_attention as fa_module
import nanochat.gpt as gpt_module
from nanochat.gpt import GPT, GPTConfig

# The frozen copy of nanochat/gpt.py at the pinned commit 92d63d4, loaded by path
# (there is no tests/__init__.py).
_STOCK_PATH = Path(__file__).parent / "fixtures" / "gpt_stock.py"
_spec = importlib.util.spec_from_file_location("gpt_stock", _STOCK_PATH)
gpt_stock = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gpt_stock)

TINY = dict(sequence_len=32, vocab_size=256, n_head=4, n_kv_head=4, n_embd=64, window_pattern="L")


@pytest.fixture(autouse=True)
def fp32_sdpa(monkeypatch):
    """Force fp32 compute and the SDPA attention path for every test in this file."""
    for module in (gpt_module, engine_module, gpt_stock):
        monkeypatch.setattr(module, "COMPUTE_DTYPE", torch.float32)
    monkeypatch.setattr(fa_module, "_override_impl", "sdpa")
    monkeypatch.setattr(fa_module, "USE_FA3", False)


def build(gpt_cls, config):
    """Same meta-device -> to_empty -> init_weights sequence the training script uses."""
    with torch.device("meta"):
        model = gpt_cls(config)
    model.to_empty(device="cpu")
    model.init_weights()
    return model.eval()


@torch.no_grad()
def randomize_(model, seed):
    """Overwrite every parameter with non-degenerate values so that no block is a no-op."""
    g = torch.Generator().manual_seed(seed)
    for name, p in model.named_parameters():
        if name == "resid_lambdas":
            p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=g))
        elif name in ("x0_lambdas", "smear_lambda", "backout_lambda"):
            p.copy_(0.3 + 0.2 * torch.randn(p.shape, generator=g))
        else:
            p.copy_(0.1 * torch.randn(p.shape, generator=g))
    return model


def tokens(seed, B=2, T=16):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, TINY["vocab_size"], (B, T), generator=g)


def untie(looped):
    """
    Build the stock (untied) model that a looped model unrolls to: one stock layer per visit,
    each loaded with the weights, per-layer scalars and value embedding of the unique layer
    visited there.

    Only possible when the value-embedding pattern lines up, i.e. has_ve(e, E) for the
    unrolled model equals has_ve(u, U) for the unique layer visited at e. That holds for every
    layout in the proposal (even K). A layout with odd K cannot be written as a stock model.
    """
    schedule = looped.visit_schedule()
    E = len(schedule)
    stock = build(gpt_stock.GPT, gpt_stock.GPTConfig(n_layer=E, **TINY))
    src = looped.state_dict()
    dst = {}
    for key in stock.state_dict():
        parts = key.split(".")
        if key in ("resid_lambdas", "x0_lambdas"):
            dst[key] = torch.stack([src[key][u] for u in schedule])
        elif parts[:2] == ["transformer", "h"]:
            u = schedule[int(parts[2])]
            dst[key] = src[".".join(["transformer", "h", str(u)] + parts[3:])].clone()
        elif parts[0] == "value_embeds":
            u = schedule[int(parts[1])]
            dst[key] = src[".".join(["value_embeds", str(u)] + parts[2:])].clone()
        else:
            dst[key] = src[key].clone()
    stock.load_state_dict(dst, strict=True)
    return stock, schedule


# -----------------------------------------------------------------------------
# T1 Stock equivalence

def test_t1_stock_equivalence():
    """Default config: refactored gpt.py == frozen stock gpt.py (same keys, same logits)."""
    new = build(GPT, GPTConfig(n_layer=6, **TINY))
    old = build(gpt_stock.GPT, gpt_stock.GPTConfig(n_layer=6, **TINY))
    assert list(new.state_dict().keys()) == list(old.state_dict().keys())
    randomize_(new, seed=1)
    old.load_state_dict(new.state_dict(), strict=True)
    idx = tokens(seed=2)
    torch.testing.assert_close(new(idx), old(idx), atol=1e-5, rtol=0)
    torch.testing.assert_close(new(idx, targets=idx), old(idx, targets=idx), atol=1e-5, rtol=0)


# -----------------------------------------------------------------------------
# T2 Layout equivalence

def test_t2_layout_equivalence():
    """Layout "6" expressed through the new fields (n_prelude=6) == stock 6-layer model."""
    new = build(GPT, GPTConfig(n_layer=6, n_prelude=6, n_core=0, n_coda=0, n_loop=1, **TINY))
    old = build(gpt_stock.GPT, gpt_stock.GPTConfig(n_layer=6, **TINY))
    assert new.visit_schedule() == list(range(6))
    assert list(new.state_dict().keys()) == list(old.state_dict().keys())
    randomize_(new, seed=3)
    old.load_state_dict(new.state_dict(), strict=True)
    idx = tokens(seed=4)
    torch.testing.assert_close(new(idx), old(idx), atol=1e-5, rtol=0)


def test_t2_parse_layout():
    assert gpt_module.parse_layout("12") == dict(n_layer=12, n_prelude=12, n_core=0, n_coda=0, n_loop=1)
    assert gpt_module.parse_layout("2,4x2,2") == dict(n_layer=8, n_prelude=2, n_core=4, n_coda=2, n_loop=2)
    assert gpt_module.parse_layout("0,6x2,0") == dict(n_layer=6, n_prelude=0, n_core=6, n_coda=0, n_loop=2)
    for bad in ("", "2,4,2", "2,4x0,2", "x", "2,0x2,2", "-1"):
        with pytest.raises(ValueError):
            gpt_module.parse_layout(bad)


def test_t2_visit_schedule():
    model = build(GPT, GPTConfig(**gpt_module.parse_layout("2,4x2,2"), **TINY))
    assert model.visit_schedule() == [0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7]
    assert model.visit_schedule(num_loops=1) == list(range(8))
    model.set_num_loops(3)
    assert len(model.visit_schedule()) == 2 + 4 * 3 + 2
    with pytest.raises(AssertionError):
        GPTConfig(n_layer=8, n_prelude=2, n_core=4, n_coda=1, **TINY)  # 2+4+1 != 8


# -----------------------------------------------------------------------------
# T3 Untied equivalence

def test_t3_untied_equivalence():
    """
    Looped 1,2x2,1 == stock 6-layer model whose layers 1-2 and 3-4 hold the same weights.
    This is the proof that looping is exactly weight sharing and nothing else.
    """
    looped = build(GPT, GPTConfig(**gpt_module.parse_layout("1,2x2,1"), **TINY))
    randomize_(looped, seed=5)
    stock, schedule = untie(looped)
    assert schedule == [0, 1, 2, 1, 2, 3]
    idx = tokens(seed=6)
    torch.testing.assert_close(looped(idx), stock(idx), atol=1e-5, rtol=0)
    # guard against a vacuous pass: the second visit of the core must change the output
    looped.set_num_loops(1)
    assert not torch.allclose(looped(idx), stock(idx), atol=1e-3)
