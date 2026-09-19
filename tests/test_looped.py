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
    for bad in (0, 2.0, True, torch.tensor(2)): # must stay a plain Python int, torch.compile guards on it
        with pytest.raises(AssertionError):
            model.set_num_loops(bad)


def test_t1_stock_sliding_windows_unchanged():
    """The stock path keeps its sliding window tiling (layouts are full context only and reject it)."""
    sliding = dict(TINY, sequence_len=2048, window_pattern="SSSL")
    new = build(GPT, GPTConfig(n_layer=6, **sliding))
    old = build(gpt_stock.GPT, gpt_stock.GPTConfig(n_layer=6, **sliding))
    assert new.window_sizes == old.window_sizes and len(set(new.window_sizes)) == 2
    assert new.estimate_flops() == old.estimate_flops()
    randomize_(new, seed=15)
    old.load_state_dict(new.state_dict(), strict=True)
    g = torch.Generator().manual_seed(16)
    idx = torch.randint(0, TINY["vocab_size"], (1, 1024), generator=g) # longer than the short window (768)
    torch.testing.assert_close(new(idx), old(idx), atol=1e-5, rtol=0)
    with pytest.raises(AssertionError):
        GPT(GPTConfig(**gpt_module.parse_layout("1,2x2,1"), **sliding))


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


# -----------------------------------------------------------------------------
# T4 KV cache consistency

class StubTokenizer:
    """Engine only needs the special token ids. They sit above the tiny vocab, so never get sampled."""
    def encode_special(self, s):
        return 1000 + len(s)

    def get_bos_token_id(self):
        return 999


@pytest.mark.parametrize("num_loops", [1, 2, 3])
def test_t4_kv_cache_consistency(num_loops):
    """
    Prefill-then-decode logits match full-sequence forward logits at every position.
    Engine.generate yields tokens and not logits, so the logits are compared by driving the model
    with a KV cache sized exactly the way Engine sizes it; Engine itself is then checked by
    comparing its greedy tokens with the cache-free GPT.generate.
    """
    model = randomize_(build(GPT, GPTConfig(**gpt_module.parse_layout("1,2x2,1"), **TINY)), seed=7)
    model.set_num_loops(num_loops)
    engine = engine_module.Engine(model, StubTokenizer())
    E = 1 + 2 * num_loops + 1
    assert engine.num_cache_layers() == E

    idx = tokens(seed=8, B=1, T=12)
    full = model(idx)
    kv_cache = engine_module.KVCache(batch_size=1, num_heads=TINY["n_kv_head"], seq_len=12,
        head_dim=TINY["n_embd"] // TINY["n_head"], num_layers=E, device="cpu", dtype=torch.float32)
    n_prefill = 5
    chunks = [model(idx[:, :n_prefill], kv_cache=kv_cache)]
    for t in range(n_prefill, 12):
        chunks.append(model(idx[:, t:t+1], kv_cache=kv_cache))
    assert kv_cache.get_pos() == 12 # advanced exactly once per forward, after the last effective layer
    torch.testing.assert_close(torch.cat(chunks, dim=1), full, atol=1e-4, rtol=0)

    prompt = idx[0, :6].tolist()
    reference = list(model.generate(prompt, max_tokens=8, temperature=0.0))
    results, _ = engine.generate_batch(prompt, num_samples=1, max_tokens=8, temperature=0.0)
    assert results[0][len(prompt):] == reference


def test_t4_wrong_cache_size_is_rejected():
    """A cache sized by unique layers (the stock sizing) must not be silently accepted."""
    model = build(GPT, GPTConfig(**gpt_module.parse_layout("1,2x2,1"), **TINY))
    kv_cache = engine_module.KVCache(batch_size=1, num_heads=TINY["n_kv_head"], seq_len=8,
        head_dim=TINY["n_embd"] // TINY["n_head"], num_layers=model.config.n_layer, device="cpu", dtype=torch.float32)
    with pytest.raises(AssertionError):
        model(tokens(seed=9, B=1, T=8), kv_cache=kv_cache)


# -----------------------------------------------------------------------------
# T5 Causality

@pytest.mark.parametrize("layout", ["4", "1,2x2,1", "0,2x3,0"])
def test_t5_causality(layout):
    """Changing token t+1 leaves the logits at positions <= t unchanged."""
    model = randomize_(build(GPT, GPTConfig(**gpt_module.parse_layout(layout), **TINY)), seed=10)
    assert model.smear_lambda.item() != 0 # otherwise smear (which looks at t-1) is tested vacuously
    idx = tokens(seed=11)
    t = 7
    changed = idx.clone()
    changed[:, t + 1] = (changed[:, t + 1] + 1) % TINY["vocab_size"]
    a, b = model(idx), model(changed)
    torch.testing.assert_close(a[:, :t + 1], b[:, :t + 1], atol=0, rtol=0)
    assert not torch.equal(a[:, t + 1:], b[:, t + 1:])


# -----------------------------------------------------------------------------
# T6 Accounting

D12 = dict(sequence_len=2048, vocab_size=32768, n_head=6, n_kv_head=6, n_embd=768, window_pattern="L")

def build_meta(layout):
    with torch.device("meta"):
        return GPT(GPTConfig(**gpt_module.parse_layout(layout), **D12))


def test_t6_accounting():
    d, V, t = 768, 32768, 2048
    block = 12 * d * d # attn 4 d^2 + mlp 8 d^2
    ve_gate = 12 * 6
    L8, B12 = build_meta("2,4x2,2"), build_meta("12")

    # Unique (stored) parameters of 2,4x2,2 by hand. Value embeddings on unique layers 1,3,5,7.
    counts = L8.num_scaling_params()
    assert counts["transformer_matrices"] == 8 * block + 4 * ve_gate
    assert counts["value_embeds"] == 4 * V * d
    assert counts["wte"] == V * d and counts["lm_head"] == V * d
    assert counts["scalars"] == 8 + 8 + 24 + 1 + 1
    assert counts["total"] == sum(p.numel() for p in L8.parameters())
    assert counts["unique_layers"] == 8 and counts["effective_layers"] == 12
    assert B12.num_scaling_params()["unique_layers"] == B12.num_scaling_params()["effective_layers"] == 12

    # Compute: blocks are counted once per visit. The 12 visits of 2,4x2,2 include 6 value
    # embedding visits (u = 1,3,5,3,5,7), the same number as the 12-layer model has.
    assert L8.num_matmul_params() == 12 * block + 6 * ve_gate + V * d + 24
    assert L8.num_matmul_params() == B12.num_matmul_params()
    assert abs(L8.estimate_flops() / B12.estimate_flops() - 1) < 0.005
    assert L8.estimate_flops() == 6 * L8.num_matmul_params() + 12 * (12 * 6 * 128 * t)

    # The layout-free stock model reports exactly what the frozen stock code reports.
    with torch.device("meta"):
        new = GPT(GPTConfig(n_layer=12, **D12))
        old = gpt_stock.GPT(gpt_stock.GPTConfig(n_layer=12, **D12))
    assert new.estimate_flops() == old.estimate_flops()
    assert new.kv_bytes_per_token() == old.kv_bytes_per_token()
    assert new.kv_read_bytes(1000) == old.kv_read_bytes(1000)
    assert new.estimate_decode_flops(1000) == old.estimate_decode_flops(1000)
    assert new.estimate_prefill_flops(1000) == old.estimate_prefill_flops(1000)


def test_t6_accounting_follows_loop_count():
    L8 = build_meta("2,4x2,2")
    d, t = 768, 2048
    per_loop = 6 * (4 * 12 * d * d + 2 * 12 * 6) + 4 * (12 * 6 * 128 * t) # 4 core blocks, 2 of them with ve_gate
    assert L8.estimate_flops(num_loops=3) - L8.estimate_flops(num_loops=2) == per_loop
    assert L8.estimate_flops(num_loops=2) == L8.estimate_flops()
    kv2, read2, dec2, pre2 = L8.kv_bytes_per_token(), L8.kv_read_bytes(500), L8.estimate_decode_flops(500), L8.estimate_prefill_flops(500)
    L8.set_num_loops(4) # E: 12 -> 20
    assert L8.kv_bytes_per_token() * 12 == kv2 * 20
    assert L8.kv_read_bytes(500) * 12 == read2 * 20
    assert L8.estimate_decode_flops(500) > dec2 and L8.estimate_prefill_flops(500) > pre2
    assert L8.num_scaling_params()["effective_layers"] == 20


# -----------------------------------------------------------------------------
# T7 Gradient sharing

def test_t7_gradient_sharing():
    """
    After one backward pass, the gradient of every shared parameter equals the sum of the
    gradients of its copies in the untied model. Summation order differs, hence the rtol.
    """
    looped = randomize_(build(GPT, GPTConfig(**gpt_module.parse_layout("1,2x2,1"), **TINY)), seed=12)
    stock, schedule = untie(looped)
    idx, targets = tokens(seed=13), tokens(seed=14)
    looped(idx, targets=targets).backward()
    stock(idx, targets=targets).backward()
    stock_grads = {name: p.grad for name, p in stock.named_parameters()}

    checked_shared = 0
    for name, p in looped.named_parameters():
        parts = name.split(".")
        if name in ("resid_lambdas", "x0_lambdas"):
            expected = torch.zeros_like(p)
            for e, u in enumerate(schedule):
                expected[u] += stock_grads[name][e]
        elif parts[:2] == ["transformer", "h"] or parts[0] == "value_embeds":
            i = 2 if parts[0] == "transformer" else 1
            visits = [e for e, u in enumerate(schedule) if u == int(parts[i])]
            keys = [".".join(parts[:i] + [str(e)] + parts[i + 1:]) for e in visits]
            assert visits and all(k in stock_grads for k in keys), f"{name} has no untied counterpart at every visit"
            expected = sum(stock_grads[k] for k in keys)
            checked_shared += len(keys) > 1
        else:
            expected = stock_grads[name]
        assert p.grad is not None and p.grad.abs().sum() > 0, f"no gradient reached {name}"
        torch.testing.assert_close(p.grad, expected, rtol=1e-4, atol=1e-6, msg=lambda m: f"{name}: {m}")
    assert checked_shared > 0


# -----------------------------------------------------------------------------
# Old checkpoints

def test_old_checkpoint_config_gets_stock_layout_defaults():
    from nanochat.checkpoint_manager import _patch_missing_config_keys
    old = dict(sequence_len=32, vocab_size=256, n_layer=4, n_head=4, n_kv_head=4, n_embd=64, window_pattern="L")
    _patch_missing_config_keys(old)
    config = GPTConfig(**old)
    assert (config.n_prelude, config.n_core, config.n_coda, config.n_loop) == (-1, 0, 0, 1)
    looped = dict(old, **gpt_module.parse_layout("1,2x2,1"))
    _patch_missing_config_keys(looped) # must not clobber a looped checkpoint's layout
    assert GPTConfig(**looped).n_core == 2


# -----------------------------------------------------------------------------
# T8 / T9: the training script, as a subprocess on CPU

import csv
import json
import os
import re
import subprocess
import sys

REPO = Path(__file__).parent.parent
REAL_BASE_DIR = Path(os.environ.get("NANOCHAT_BASE_DIR", Path.home() / ".cache" / "nanochat"))
needs_data = pytest.mark.skipif(
    not (REAL_BASE_DIR / "tokenizer" / "tokenizer.pkl").exists() or not list((REAL_BASE_DIR / "base_data_climbmix").glob("*.parquet")),
    reason="needs the trained tokenizer and the pretraining data shards in the nanochat base dir")


def run_base_train(tmp_path, tag, *args):
    """
    Run a 20 step scripts.base_train on CPU. Checkpoints go to a scratch base dir that only links
    the real tokenizer and data, and results to a scratch results dir: a test must never touch
    the results file of the study.
    No GPU => fp32 and SDPA. Dynamo is disabled because CPU inductor needs setuptools, which the
    gpu environment does not install; compiled-graph behavior is checked in the GPU smoke runs.
    """
    base_dir, results_dir = tmp_path / "base", tmp_path / f"results_{tag}"
    base_dir.mkdir(exist_ok=True)
    for name in ("tokenizer", "base_data_climbmix"):
        if not (base_dir / name).exists():
            (base_dir / name).symlink_to(REAL_BASE_DIR / name)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TORCHDYNAMO_DISABLE="1", NANOCHAT_BASE_DIR=str(base_dir))
    cmd = [sys.executable, "-m", "scripts.base_train", "--window-pattern", "L", "--max-seq-len", "256",
        "--total-batch-size", "1024", "--num-iterations", "20", "--eval-every", "10", "--eval-tokens", "4096",
        "--core-metric-every", "-1", "--sample-every", "-1", "--no-save-optimizer",
        "--model-tag", f"looped_test_{tag}", "--arm", tag, "--results-dir", str(results_dir), *args]
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    with open(results_dir / "looped_results.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    logs = list((results_dir / "logs").glob("*.jsonl"))
    assert len(logs) == 1
    events = [json.loads(line) for line in logs[0].read_text().splitlines()]
    losses = [float(x) for x in re.findall(r"^step \d+/\d+ .*? loss: ([\d.]+)", proc.stdout, flags=re.M)]
    return rows[0], events, losses


@needs_data
@pytest.mark.slow
def test_t8_data_order_is_independent_of_seed(tmp_path):
    """--seed changes the parameter init and nothing else: not the data, not the loop counts."""
    layout = ["--layout", "1,2x2,1", "--width-depth", "4", "--ref-layout", "4", "--train-loops", "1,2,3"]
    a, events_a, _ = run_base_train(tmp_path, "seed0", *layout, "--seed", "0", "--device-batch-size", "2")
    b, events_b, _ = run_base_train(tmp_path, "seed1", *layout, "--seed", "1", "--device-batch-size", "2")
    # the hash counts rows, not micro-batches, so it may not depend on the device batch size either
    c, _, _ = run_base_train(tmp_path, "seed1_b1", *layout, "--seed", "1", "--device-batch-size", "1")
    assert a["data_sha256"] == b["data_sha256"] == c["data_sha256"]
    assert a["data_hash_rows"] == b["data_hash_rows"] == c["data_hash_rows"] != "0"
    config = lambda events: next(e for e in events if e["kind"] == "config")
    assert config(events_a)["loop_schedule_sha256"] == config(events_b)["loop_schedule_sha256"]
    assert a["train_flops"] == b["train_flops"]
    assert a["final_val_bpb"] != b["final_val_bpb"] # different init
    # same seed, same data, only the micro-batch size differs: the same run up to float summation order
    assert float(b["final_val_bpb"]) == pytest.approx(float(c["final_val_bpb"]), abs=1e-3)


@needs_data
@pytest.mark.slow
@pytest.mark.parametrize("tag,args", [
    ("plain", ["--layout", "4", "--width-depth", "4", "--ref-layout", "6"]),
    ("looped", ["--layout", "1,2x2,1", "--width-depth", "4", "--ref-layout", "6"]),
    ("random_r", ["--layout", "1,2x2,1", "--width-depth", "4", "--ref-layout", "6", "--train-loops", "1,2,3"]),
])
def test_t9_smoke(tmp_path, tag, args):
    row, events, losses = run_base_train(tmp_path, tag, *args, "--seed", "0", "--device-batch-size", "2")
    # 20 steps are all inside the LR warmup and batches are 1024 tokens, so the loss falls slowly
    # and noisily: compare the first and last five steps rather than two single steps
    assert len(losses) == 20 and sum(losses[-5:]) / 5 < sum(losses[:5]) / 5 - 0.01, losses
    assert row["diverged"] == "0" and row["steps"] == "20" and row["tokens"] == str(20 * 1024)
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "config" and kinds[-1] == "final" and kinds.count("eval") == 3 and kinds.count("diag") == 3
    config, final = events[0], events[-1]
    # Everything derived from the model size comes from the reference layout (6 layers), not from
    # the arm (4 unique layers): the horizon is the reference model's, computed here by hand
    with torch.device("meta"):
        ref = GPT(GPTConfig(sequence_len=256, vocab_size=config["model_config"]["vocab_size"], n_layer=6, n_head=2, n_kv_head=2, n_embd=256, window_pattern="L"))
    ref_counts, own_counts = ref.num_scaling_params(), config["param_counts"]
    assert config["target_tokens"] == 12 * (ref_counts["transformer_matrices"] + ref_counts["lm_head"])
    assert config["target_tokens"] != 12 * (own_counts["transformer_matrices"] + own_counts["lm_head"])
    E = config["param_counts"]["effective_layers"]
    assert E == (4 if tag == "plain" else 6)
    assert len(final["resid_lambdas"]) == len(final["x0_lambdas"]) == 4 # per unique layer
    diag = next(e for e in events if e["kind"] == "diag")["residual_rms"]
    if tag == "random_r":
        assert {r: len(v) for r, v in diag.items()} == {"1": 4, "2": 6, "3": 8} # one entry per effective layer, for every trained R
        per_token = config["flops_per_token_at_loops"]
        assert per_token["1"] < per_token["2"] < per_token["3"]
        assert per_token["1"] * 20 * 1024 < int(row["train_flops"]) < per_token["3"] * 20 * 1024
    else:
        assert list(diag) == [str(config["model_config"]["n_loop"])] and len(diag[list(diag)[0]]) == E
        assert int(row["train_flops"]) == config["num_flops_per_token"] * 20 * 1024
    train = next(e for e in events if e["kind"] == "train")
    assert train["grad_norm_noncore"] > 0 and (train["grad_norm_core"] > 0) == (tag != "plain")
