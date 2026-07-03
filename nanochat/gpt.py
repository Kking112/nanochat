"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.checkpoint

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW

# Our custom Flash Attention module that automatically uses FA3 when compatible and SDPA fallback otherwise
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"

    # --- LoopMoE additions (all default to stock dense nanochat behavior) -----------------
    # Mixture-of-Experts feed-forward (Change 1). n_routed_experts=0 => dense ReLU^2 MLP.
    n_routed_experts: int = 0       # number of routed experts (0 disables MoE => dense MLP)
    n_shared_experts: int = 0       # always-on shared experts added unconditionally
    top_k: int = 1                  # experts selected per token
    expert_hidden_dim: int = 0      # per-expert FFN hidden dim (0 => auto: match dense FLOPs)
    load_balance_mode: str = "bias_free"  # "bias_free" (DeepSeek-V3) or "aux_loss"
    aux_loss_coef: float = 0.01     # coefficient for the classic aux load-balancing loss
    router_bias_update_rate: float = 0.001  # bias nudge speed for bias_free balancing

    # Sandwich loop layout (Change 2). n_loop_layers=0 => standard n_layer stack (no loop).
    n_prefix_layers: int = 0        # Lp: non-shared prefix blocks
    n_loop_layers: int = 0          # Lb: shared loop-body blocks (0 disables looping)
    n_suffix_layers: int = 0        # Ls: non-shared suffix blocks
    loop_iters: int = 1             # K: times the shared loop body is applied

    # IterAdaLN inside the loop body (Change 3).
    use_iter_adaln: bool = False    # token+iteration conditioned modulation in loop body
    grad_checkpoint_loop: bool = False  # gradient-checkpoint loop-body executions

    @property
    def use_moe(self):
        return self.n_routed_experts > 0

    @property
    def use_loop(self):
        return self.n_loop_layers > 0


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    # note: this rotates by -theta, the transpose of the textbook convention. Functionally
    # equivalent (only the relative q/k rotation matters), kept for checkpoint compatibility.
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache, kv_layer_idx=None):
        B, T, C = x.size()
        # kv_layer_idx is the cache slot for THIS execution. For the standard stack it equals
        # self.layer_idx, but a shared loop-body block runs at several schedule positions, so
        # the caller passes the execution index explicitly. Defaults to self.layer_idx.
        li = self.layer_idx if kv_layer_idx is None else kv_layer_idx

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Flash Attention (FA3 or SDPA fallback)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(li)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after the last cache slot (last execution) processes
            if li == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x, None  # (output, aux_loss); dense MLP never produces an aux loss


class Expert(nn.Module):
    """A single ReLU^2 FFN, identical in spirit to MLP but with a configurable hidden dim.
    Kept as its own class so the router/MoE code reads cleanly and so init is explicit."""
    def __init__(self, n_embd, hidden_dim):
        super().__init__()
        self.c_fc = Linear(n_embd, hidden_dim, bias=False)
        self.c_proj = Linear(hidden_dim, n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


def moe_expert_hidden_dim(config):
    """Resolve the per-expert hidden dim. If expert_hidden_dim==0, auto-size so that the
    per-token FFN compute (top_k routed + shared experts) roughly matches the dense 4*n_embd
    MLP it replaces: dense hidden = 4*n_embd spread over (top_k + n_shared) active experts."""
    if config.expert_hidden_dim > 0:
        return config.expert_hidden_dim
    active = max(1, config.top_k + config.n_shared_experts)
    h = (4 * config.n_embd) // active
    return max(1, (h // 128) * 128)  # round down to a tensor-core-friendly multiple of 128


class MoE(nn.Module):
    """DeepSeekMoE-style Mixture-of-Experts feed-forward layer (LoopMoE Change 1).

    - n_routed_experts independent ReLU^2 FFNs; a token is routed to its top_k experts.
    - Router: single Linear(n_embd, n_routed_experts) -> softmax affinities; top_k selected,
      outputs combined by the (renormalized) softmax weights of the chosen experts.
    - n_shared_experts always-on FFNs added unconditionally.
    - Load balancing: default is DeepSeek-V3's auxiliary-loss-free per-expert bias that nudges
      the *selection* logits toward balanced load (the combine weights still use the raw
      softmax affinity). An optional classic aux load-balancing loss is available via config.

    Correctness-first implementation: a plain Python for-loop over experts. Every expert is
    always invoked (on a possibly-empty token slice) so that every expert weight receives a
    gradient each step -- required because nanochat's optimizer stacks per-param grads and a
    None grad would break it. This routing is data-dependent, so forward is excluded from
    torch.compile.
    TODO: replace the per-expert loop with a batched/grouped-GEMM expert path for speed.
    """
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.top_k = config.top_k
        self.load_balance_mode = config.load_balance_mode
        self.aux_loss_coef = config.aux_loss_coef
        self.bias_update_rate = config.router_bias_update_rate
        assert self.top_k <= self.n_routed_experts, "top_k cannot exceed n_routed_experts"
        assert self.load_balance_mode in ("bias_free", "aux_loss")
        hidden = moe_expert_hidden_dim(config)
        self.expert_hidden_dim = hidden
        self.router = Linear(config.n_embd, config.n_routed_experts, bias=False)
        self.experts = nn.ModuleList([Expert(config.n_embd, hidden) for _ in range(config.n_routed_experts)])
        self.shared_experts = nn.ModuleList([Expert(config.n_embd, hidden) for _ in range(config.n_shared_experts)])
        # Per-expert selection bias for the auxiliary-loss-free balancing strategy.
        # Not a leaf we optimize with SGD: it is nudged by step_bias() AFTER each optimizer step
        # (deferred out of forward so it is safe under gradient checkpointing, which recomputes
        # the loop-body forward). Persistent so it survives checkpoint save/resume.
        self.register_buffer("expert_bias", torch.zeros(config.n_routed_experts), persistent=True)
        # Instrumentation (detached, for logging only; filled each training forward).
        self.register_buffer("last_load", torch.zeros(config.n_routed_experts), persistent=False)
        self.last_entropy = 0.0
        # Latest global token counts stashed by forward, consumed by step_bias(). Plain attribute
        # (not a buffer/param) so it never enters state_dict, meta init, or the autograd graph.
        self.last_global_counts = None

    @torch.compiler.disable
    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)  # (N, C) with N = B*T tokens

        # Router affinities. softmax over experts, per the task spec.
        logits = self.router(x_flat).float()          # (N, E) in fp32 for stable routing math
        probs = F.softmax(logits, dim=-1)              # (N, E) combine affinities

        # Top-k selection. bias_free adds a per-expert bias to the *selection* score only.
        if self.load_balance_mode == "bias_free":
            select_score = probs + self.expert_bias.float()
        else:
            select_score = probs
        topk_val, topk_idx = torch.topk(select_score, self.top_k, dim=-1)  # (N, k)
        # Combine weights come from the raw affinity (not the biased score), renormalized over k.
        weights = probs.gather(1, topk_idx)                                # (N, k)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        weights = weights.to(x.dtype)

        # Dispatch. Loop over experts; each processes the tokens routed to it. We always call
        # every expert (empty slice -> zero grad, not None) to keep all expert grads defined.
        y = torch.zeros_like(x_flat)
        # one-hot-ish assignment mask per expert: (N,) bool for whether expert e is in a token's top-k
        # and the corresponding combine weight for that token.
        counts = torch.zeros(self.n_routed_experts, device=x.device, dtype=torch.float32)
        for e, expert in enumerate(self.experts):
            sel = (topk_idx == e)                       # (N, k) where this expert was chosen
            tok_mask = sel.any(dim=-1)                  # (N,)
            idx = tok_mask.nonzero(as_tuple=True)[0]    # token indices routed to expert e
            counts[e] = idx.numel()
            xe = x_flat.index_select(0, idx)            # (n_e, C); may be empty
            oe = expert(xe)                             # (n_e, C); empty-safe, grad stays defined
            # weight for expert e at each of its tokens: pick the matching top-k slot
            we = (weights * sel.to(weights.dtype)).sum(dim=-1)  # (N,)
            we = we.index_select(0, idx).unsqueeze(-1)          # (n_e, 1)
            y.index_add_(0, idx, oe * we)

        # Shared experts: always-on, added unconditionally to every token.
        for expert in self.shared_experts:
            y = y + expert(x_flat)

        # --- Load balancing bookkeeping ---------------------------------------------------
        # aux must ALWAYS be computed when training (independent of grad state) so the module's
        # output structure is stable under gradient checkpointing, which runs forward twice.
        aux = None
        if self.training and self.load_balance_mode == "aux_loss":
            # Classic Switch-Transformer aux loss: E * sum_i f_i * P_i, encourages the fraction
            # of tokens f_i and mean routing prob P_i to both be ~1/E (balanced).
            f_i = counts / counts.sum().clamp_min(1)     # (E,) local token fractions
            P_i = probs.mean(dim=0)                       # (E,)
            aux = self.n_routed_experts * (f_i * P_i).sum()

        # Stat/bookkeeping side effects must happen exactly ONCE per optimizer step. Under
        # gradient checkpointing the first forward runs with grad disabled and is recomputed
        # with grad enabled in backward; gating on torch.is_grad_enabled() makes these run once
        # (in the recompute), avoiding an extra all-reduce on the discarded first pass. Without
        # checkpointing, training forwards have grad enabled, so this runs normally. Crucially we
        # only READ here and stash counts -- expert_bias is NOT mutated in forward (see step_bias),
        # because the same shared loop block is checkpointed multiple times per step and mutating
        # it here would make an earlier execution's recompute diverge from its saved activations.
        if self.training and torch.is_grad_enabled():
            global_counts = counts
            if dist.is_available() and dist.is_initialized():
                global_counts = counts.clone()
                dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
            self.last_global_counts = global_counts.detach()
            self.last_load = (global_counts / global_counts.sum().clamp_min(1)).detach()
            mean_p = probs.mean(dim=0)  # routing entropy: cheap router-collapse detector
            self.last_entropy = float(-(mean_p * (mean_p + 1e-9).log()).sum().detach())

        return y.view(B, T, C), aux

    @torch.no_grad()
    def step_bias(self):
        """DeepSeek-V3 aux-loss-free update: nudge each expert's selection bias by the sign of
        its load error (underloaded experts get a positive nudge, overloaded a negative one).
        Called once per optimizer step (after backward), not inside forward -- see the note in
        forward(). No-op unless bias_free balancing is active and a forward has run."""
        if self.load_balance_mode != "bias_free" or self.last_global_counts is None:
            return
        counts = self.last_global_counts
        load = counts / counts.sum().clamp_min(1)
        target = 1.0 / self.n_routed_experts
        err = target - load                       # >0 => underloaded => raise its bias
        self.expert_bias += self.bias_update_rate * torch.sign(err)


def sinusoidal_iter_embedding(k, dim, device, dtype):
    """DiT-style fixed sinusoidal positional encoding of the (scalar) iteration index k.
    Returns a (dim,) tensor. k is a Python int (constant per unrolled loop iteration)."""
    half = dim // 2
    # frequencies spaced geometrically, same convention as diffusion timestep embeddings
    freqs = torch.exp(-torch.arange(half, device=device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half))
    args = k * freqs
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:  # pad if odd
        emb = torch.cat([emb, torch.zeros(1, device=device, dtype=torch.float32)], dim=-1)
    return emb.to(dtype)


class IterAdaLN(nn.Module):
    """Iteration- and token-conditioned modulation for the loop body (LoopMoE Change 3).

    Replaces a pre-norm RMSNorm with:  (1 + gamma(c)) * RMSNorm_affine_free(h) + beta(c),
    where the condition fuses a token stream and an iteration stream (broadcast add):
        c = token_proj(hidden_state) + iter_mlp(sinusoidal_PE(k))
    gamma/beta (and, for the attention branch only, a residual gate alpha) come from
    zero-initialized output heads (AdaLN-Zero), so the loop starts as near-identity and
    differentiates during training.

    `gate=True` builds the attention-branch alpha gate; `gate=False` (MoE branch) omits it
    (per the paper: a symmetric gate on the MoE branch is absorbed into routing weights and
    interferes with load balancing).
    """
    PE_DIM = 128  # dimensionality of the sinusoidal iteration embedding

    def __init__(self, config, gate):
        super().__init__()
        n_embd = config.n_embd
        cond = n_embd
        self.gate = gate
        # NOTE: nanochat's custom Linear is bias-free by design (it overrides forward and never
        # applies a bias), so every layer here is bias=False. AdaLN-Zero still works: zero-init
        # the head *weights* and the modulation output is zero at init (identity), differentiating
        # as the weights learn.
        # token stream: project the current per-token hidden state into the condition space
        self.adaln_token_proj = Linear(n_embd, cond, bias=False)
        # iteration stream: small MLP over the sinusoidal PE of the iteration index
        self.adaln_iter_mlp = nn.Sequential(
            Linear(self.PE_DIM, cond, bias=False),
            nn.SiLU(),
            Linear(cond, cond, bias=False),
        )
        # AdaLN-Zero output heads (zero-initialized): shift (beta), scale (gamma), gate (alpha)
        self.adaln_gamma = Linear(cond, n_embd, bias=False)
        self.adaln_beta = Linear(cond, n_embd, bias=False)
        self.adaln_alpha = Linear(cond, n_embd, bias=False) if gate else None

    def forward(self, h, pe):
        # pe: precomputed sinusoidal PE of the iteration index, shape (PE_DIM,). Passed in (not
        # computed here from a python int) so torch.compile does not specialize/recompile per k.
        # condition = token stream (per token) + iteration stream (broadcast over tokens)
        c = self.adaln_token_proj(h) + self.adaln_iter_mlp(pe.to(h.dtype))      # (B,T,cond)
        c = F.silu(c)
        gamma = self.adaln_gamma(c)
        beta = self.adaln_beta(c)
        out = (1 + gamma) * norm(h) + beta  # affine-free RMSNorm, dynamically modulated
        if self.gate:
            alpha = self.adaln_alpha(c)
            return out, alpha
        return out


class Block(nn.Module):
    def __init__(self, config, layer_idx, is_loop_body=False):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        # Feed-forward is either the dense ReLU^2 MLP or the MoE layer. Both return (out, aux).
        self.mlp = MoE(config) if config.use_moe else MLP(config)
        # IterAdaLN lives in the loop body only; prefix/suffix keep standard normalization.
        self.is_loop_body = is_loop_body
        self.use_adaln = is_loop_body and config.use_iter_adaln
        if self.use_adaln:
            self.adaln_attn = IterAdaLN(config, gate=True)   # attention branch: gated
            self.adaln_moe = IterAdaLN(config, gate=False)   # MoE/FFN branch: no gate

    def forward(self, x, ve, cos_sin, window_size, kv_cache, kv_layer_idx=None, iter_pe=None):
        # iter_pe is the precomputed sinusoidal PE of the loop iteration index (None for
        # non-loop / prefix / suffix). Consumed by IterAdaLN in the loop body (Change 3);
        # ignored by prefix/suffix/standard blocks.
        if self.use_adaln:
            # Attention branch (gated): m = h + alpha * Attn(IterAdaLN(h; c_attn)).
            # Token stream for c_attn uses the block INPUT state h.
            h_normed, alpha = self.adaln_attn(x, iter_pe)
            m = x + alpha * self.attn(h_normed, ve, cos_sin, window_size, kv_cache, kv_layer_idx)
            # MoE branch (ungated): h_out = m + MoE(IterAdaLN(m; c_moe)).
            # Token stream for c_moe uses the post-attention state m.
            ff_out, aux = self.mlp(self.adaln_moe(m, iter_pe))
            return m + ff_out, aux
        # Standard pre-norm block (prefix, suffix, or non-looped stack)
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache, kv_layer_idx)
        ff_out, aux = self.mlp(norm(x))
        x = x + ff_out
        return x, aux


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        # Which unique blocks form the shared loop body (they get IterAdaLN when enabled).
        _lb = None
        if config.use_loop:
            _lb = (config.n_prefix_layers, config.n_prefix_layers + config.n_loop_layers)
        def _is_loop_body(i):
            return _lb is not None and _lb[0] <= i < _lb[1]
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx, is_loop_body=_is_loop_body(layer_idx))
                                for layer_idx in range(config.n_layer)]),
        })
        # Execution schedule: list of (unique_block_idx, iter_k). For the stock (non-looped)
        # model this is simply each unique block once. For the sandwich loop it is
        # prefix once + loop-body K times (shared weights) + suffix once (LoopMoE Change 2).
        # Per-block machinery (window sizes, resid/x0 scalars, value embeddings) is indexed by
        # the UNIQUE block index, so the number of unique parameters is independent of K.
        self.exec_schedule = self._build_exec_schedule(config)
        self.n_exec = len(self.exec_schedule)  # effective depth = number of block executions
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

        # Precomputed sinusoidal iteration positional encodings for IterAdaLN, one row per loop
        # iteration. Precomputing (rather than building from a python int inside forward) keeps
        # torch.compile from specializing/recompiling per iteration index. Filled in init_weights.
        if config.use_iter_adaln and config.use_loop:
            iter_pe = torch.zeros(config.loop_iters, IterAdaLN.PE_DIM)
            self.register_buffer("iter_pe", iter_pe, persistent=False)
        else:
            self.iter_pe = None

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            if block.use_adaln:
                # The attention branch is scaled by a zero-initialized alpha gate (AdaLN-Zero),
                # so c_proj must NOT also be zero -- both zero would make the branch dead (no
                # gradient reaches either). Use a SMALL non-zero init (0.4x): large enough to keep
                # the branch alive, small enough that the first gradient into the zero alpha gate
                # (which is proportional to the attention output magnitude) does not blow up.
                torch.nn.init.uniform_(block.attn.c_proj.weight, -s * 0.4, s * 0.4)
            else:
                torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero (residual-zero init)
            if isinstance(block.mlp, MoE):
                self._init_moe(block.mlp, s)
            else:
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if block.use_adaln:
                self._init_adaln(block.adaln_attn)
                self._init_adaln(block.adaln_moe)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Smear/backout scalars and smear gate must be explicitly initialized 
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Fill the sinusoidal iteration PE table for IterAdaLN (adaln nets are inited in the
        # per-block loop above). One row per loop iteration, cast to the compute dtype.
        if self.iter_pe is not None:
            device = self.transformer.wte.weight.device
            rows = [sinusoidal_iter_embedding(k, IterAdaLN.PE_DIM, device, torch.float32)
                    for k in range(self.config.loop_iters)]
            self.iter_pe = torch.stack(rows, dim=0).to(COMPUTE_DTYPE)

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    @torch.no_grad()
    def _init_moe(self, moe, s):
        """Initialize an MoE block. Experts mirror the dense MLP init (c_fc uniform at 0.4x
        scale, c_proj zeros so each expert starts as a near-identity residual). The router gets
        a small normal init to lightly break symmetry while starting near-uniform routing.
        Buffers (expert_bias, last_load) are created on the meta device, so they hold garbage
        after to_empty() and must be explicitly zeroed here."""
        torch.nn.init.normal_(moe.router.weight, mean=0.0, std=0.02)
        for expert in list(moe.experts) + list(moe.shared_experts):
            torch.nn.init.uniform_(expert.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(expert.c_proj.weight)
        moe.expert_bias.zero_()
        moe.last_load.zero_()

    @torch.no_grad()
    def _init_adaln(self, adaln):
        """Initialize an IterAdaLN module (AdaLN-Zero). The output heads (gamma/beta/alpha) are
        ZERO so the modulation starts as the identity ((1+0)*norm(h)+0) and the alpha gate starts
        closed, i.e. the loop body begins as near-identity and differentiates during training.
        The conditioning nets (token_proj, iter_mlp) get small non-zero inits so a gradient
        signal exists once the zero heads start to move."""
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        torch.nn.init.uniform_(adaln.adaln_token_proj.weight, -s * 0.4, s * 0.4)
        for layer in adaln.adaln_iter_mlp:
            if isinstance(layer, nn.Linear):
                # iter_mlp maps PE_DIM -> cond; scale by its own input dim, not n_embd
                si = 3**0.5 * layer.weight.shape[1] ** -0.5
                torch.nn.init.uniform_(layer.weight, -si * 0.4, si * 0.4)
        # AdaLN-Zero: zero the output head weights (bias-free) so modulation starts at identity
        for head in (adaln.adaln_gamma, adaln.adaln_beta, adaln.adaln_alpha):
            if head is not None:
                torch.nn.init.zeros_(head.weight)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def _build_exec_schedule(self, config):
        """Build the block execution schedule as a list of (unique_block_idx, iter_k).

        Non-looped: [(0,0), (1,0), ..., (n_layer-1, 0)] -- reduces exactly to the stock stack.
        Looped (LoopMoE Change 2, no prefix re-injection):
            prefix blocks [0, Lp)            run once
            loop-body blocks [Lp, Lp+Lb)     run K times with the SAME shared weights
            suffix blocks [Lp+Lb, Lp+Lb+Ls)  run once
        Effective depth = Lp + K*Lb + Ls; unique layers = Lp + Lb + Ls == n_layer.
        """
        if not config.use_loop:
            return [(i, 0) for i in range(config.n_layer)]
        Lp, Lb, Ls, K = config.n_prefix_layers, config.n_loop_layers, config.n_suffix_layers, config.loop_iters
        n_unique = Lp + Lb + Ls
        assert n_unique == config.n_layer, (
            f"When looping, n_layer ({config.n_layer}) must equal "
            f"n_prefix_layers+n_loop_layers+n_suffix_layers ({Lp}+{Lb}+{Ls}={n_unique})"
        )
        assert K >= 1, "loop_iters (K) must be >= 1"
        schedule = [(i, 0) for i in range(Lp)]                          # prefix, once
        for k in range(K):                                             # loop body, K times
            schedule += [(Lp + j, k) for j in range(Lb)]
        schedule += [(Lp + Lb + i, 0) for i in range(Ls)]              # suffix, once
        return schedule

    def loop_body_range(self):
        """(start, end) unique-block indices of the shared loop body, or None if not looping."""
        if not self.config.use_loop:
            return None
        Lp, Lb = self.config.n_prefix_layers, self.config.n_loop_layers
        return (Lp, Lp + Lb)

    def update_router_bias(self):
        """Apply the aux-loss-free per-expert bias nudge for every MoE layer. Call once per
        optimizer step, after backward(). No-op when MoE / bias_free balancing is not in use."""
        for m in self.modules():
            if isinstance(m, MoE):
                m.step_bias()

    def moe_load_stats(self):
        """Return per-MoE routing instrumentation for logging: list of dicts with the max
        per-expert load fraction and the routing entropy. Empty when MoE is not in use."""
        stats = []
        for m in self.modules():
            if isinstance(m, MoE):
                stats.append({"max_load": float(m.last_load.max()), "entropy": m.last_entropy})
        return stats

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        # Sum the matmul params actually engaged per token, walking the execution schedule.
        # This correctly reflects (a) looped compute: a shared block counts once per execution
        # (K times), and (b) MoE: only the top_k routed + shared experts run, not the full pool.
        matmul_numel = self.lm_head.weight.numel()  # unembedding runs once per token
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        attn_flops = 0
        for block_idx, _k in self.exec_schedule:
            block = self.transformer.h[block_idx]
            matmul_numel += (self._attn_matmul_numel(block) + self._active_ffn_numel(block)
                             + self._adaln_matmul_numel(block))
            window = self.window_sizes[block_idx][0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * matmul_numel + attn_flops
        return num_flops_per_token

    def _attn_matmul_numel(self, block):
        """Number of attention matmul weights in a block (always active per token)."""
        a = block.attn
        n = a.c_q.weight.numel() + a.c_k.weight.numel() + a.c_v.weight.numel() + a.c_proj.weight.numel()
        if a.ve_gate is not None:
            n += a.ve_gate.weight.numel()
        return n

    def _adaln_matmul_numel(self, block):
        """Per-token IterAdaLN matmul weights (token projection + gamma/beta/alpha heads). The
        iteration MLP runs once per iteration, not per token, so it is omitted here."""
        if not getattr(block, "use_adaln", False):
            return 0
        n = 0
        for adaln in (block.adaln_attn, block.adaln_moe):
            n += adaln.adaln_token_proj.weight.numel()
            n += adaln.adaln_gamma.weight.numel() + adaln.adaln_beta.weight.numel()
            if adaln.adaln_alpha is not None:
                n += adaln.adaln_alpha.weight.numel()
        return n

    def _active_ffn_numel(self, block):
        """Feed-forward matmul weights active per token: dense MLP -> both matrices; MoE ->
        the router plus (top_k routed + shared) experts, i.e. active params, not the full pool."""
        mlp = block.mlp
        if isinstance(mlp, MoE):
            per_expert = self.config.n_embd * mlp.expert_hidden_dim * 2  # c_fc + c_proj
            active_experts = mlp.top_k + mlp.n_shared_experts
            return mlp.router.weight.numel() + active_experts * per_expert
        return mlp.c_fc.weight.numel() + mlp.c_proj.weight.numel()

    def param_report(self):
        """Parameter-count breakdown for logging, distinguishing total vs active (the point of
        MoE + looping). Returns:
        - total_params: every unique weight in the model.
        - active_params_physical: unique weights engaged per token, counting a shared loop block
          only ONCE even if it runs K times (attention/FFN 'physical' footprint).
        - active_params_flops: weights engaged per token counting each execution (K x for loop
          body) -- this is what drives FLOPs/MFU.
        - effective_depth (n_exec), unique_layers (n_layer).
        """
        total_params = sum(p.numel() for p in self.parameters())
        embed_active = self.config.n_embd  # one embedding row per token (lookup)
        # flops-active: walk the schedule (K executions counted)
        active_flops = self.lm_head.weight.numel() + embed_active
        for block_idx, _k in self.exec_schedule:
            block = self.transformer.h[block_idx]
            active_flops += (self._attn_matmul_numel(block) + self._active_ffn_numel(block)
                             + self._adaln_matmul_numel(block))
        # physical-active: each unique executed block counted once
        unique_used = sorted({b for b, _k in self.exec_schedule})
        active_physical = self.lm_head.weight.numel() + embed_active
        for block_idx in unique_used:
            block = self.transformer.h[block_idx]
            active_physical += (self._attn_matmul_numel(block) + self._active_ffn_numel(block)
                                + self._adaln_matmul_numel(block))
        return {
            'total_params': total_params,
            'active_params_physical': active_physical,
            'active_params_flops': active_flops,
            'effective_depth': self.n_exec,
            'unique_layers': self.config.n_layer,
        }

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def _moe_router_param_ids(self):
        """ids of router weights: these are classifier-like and go to AdamW, not Muon."""
        ids = set()
        for m in self.modules():
            if isinstance(m, MoE):
                ids.add(id(m.router.weight))
        return ids

    def _adaln_param_ids(self):
        """ids of IterAdaLN / iteration-MLP params: small conditioning nets -> AdamW, not Muon.
        Matched by name so this is robust to the module being added in a later milestone."""
        ids = set()
        for name, p in self.transformer.h.named_parameters():
            if "adaln" in name or "iter_mlp" in name:
                ids.add(id(p))
        return ids

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd

        # Route parameters inside the transformer blocks. Attention matrices and MoE expert
        # matrices (2D matmul weights) -> Muon. Router weights and IterAdaLN conditioning nets
        # -> AdamW (per the task spec: routers/modulation MLPs are not Muon-appropriate).
        router_ids = self._moe_router_param_ids()
        adaln_ids = self._adaln_param_ids()
        adamw_block_ids = router_ids | adaln_ids
        matrix_params = [p for p in self.transformer.h.parameters() if id(p) not in adamw_block_ids]
        router_params = [p for p in self.transformer.h.parameters() if id(p) in router_ids]
        adaln_params = [p for p in self.transformer.h.parameters() if id(p) in adaln_ids]

        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # AdamW group for the MoE routers (classifier-like; full matrix LR is stable for them).
        if router_params:
            param_groups.append(dict(kind='adamw', params=router_params, lr=matrix_lr * dmodel_lr_scale, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0))
        # AdamW group for IterAdaLN conditioning nets + AdaLN-Zero heads. These are given a much
        # LOWER LR (ADALN_LR_SCALE x the matrix LR): the modulation (1+gamma)/beta and the alpha
        # gate are unbounded, and at the matrix LR they diverge in bf16 within a few steps. A
        # conservative LR keeps the near-identity loop body differentiating smoothly. Tunable.
        ADALN_LR_SCALE = 0.1
        if adaln_params:
            param_groups.append(dict(kind='adamw', params=adaln_params, lr=matrix_lr * dmodel_lr_scale * ADALN_LR_SCALE, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0))
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        # Partition assertion: every model parameter lands in exactly one optimizer group.
        grouped_ids = [id(p) for g in param_groups for p in g['params']]
        all_ids = [id(p) for p in self.parameters()]
        assert len(grouped_ids) == len(set(grouped_ids)), "a parameter was assigned to multiple optimizer groups"
        assert set(grouped_ids) == set(all_ids), (
            f"optimizer groups do not partition model parameters: "
            f"{len(set(all_ids) - set(grouped_ids))} params ungrouped, "
            f"{len(set(grouped_ids) - set(all_ids))} stray"
        )
        assert len(grouped_ids) == len(all_ids), "parameter count mismatch across optimizer groups"

        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer, walking the execution schedule.
        # For the stock model this is just each block once; for the sandwich loop it is
        # prefix once + loop-body K times (shared weights, no prefix re-injection) + suffix once.
        # Per-block machinery is indexed by the UNIQUE block index `bi`; the KV-cache slot and
        # the backout midpoint use the EXECUTION position `pos` (effective depth), so a shared
        # loop block gets a distinct cache slot on each iteration.
        x0 = x  # save initial normalized embedding for x0 residual (nanochat feature; note this
                # blends the token embedding, NOT the prefix output, so it is not "re-injection")
        backout_pos = self.n_exec // 2  # cache at the effective-depth halfway point
        x_backout = None
        aux_loss_total = None  # accumulated MoE auxiliary load-balancing loss (aux_loss mode)
        loop_range = self.loop_body_range()  # (start, end) unique idx of shared body, or None
        # Gradient-checkpoint the loop body only (its activations scale with K). Training only,
        # and only when there is no KV cache (i.e. not incremental inference).
        do_ckpt = self.config.grad_checkpoint_loop and self.training and kv_cache is None
        for pos, (bi, iter_k) in enumerate(self.exec_schedule):
            x = self.resid_lambdas[bi] * x + self.x0_lambdas[bi] * x0
            ve = self.value_embeds[str(bi)](idx).to(x.dtype) if str(bi) in self.value_embeds else None
            block = self.transformer.h[bi]
            window = self.window_sizes[bi]
            is_loop_body = loop_range is not None and loop_range[0] <= bi < loop_range[1]
            # IterAdaLN iteration PE row for this loop iteration (None unless adaln loop body)
            iter_pe = self.iter_pe[iter_k] if (self.iter_pe is not None and is_loop_body) else None
            if do_ckpt and is_loop_body:
                # non-reentrant checkpoint: recomputes this block's forward in backward to save
                # activation memory; autograd still accumulates the shared-weight grads over K.
                x, aux = torch.utils.checkpoint.checkpoint(
                    block, x, ve, cos_sin, window, kv_cache, pos, iter_pe, use_reentrant=False)
            else:
                x, aux = block(x, ve, cos_sin, window, kv_cache, pos, iter_pe)
            if aux is not None:
                aux_loss_total = aux if aux_loss_total is None else aux_loss_total + aux
            if pos == backout_pos:
                x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            # Add the MoE auxiliary load-balancing loss (only populated in aux_loss mode; the
            # default bias_free strategy returns no aux loss, so this is zero-cost when disabled).
            if aux_loss_total is not None:
                loss = loss + self.config.aux_loss_coef * aux_loss_total.to(loss.dtype)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
