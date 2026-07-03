"""
FSDP2 (fully_shard) helper for the large LoopMoE config.

nanochat's default training path does NOT use a DDP wrapper: the MuonAdamW optimizer replicates
the model on every rank and shards only the optimizer *state* (ZeRO-2 style). That is enough for
small/medium models. For the large config (>=10B total params, dominated by the routed-expert
pool) a replicated model does not fit, so we need to shard the *parameters* themselves.

This module applies PyTorch FSDP2 (`torch.distributed.fsdp.fully_shard`, available in torch 2.9)
per transformer Block, plus the embeddings and lm_head, and finally the root module. Sharding
per block is a sensible boundary: each block (its attention matrices and, crucially, its MoE
expert pool) is a large, self-contained parameter group.

The FSDP path is gated behind `--fsdp` in scripts/base_train.py and pairs with the plain-AdamW
optimizer (`GPT.setup_adamw_optimizer`), because FSDP2 manages parameter/gradient communication
itself (params become DTensors) which is incompatible with MuonAdamW's hand-rolled collectives.

MoE correctness across ranks is unaffected: the per-expert token-count all-reduce and the
bias-free bias update in nanochat/gpt.py use the default process group, which FSDP also uses, so
load statistics remain global.
"""

import torch


def is_fsdp_available():
    try:
        from torch.distributed.fsdp import fully_shard  # noqa: F401
        return True
    except Exception:
        return False


def apply_fsdp(model):
    """Shard a GPT model in place with FSDP2.

    Args:
        model: the (uncompiled) GPT instance, already on the target device.

    Returns the same model (fully_shard mutates in place and also returns it).

    Note on mixed precision: we deliberately do NOT attach a FSDP MixedPrecisionPolicy. nanochat's
    custom `Linear` already casts its (fp32 master) weight to the activation dtype inside forward,
    so compute runs in bf16 regardless. FSDP2 shards each parameter individually (no flat-param),
    so the model's mixed native dtypes (bf16 embeddings, fp32 matrices) are fine without a policy;
    a MixedPrecisionPolicy would instead assert on the non-uniform original dtype.
    TODO: a bf16 parameter all-gather (via a uniform-fp32 model + MixedPrecisionPolicy) would cut
    communication volume for the large config; left as a follow-up.
    """
    from torch.distributed.fsdp import fully_shard

    # FSDP2 requires a uniform parameter dtype within each shard group. Stock nanochat pre-casts
    # the token/value embeddings to the compute dtype (bf16) as a memory optimization, which makes
    # the root group's dtypes non-uniform (bf16 embeddings + fp32 scalars). Cast all *parameters*
    # back to fp32 for uniformity; forward re-casts activations to the compute dtype anyway, so
    # behavior is unchanged. Buffers (cos/sin rotary, iter_pe) are left in their compute dtype
    # (the forward pass asserts cos.dtype == COMPUTE_DTYPE).
    model.transformer.wte.to(dtype=torch.float32)
    for ve in model.value_embeds.values():
        ve.to(dtype=torch.float32)

    # Shard each transformer block independently (attention + MoE expert pool per block).
    for block in model.transformer.h:
        fully_shard(block)
    # Shard the big embedding-like modules too.
    fully_shard(model.transformer.wte)
    fully_shard(model.lm_head)
    # Finally shard the root (covers the remaining small params / scalars).
    fully_shard(model)
    return model
