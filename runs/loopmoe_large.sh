#!/bin/bash

# LoopMoE — LARGE config (multi-GPU, sharded).
#
# A looped Mixture-of-Experts model that decouples total from active params:
#   ~10.96B total params, ~2.39B active params (compute stays modest; the routed-expert pool
#   carries the parameter count).
#
# Architecture:
#   n_embd = 2432 (depth 12 x aspect 200, head_dim 128 -> 19 heads)
#   sandwich loop: 4 prefix + 4 loop-body + 4 suffix unique blocks, K=4 loop iterations
#                  => effective depth 4 + 4*4 + 4 = 24
#   MoE: 144 routed experts, top_k=6, 2 shared experts, auto expert-hidden=1152
#        auxiliary-loss-free (bias) load balancing
#   IterAdaLN modulation inside the loop body
#   gradient checkpointing on the loop body (activation memory scales with K)
#
# This config REQUIRES the sharded path: a replicated 10B+ model + optimizer will not fit.
# It uses FSDP2 (--fsdp), which shards parameters per block and trains with plain AdamW
# (Muon's hand-rolled collectives are incompatible with FSDP's sharded DTensors).
#
# Run on an 8-GPU node:
#   torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- <args below>
# or simply:
#   bash runs/loopmoe_large.sh
# With wandb logging:
#   WANDB_RUN=loopmoe_large bash runs/loopmoe_large.sh
#
# NOTE: the correctness-first per-expert loop makes 144 experts slow (see the grouped-GEMM TODO
# in nanochat/gpt.py). This config is meant to demonstrate the sharded 10B+ path, not to be fast.

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

NPROC="${NPROC:-8}"  # number of GPUs

# -----------------------------------------------------------------------------
# Python venv setup with uv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Tokenizer + data (skip if you already have a tokenizer + shards)
python -m nanochat.dataset -n 24
python -m scripts.tok_train --max-chars=2000000000 || true
python -m scripts.tok_eval || true

# -----------------------------------------------------------------------------
# Train the large LoopMoE model, sharded across GPUs with FSDP2.
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    --depth=12 \
    --aspect-ratio=200 \
    --head-dim=128 \
    --window-pattern=L \
    --max-seq-len=2048 \
    --device-batch-size=4 \
    --fsdp \
    --n-prefix-layers=4 \
    --n-loop-layers=4 \
    --n-suffix-layers=4 \
    --loop-iters=4 \
    --n-routed-experts=144 \
    --n-shared-experts=2 \
    --top-k=6 \
    --load-balance-mode=bias_free \
    --use-iter-adaln \
    --grad-checkpoint-loop \
    --num-iterations=2000 \
    --eval-every=250 \
    --sample-every=1000 \
    --save-every=250 \
    --run=$WANDB_RUN

# A non-finite gradient (transient MoE/IterAdaLN spike) skips that step rather than NaN-poisoning
# the model; --save-every=250 gives resumable checkpoints (--resume-from-step=N to continue).
