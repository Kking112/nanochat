#!/bin/bash

# LoopMoE — SMALL config (single-GPU friendly).
#
# A looped Mixture-of-Experts model with IterAdaLN, sized to run on a single 96 GB GPU:
#   ~0.97B total params, ~230M active params (< 500M active target).
#
# Architecture:
#   n_embd = 1024 (depth 6 x aspect 170, head_dim 128 -> 8 heads)
#   sandwich loop: 2 prefix + 2 loop-body + 2 suffix unique blocks, K=4 loop iterations
#                  => effective depth 2 + 4*2 + 2 = 12
#   MoE: 160 routed experts, top_k=8, 1 shared expert, auto expert-hidden=384
#        auxiliary-loss-free (bias) load balancing (DeepSeek-V3 style)
#   IterAdaLN modulation inside the loop body
#
# Run (single GPU):
#   bash runs/loopmoe_small.sh
# With wandb logging:
#   WANDB_RUN=loopmoe_small bash runs/loopmoe_small.sh
#
# NOTE: the first-correct MoE uses a per-expert Python loop (see the TODO in nanochat/gpt.py
# about a grouped-GEMM kernel), so 160 experts is correctness-first, not speed-first. Reduce
# --n-routed-experts for faster iteration while studying the model.

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# -----------------------------------------------------------------------------
# Python venv setup with uv (mirrors speedrun.sh)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Tokenizer + a little data (skip if you already ran speedrun.sh / have a tokenizer)
python -m nanochat.dataset -n 8
python -m scripts.tok_train --max-chars=2000000000 || true
python -m scripts.tok_eval || true

# -----------------------------------------------------------------------------
# Train the small LoopMoE model on a single GPU.
# --num-iterations is set explicitly: with MoE the automatic data:param horizon counts the full
# expert pool and would be enormous. Treat this as a starter horizon and increase for real runs.
python -m scripts.base_train \
    --depth=6 \
    --aspect-ratio=170 \
    --head-dim=128 \
    --window-pattern=L \
    --max-seq-len=2048 \
    --device-batch-size=16 \
    --n-prefix-layers=2 \
    --n-loop-layers=2 \
    --n-suffix-layers=2 \
    --loop-iters=4 \
    --n-routed-experts=160 \
    --n-shared-experts=1 \
    --top-k=8 \
    --load-balance-mode=bias_free \
    --use-iter-adaln \
    --num-iterations=4000 \
    --eval-every=250 \
    --sample-every=1000 \
    --save-every=500 \
    --run=$WANDB_RUN

# Notes:
# - A non-finite gradient (transient MoE/IterAdaLN loss spike) now SKIPS that optimizer step
#   instead of NaN-poisoning the whole model permanently; watch for "skipping optimizer step".
# - --save-every=500 writes resumable checkpoints. If a run dies, resume from the last good step:
#     python -m scripts.base_train ...same args... --resume-from-step=3500
# - If you see frequent skips, add e.g. --grad-clip=1.0 and/or lower --matrix-lr.

# Evaluate
python -m scripts.base_eval --device-batch-size=8 || true
