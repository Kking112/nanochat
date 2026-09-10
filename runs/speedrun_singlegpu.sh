#!/bin/bash

# Single-GPU variant of runs/speedrun.sh
# ---------------------------------------------------------------------------
# This trains the same GPT-2 grade d24 model as the reference speedrun, but on
# ONE GPU instead of an 8XH100 node. The only functional change is dropping
# `torchrun --nproc_per_node=8`: nanochat then automatically falls back to
# gradient accumulation, keeping the same effective (total) batch size, so the
# results are ~identical to the multi-GPU run.
#
# Expect this to take roughly 8x+ the wall-clock of the 8XH100 run (~3h),
# i.e. on the order of a day for pretraining alone, plus tokenizer/eval/SFT.
# Run it inside a screen/tmux session so it survives disconnects:
#   screen -L -Logfile runs/speedrun_singlegpu.log -S speedrun bash runs/speedrun_singlegpu.sh
#
# Tested target: a single large-VRAM card (>=80GB), e.g. RTX Pro 6000 Blackwell
# (96GB) or H100 (80GB). With 96GB you have headroom to raise DEVICE_BATCH_SIZE
# toward 32; if you OOM on a smaller card, lower it (16 -> 8 -> 4 -> 2 -> 1).
# ---------------------------------------------------------------------------

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Tunables (override from the environment, e.g. `DEVICE_BATCH_SIZE=32 bash ...`)

# Per-device micro-batch. 16 matches the reference speedrun and fits in <=80GB.
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"

# FP8 is a speed optimization that the reference d24 baseline does NOT require.
# It officially targets H100+; Blackwell has fp8 tensor cores but the torchao
# fp8 path on sm_120 can be a sharp edge. Default OFF for portability. Set
# USE_FP8=1 to opt in (and just unset it again if you hit an fp8 error).
USE_FP8="${USE_FP8:-0}"
if [ "$USE_FP8" = "1" ]; then
    FP8_FLAG="--fp8"
else
    FP8_FLAG=""
fi

# Attention window pattern. The reference speedrun uses sliding-window 'SSSL',
# but Flash Attention 3 is Hopper-only, so on Blackwell (and any non-Hopper GPU)
# nanochat falls back to PyTorch SDPA, which has no efficient sliding-window path
# -> "GPU utilization will be terrible". 'L' = full-context attention on every
# layer, which SDPA handles well. This deviates slightly from the reference
# recipe (results may differ marginally) but is far more efficient here. If you
# install an FA3 build for your GPU, set WINDOW_PATTERN=SSSL to match the speedrun.
WINDOW_PATTERN="${WINDOW_PATTERN:-L}"

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies (cu128 wheels include Blackwell sm_120 kernels)
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup (optional). Set WANDB_RUN=<name> to enable logging; otherwise the
# special value "dummy" disables it.
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Reset the report (writes system info + start timestamp into the base dir)
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Download the first ~2B characters of pretraining dataset (8 shards ~ 800MB).
python -m nanochat.dataset -n 8
# Kick off downloading the rest in the background while the tokenizer trains.
# ~150 shards are needed for GPT-2 capability pretraining, +20 for padding.
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
# train + evaluate the tokenizer (vocab size 2**15 = 32768 on ~2B chars)
python -m scripts.tok_train
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining) -- SINGLE GPU (no torchrun => gradient accumulation)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# d24 model (slightly undertrained to beat GPT-2 => data:param ratio 8 instead of 12)
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=24 \
    --target-param-data-ratio=8 \
    --device-batch-size=$DEVICE_BATCH_SIZE \
    --window-pattern=$WINDOW_PATTERN \
    $FP8_FLAG \
    --run=$WANDB_RUN
# evaluate the model: CORE metric, BPB on train/val, and draw samples
torchrun --standalone --nproc_per_node=1 -m scripts.base_eval -- --device-batch-size=$DEVICE_BATCH_SIZE


# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run SFT and eval the model (single GPU)
torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- --device-batch-size=$DEVICE_BATCH_SIZE --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
python -m nanochat.report generate
