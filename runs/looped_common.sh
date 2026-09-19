#!/bin/bash

# Shared setup for the looped-transformer study (looped_nanochat_proposal.md). Source it, don't run it:
#   source runs/looped_common.sh
# Single GPU only: everything runs as `python -m scripts.base_train`, no torchrun.

set -euo pipefail

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1 # output is piped through tee: without this the step lines arrive in big delayed chunks
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# The codebase and its environment are pinned for the study. ALWAYS --frozen: a plain `uv sync` /
# `uv run` was seen to re-resolve uv.lock to the latest versions of nearly every dependency.
export UV_FROZEN=1
command -v uv &> /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
# Only set the environment up if there is none: these scripts get started while another one is
# training, and that one's .venv must not be touched. After a deliberate change of uv.lock, sync by hand.
[ -d ".venv" ] || { uv venv; uv sync --frozen --extra gpu --group dev; }
source .venv/bin/activate

WANDB_RUN="${WANDB_RUN:-dummy}" # "dummy" disables wandb, the local JSONL + CSV are always written
RESULTS_DIR="${RESULTS_DIR:-results}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}" # raise if VRAM allows: the total batch size stays at the reference value
mkdir -p "$RESULTS_DIR/stdout"

# The arms of the study (proposal section 3.1). All at d12 width, all with B12's horizon and hyperparameters.
ARMS=(B12 B8 B6 L8 L6s L6p LR)
declare -A ARM_LAYOUT=( [B12]="12" [B8]="8" [B6]="6" [L8]="2,4x2,2" [L6s]="2,2x4,2" [L6p]="0,6x2,0" [LR]="2,4x2,2" )
declare -A ARM_EXTRA=( [LR]="--train-loops 1,2,3,4" ) # LR: R ~ U{1..4} per training step, evaluated at R=2
WIDTH_DEPTH=12
REF_LAYOUT=12

# All attention in the study must be SDPA with full context. Blackwell has no FA3 kernel.
python -c "
from nanochat.flash_attention import USE_FA3
from nanochat.common import COMPUTE_DTYPE
assert not USE_FA3, 'the study runs on SDPA: FA3 got selected on this machine'
print(f'attention backend: sdpa | compute dtype: {COMPUTE_DTYPE}')
"

model_tag() { # arm lr_mult horizon_frac seed. Never of the form d<number>: those are stock nanochat tags.
    echo "looped_${1}_lr${2}_h${3}_s${4}"
}

is_done() { # arm lr_mult horizon_frac seed: is there a row for this run already? (a diverged run counts: it is a result)
    python - "$RESULTS_DIR/looped_results.csv" "$@" <<'PY'
import csv, os, sys
path, arm, lr_mult, horizon_frac, seed = sys.argv[1:]
done = os.path.exists(path) and any(
    r["arm"] == arm and float(r["lr_mult"]) == float(lr_mult) and float(r["horizon_frac"]) == float(horizon_frac) and r["seed"] == seed
    for r in csv.DictReader(open(path, newline="")))
sys.exit(0 if done else 1)
PY
}

train_arm() { # arm lr_mult horizon_frac seed [extra base_train args...]
    local arm="$1" lr_mult="$2" horizon_frac="$3" seed="$4"; shift 4
    local tag; tag="$(model_tag "$arm" "$lr_mult" "$horizon_frac" "$seed")"
    if is_done "$arm" "$lr_mult" "$horizon_frac" "$seed"; then
        echo "=== $tag: already in $RESULTS_DIR/looped_results.csv, skipping"
        return 0
    fi
    echo "=== $tag"
    # shellcheck disable=SC2086
    python -m scripts.base_train \
        --layout "${ARM_LAYOUT[$arm]}" --width-depth $WIDTH_DEPTH --ref-layout $REF_LAYOUT ${ARM_EXTRA[$arm]:-} \
        --window-pattern L \
        --matrix-lr-mult "$lr_mult" --horizon-frac "$horizon_frac" --seed "$seed" \
        --device-batch-size "$DEVICE_BATCH_SIZE" \
        --arm "$arm" --model-tag "$tag" --results-dir "$RESULTS_DIR" --run "$WANDB_RUN" \
        --sample-every -1 --no-save-optimizer \
        "$@" 2>&1 | tee "$RESULTS_DIR/stdout/${tag}_$(date +%Y%m%d_%H%M%S).log"
}
