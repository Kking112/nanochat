#!/usr/bin/env bash
# Run from the repository root. Artifacts stay under NANOCHAT_BASE_DIR.
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: bash runs/reasoning_singlegpu.sh SOURCE INPUT_TAG INPUT_STEP RUN_TAG" >&2
    echo "Default MODE=smoke; MODE=full selects a full SFT epoch and configured RL budget." >&2
    exit 2
fi
source_name="$1"
input_tag="$2"
input_step="$3"
run_tag="$4"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mode="${MODE:-smoke}"
if [[ "$mode" != smoke && "$mode" != full ]]; then
    echo "MODE must be smoke or full" >&2
    exit 2
fi
if [[ ! "$input_step" =~ ^[0-9]+$ || ! "$run_tag" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "Provide a numeric checkpoint step and a simple run tag." >&2
    exit 2
fi

uv sync --extra gpu --group dev
run_gpu() {
    uv run --no-sync python -m scripts.reason_gpu_guard \
        --limit-percent "${GPU_VRAM_LIMIT:-80}" -- uv run --no-sync "$@"
}
sft_tag="${run_tag}-sft"
rl_tag="${run_tag}-${REWARD_MODE:-shaped}-${SEED:-42}"
sft_options=()
eval_options=()
if [[ "$mode" == smoke ]]; then
    sft_options=(--num-iterations 50 --metamath-rows 256 --omi2-rows 256
                 --gsm8k-epochs 1 --procedural-rows 512 --validation-size 32
                 --eval-tokens 8192 --chatcore-every -1)
    eval_options=(--max-examples "${EVAL_EXAMPLES:-8}")
    rl_steps="${RL_STEPS:-5}"
    prompts="${EXAMPLES_PER_STEP:-4}"
else
    # Launch this mode only after deciding the measured campaign budget.
    sft_options=(--num-iterations "${SFT_ITERATIONS:--1}")
    rl_steps="${RL_STEPS:-300}"
    prompts="${EXAMPLES_PER_STEP:-16}"
fi
run_gpu python -m scripts.reason_sft \
    --source "$source_name" --model-tag "$input_tag" --model-step "$input_step" \
    --output-tag "$sft_tag" --seed "${SEED:-42}" \
    --device-batch-size "${DEVICE_BATCH_SIZE:-2}" \
    --total-batch-size "${TOTAL_BATCH_SIZE:-16384}" "${sft_options[@]}"
run_gpu python -m scripts.reason_eval \
    --source reason_sft --model-tag "$sft_tag" \
    --output-dir "$NANOCHAT_BASE_DIR/reasoning_runs/${sft_tag}-eval" \
    --max-new-tokens 1024 --device-batch-size "${DEVICE_BATCH_SIZE:-2}" "${eval_options[@]}"
run_gpu python -m scripts.reason_rl \
    --source reason_sft --model-tag "$sft_tag" --output-tag "$rl_tag" \
    --reward-mode "${REWARD_MODE:-shaped}" --seed "${SEED:-42}" \
    --num-steps "$rl_steps" --examples-per-step "$prompts" --num-samples 8 \
    --device-batch-size "${DEVICE_BATCH_SIZE:-2}" --max-new-tokens 768
run_gpu python -m scripts.reason_eval \
    --source reason_rl --model-tag "$rl_tag" \
    --output-dir "$NANOCHAT_BASE_DIR/reasoning_runs/${rl_tag}-eval" \
    --max-new-tokens 1024 --device-batch-size "${DEVICE_BATCH_SIZE:-2}" "${eval_options[@]}"
for source_and_tag in "reason_sft:$sft_tag" "reason_rl:$rl_tag"; do
    chat_options=()
    if [[ "$mode" == smoke ]]; then chat_options=(-x "${EVAL_EXAMPLES:-8}"); fi
    run_gpu python -m scripts.chat_eval \
        -i "${source_and_tag%%:*}" -g "${source_and_tag#*:}" -m 1024 "${chat_options[@]}" \
        > "$NANOCHAT_BASE_DIR/reasoning_runs/${source_and_tag#*:}-chatcore.log"
done
