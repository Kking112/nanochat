#!/bin/bash

# Test-time loop scaling (proposal section 4.3, H4): val_bpb and CORE at inference R in {1,2,3,4,5,6,8}
# for the LR arm (trained with R ~ U{1..4}) and, as the control, L8 (trained at fixed R=2).
# Appends to results/looped_loopsweep.csv. Resumable: (model, R) pairs already there are skipped.
#
#   bash runs/looped_loopsweep.sh

source runs/looped_common.sh

SELECTED_LR="$RESULTS_DIR/selected_lr.json"
[ -f "$SELECTED_LR" ] || { echo "$SELECTED_LR not found: run the sweep and scripts.looped_select_lr first"; exit 1; }
LOOPS="${LOOPS:-1 2 3 4 5 6 8}"
SEEDS="${SEEDS:-0 1 2}"
# Same protocol as the evals inside base_train, so that R=2 reproduces the numbers of the main
# matrix: 80*524288 val tokens, 500 examples per CORE task. (base_eval's own defaults differ.)
SPLIT_TOKENS=$((80 * 524288))
MAX_PER_TASK=500

for arm in LR L8; do
    lr_mult="$(python -c "import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$SELECTED_LR" "$arm")"
    for seed in $SEEDS; do
        tag="$(model_tag "$arm" "$lr_mult" 1.0 "$seed")"
        # A run that diverged stopped before its final checkpoint, and a seed may not have been run yet.
        # Neither may stop the evaluation of all the other models.
        if ! ls "$NANOCHAT_BASE_DIR/base_checkpoints/$tag"/model_*.pt > /dev/null 2>&1; then
            echo "=== $tag: no checkpoint (diverged, or not run yet), skipping"
            continue
        fi
        for r in $LOOPS; do
            if [ -f "$RESULTS_DIR/looped_loopsweep.csv" ] && python -c "
import csv, sys
sys.exit(0 if any(row['model_tag'] == sys.argv[2] and row['num_loops'] == sys.argv[3] for row in csv.DictReader(open(sys.argv[1], newline=''))) else 1)
" "$RESULTS_DIR/looped_loopsweep.csv" "$tag" "$r"; then
                echo "=== $tag R=$r: already done, skipping"
                continue
            fi
            echo "=== $tag R=$r"
            python -m scripts.base_eval --model-tag "$tag" --num-loops "$r" --eval core,bpb \
                --device-batch-size "$DEVICE_BATCH_SIZE" --split-tokens $SPLIT_TOKENS --max-per-task $MAX_PER_TASK \
                --results-dir "$RESULTS_DIR" 2>&1 | tee "$RESULTS_DIR/stdout/loopsweep_${tag}_R${r}_$(date +%Y%m%d_%H%M%S).log"
        done
    done
done
