#!/bin/bash

# Phase 1b of the looped study: the main matrix, 7 arms x seeds {0,1,2} at the full horizon, each
# arm with the matrix_lr multiplier selected by the sweep (results/selected_lr.json, written by
# `python -m scripts.looped_select_lr`).
# Resumable at the granularity of whole runs: runs that already have a row in
# results/looped_results.csv are skipped. A run that was interrupted is rerun from scratch, never
# resumed mid-run: the dataloader's resume is approximate and would change the token stream.
#
#   bash runs/looped_main.sh

source runs/looped_common.sh

SELECTED_LR="$RESULTS_DIR/selected_lr.json"
[ -f "$SELECTED_LR" ] || { echo "$SELECTED_LR not found: run the sweep and scripts.looped_select_lr first"; exit 1; }
SEEDS="${SEEDS:-0 1 2}"
ARMS_TODO="${ARMS_TODO:-${ARMS[*]}}"

for seed in $SEEDS; do # seed-major: a complete seed of every arm is a usable partial result
    for arm in $ARMS_TODO; do
        lr_mult="$(python -c "import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$SELECTED_LR" "$arm")"
        # CORE once, at the final step. val_bpb every 250 steps (the default) for the curves.
        train_arm "$arm" "$lr_mult" 1.0 "$seed" --core-metric-every 999999
    done
done
