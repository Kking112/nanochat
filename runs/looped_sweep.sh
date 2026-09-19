#!/bin/bash

# Phase 1a of the looped study: matrix_lr sweep (proposal section 3.3).
# 7 arms x {0.5x, 1x, 2x} of the stock Muon matrix_lr, seed 0, 40% of the token horizon.
# Resumable: runs that already have a row in results/looped_results.csv are skipped.
#
#   bash runs/looped_sweep.sh                      # the grid
#   LR_MULTS="4" ARMS_TODO="L6s" bash runs/looped_sweep.sh   # extend the grid for one arm
#
# Then: python -m scripts.looped_select_lr   (reports arms whose optimum is on a grid edge)

source runs/looped_common.sh

HORIZON_FRAC=0.4
LR_MULTS="${LR_MULTS:-0.5 1 2}"
ARMS_TODO="${ARMS_TODO:-${ARMS[*]}}"

for arm in $ARMS_TODO; do
    for lr_mult in $LR_MULTS; do
        # val_bpb decides the sweep: no CORE
        train_arm "$arm" "$lr_mult" "$HORIZON_FRAC" 0 --core-metric-every -1
    done
done
