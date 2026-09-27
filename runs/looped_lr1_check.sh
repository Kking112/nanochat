#!/bin/bash

# Post-hoc check, NOT pre-registered (added 2026-09-27 after external review): the 2x matrix LR
# selected by the 40%-horizon sweep turned out too high at the full horizon (B12 at stock 1x:
# 0.8477 in Phase 0, at 2x: 0.8509 in the main matrix). This runs B8 and L8 at stock 1x, seed 0,
# full horizon, so that rho(L8) can also be reported at 1x, with the Phase 0 B12 run as B_E.
# Rows are appended to results/looped_results.csv under the arm names B8_lr1 and L8_lr1 and are
# reported in an appendix of the paper, separately from the pre-registered analysis.
#
#   bash runs/looped_lr1_check.sh

source runs/looped_common.sh

ARM_LAYOUT[B8_lr1]="8"
ARM_LAYOUT[L8_lr1]="2,4x2,2"
for arm in B8_lr1 L8_lr1; do
    train_arm "$arm" 1 1.0 0 --core-metric-every 999999
done
echo "1x check done. Record the two rows in dev/LOOPED_LOG.md and the paper appendix."
