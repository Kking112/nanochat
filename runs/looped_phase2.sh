#!/bin/bash

# Phase 2 of the looped study (proposal section 3.5): one-seed confirmation of H1 at d20 width.
# Arms: B_E = '20' (E = 20), B_U = '12' and the looped '2,8x2,2' (U = 12, E = 20), all at B_E's
# compute-optimal horizon (5.22B tokens, batch 1M, 4980 steps).
#
# The matrix LR multiplier is not fixed by the proposal. Decided by the author on 2026-09-22:
#   1) Phase 1b full-horizon logs showed no instability at 2x => the grid is {1x, 2x}
#      (had there been any, it would have been {0.5x, 1x});
#   2) mini-sweep on B_U (the cheaper arm), 40% horizon, seed 0, select by val_bpb;
#      if the two are within 0.003 bpb, prefer 1x;
#   3) the selected multiplier is used for all three arms.
# Resumable per whole run (finished runs are skipped).
#
#   screen -L -Logfile results/stdout/phase2.log -S phase2 bash runs/looped_phase2.sh

source runs/looped_common.sh

WIDTH_DEPTH=20
REF_LAYOUT=20
ARM_LAYOUT[d20_B20]="20"
ARM_LAYOUT[d20_B12]="12"
ARM_LAYOUT[d20_L12]="2,8x2,2"

# 1+2) mini-sweep on B_U
for lr_mult in 1 2; do
    train_arm d20_B12 "$lr_mult" 0.4 0 --core-metric-every -1
done
SELECTED="$RESULTS_DIR/selected_lr_phase2.json"
python - "$RESULTS_DIR/looped_results.csv" "$SELECTED" <<'PY'
import csv, json, sys
rows = {float(r["lr_mult"]): r for r in csv.DictReader(open(sys.argv[1], newline=""))
        if r["arm"] == "d20_B12" and float(r["horizon_frac"]) == 0.4 and r["seed"] == "0"}
bpb = {m: float("inf") if rows[m]["diverged"] == "1" else float(rows[m]["final_val_bpb"]) for m in (1.0, 2.0)}
print(f"d20_B12 at 40% horizon: 1x {bpb[1.0]:.5f}, 2x {bpb[2.0]:.5f}, difference {bpb[1.0] - bpb[2.0]:+.5f}")
selected = 2.0 if bpb[2.0] < bpb[1.0] - 0.003 else 1.0
print(f"selected {selected:g}x" + (" (within 0.003: 1x preferred)" if selected == 1.0 and bpb[2.0] < bpb[1.0] else ""))
json.dump({"d20_B20": selected, "d20_B12": selected, "d20_L12": selected, "rule": "2x only if better than 1x by more than 0.003 bpb on d20_B12 at 40% horizon"}, open(sys.argv[2], "w"), indent=2)
PY
lr_mult="$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['d20_B20'])" "$SELECTED")"

# 3) the three arms, CORE once at the end
for arm in d20_B20 d20_B12 d20_L12; do
    train_arm "$arm" "$lr_mult" 1.0 0 --core-metric-every 999999
done
echo "Phase 2 done. Record the three rows in dev/LOOPED_LOG.md."
