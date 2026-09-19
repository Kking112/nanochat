"""
Select the matrix_lr multiplier of every arm of the looped study from the LR sweep
(looped_nanochat_proposal.md section 3.3), and write it to <results-dir>/selected_lr.json
for runs/looped_main.sh.

python -m scripts.looped_select_lr

Rule: per arm, the multiplier with the lowest final val_bpb among the sweep runs. If that
optimum sits on the edge of the grid that was tried, the grid has to be extended one step in
that direction first (factor 2): the command to run is printed and nothing is written.
A diverged run is a result (it loses), it is never dropped from the table.
"""

import os
import csv
import json
import argparse

parser = argparse.ArgumentParser(description="Select matrix_lr multipliers from the LR sweep")
parser.add_argument("--results-dir", type=str, default="results")
parser.add_argument("--horizon-frac", type=float, default=0.4, help="the horizon fraction the sweep was run at")
parser.add_argument("--seed", type=int, default=0, help="the seed the sweep was run with")
parser.add_argument("--arms", type=str, default="B12,B8,B6,L8,L6s,L6p,LR", help="every one of these needs a selection")
args = parser.parse_args()

with open(os.path.join(args.results_dir, "looped_results.csv"), newline="", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if float(r["horizon_frac"]) == args.horizon_frac and int(r["seed"]) == args.seed]

selected, todo = {}, []
for arm in args.arms.split(","):
    # lr_mult -> val_bpb. A diverged run scores inf: it can never be selected, but it still marks its grid point as tried.
    sweep = {float(r["lr_mult"]): float("inf") if r["diverged"] == "1" else float(r["final_val_bpb"]) for r in rows if r["arm"] == arm}
    if len(sweep) < 3:
        todo.append(f"{arm}: only {len(sweep)} sweep runs found, need at least 3")
        continue
    mults = sorted(sweep)
    best = min(mults, key=lambda m: sweep[m])
    print(f"{arm:4s} " + "  ".join(f"{m:g}x: {sweep[m]:.5f}{' <-' if m == best else ''}" for m in mults))
    if sweep[best] == float("inf"):
        todo.append(f"{arm}: every sweep run diverged")
    elif best in (mults[0], mults[-1]):
        extend = best / 2 if best == mults[0] else best * 2
        todo.append(f'{arm}: optimum {best:g}x is on the grid edge, extend: LR_MULTS="{extend:g}" ARMS_TODO="{arm}" bash runs/looped_sweep.sh')
    else:
        selected[arm] = best

if todo:
    print("\nNo selection written:")
    for line in todo:
        print(f"  {line}")
    raise SystemExit(1)

path = os.path.join(args.results_dir, "selected_lr.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(selected, f, indent=2)
print(f"\nWrote {path}: {selected}")
