#!/bin/bash

# Phase 0 of the looped study (proposal sections 3.5 and 9.8): does the refactored model reproduce
# stock nanochat, and how long does a run take on this GPU?
#   1) data and tokenizer, only if missing
#   2) 20 step check: stock d12 and the refactored --layout 12 log the same losses under torch.compile
#   3) stock d12 full run, refactored --layout 12 full run (same seed)
#   4) throughput of the looped arm L8 (200 steps)
#   5) gate report: |val_bpb difference| < 0.003, measured tokens/s, Phase 1 estimate
# Takes two full d12 runs. Resumable: finished runs are skipped.
#
#   screen -L -Logfile results/stdout/phase0.log -S phase0 bash runs/looped_phase0.sh

source runs/looped_common.sh

# -----------------------------------------------------------------------------
# 1) Data and tokenizer. ~40 shards cover the 1.3B token horizon (35% of tokens are cropped by the
# BOS-aligned packing), 170 is what speedrun.sh downloads. Existing shards are skipped.
python -m nanochat.dataset -n 170
# NEVER retrain an existing tokenizer: tok_train overwrites it, and every checkpoint in the base dir depends on it
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    python -m scripts.tok_train
    python -m scripts.tok_eval
fi

COMMON="--window-pattern L --seed 0 --device-batch-size $DEVICE_BATCH_SIZE --sample-every -1 --no-save-optimizer --run $WANDB_RUN"
stock_run()      { python -m scripts.base_train --depth 12 $COMMON "$@"; }
refactored_run() { python -m scripts.base_train --layout 12 --width-depth $WIDTH_DEPTH --ref-layout $REF_LAYOUT $COMMON "$@"; }

# -----------------------------------------------------------------------------
# 2) Identity check under torch.compile. Its results go to a directory of their own.
# Exact equality of the two models is proven in fp32 by tests T1/T2. It can not be demanded here:
# training on this GPU is not run-to-run deterministic. Two runs of unmodified stock code log the
# same loss for ~5 steps and then drift apart in the 5th-6th digit (measured: ~1e-4 by step 30),
# and stock vs --layout 12 differ by just as much. So: same first step, then within a tolerance
# ~10x above that drift and far below anything a real difference between the models would produce.
CHECK_DIR="$RESULTS_DIR/phase0_identity_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$CHECK_DIR"
CHECK="--num-iterations 20 --eval-every -1 --core-metric-every -1 --results-dir $CHECK_DIR"
stock_run      $CHECK --arm stock12 --model-tag looped_phase0_check_stock12 2>&1 | tee "$CHECK_DIR/stock12.log"
refactored_run $CHECK --arm B12     --model-tag looped_phase0_check_B12     2>&1 | tee "$CHECK_DIR/B12.log"
python - "$CHECK_DIR/stock12.log" "$CHECK_DIR/B12.log" <<'PY'
import re, sys
losses = [re.findall(r"^step \d+/\d+ .*? loss: ([\d.]+)", open(path).read(), flags=re.M) for path in sys.argv[1:]]
stock, refactored = ([float(x) for x in run] for run in losses)
assert len(stock) == len(refactored) == 20, f"expected 20 steps, got {len(stock)} and {len(refactored)}"
assert abs(stock[0] - refactored[0]) <= 1e-6, f"different loss at step 0 (same init, same data): {stock[0]} vs {refactored[0]}"
worst = max(abs(a - b) for a, b in zip(stock, refactored))
assert worst <= 1e-3, f"stock and --layout 12 diverge by {worst}:\n{stock}\n{refactored}"
print(f"IDENTITY CHECK PASSED: stock d12 and --layout 12 agree within {worst:.1e} over 20 steps (run-to-run noise of this GPU is ~1e-4)")
PY

# -----------------------------------------------------------------------------
# 3) The two full runs. CORE once at the end.
FULL="--core-metric-every 999999 --results-dir $RESULTS_DIR"
is_done stock12 1.0 1.0 0 || stock_run      $FULL --arm stock12 --model-tag looped_phase0_stock12 2>&1 | tee "$RESULTS_DIR/stdout/looped_phase0_stock12_$(date +%Y%m%d_%H%M%S).log"
is_done B12_phase0 1.0 1.0 0 || refactored_run $FULL --arm B12_phase0 --model-tag looped_phase0_B12 2>&1 | tee "$RESULTS_DIR/stdout/looped_phase0_B12_$(date +%Y%m%d_%H%M%S).log"

# -----------------------------------------------------------------------------
# 4) Throughput of the main looped arm
is_done L8_throughput 1.0 1.0 0 || python -m scripts.base_train --layout "${ARM_LAYOUT[L8]}" --width-depth $WIDTH_DEPTH --ref-layout $REF_LAYOUT $COMMON \
    --num-iterations 200 --eval-every -1 --core-metric-every -1 --results-dir "$RESULTS_DIR" \
    --arm L8_throughput --model-tag looped_phase0_L8_throughput 2>&1 | tee "$RESULTS_DIR/stdout/looped_phase0_L8_throughput_$(date +%Y%m%d_%H%M%S).log"

# -----------------------------------------------------------------------------
# 5) Gate report
python - "$RESULTS_DIR/looped_results.csv" <<'PY'
import csv, sys
rows = {r["arm"]: r for r in csv.DictReader(open(sys.argv[1], newline=""))}
stock, b12, l8 = rows["stock12"], rows["B12_phase0"], rows["L8_throughput"]
diff = abs(float(stock["final_val_bpb"]) - float(b12["final_val_bpb"]))
tokens = int(b12["tokens"])
print(f"stock d12   : val_bpb {float(stock['final_val_bpb']):.6f} | {int(stock['tokens_per_sec']):,} tok/s | {int(stock['wall_clock_s'])/3600:.2f} h | peak {stock['peak_vram_mib']} MiB")
print(f"--layout 12 : val_bpb {float(b12['final_val_bpb']):.6f} | {int(b12['tokens_per_sec']):,} tok/s | {int(b12['wall_clock_s'])/3600:.2f} h | peak {b12['peak_vram_mib']} MiB")
print(f"L8 2,4x2,2  : {int(l8['tokens_per_sec']):,} tok/s | peak {l8['peak_vram_mib']} MiB")
print(f"GATE val_bpb difference {diff:.6f} < 0.003: {'PASS' if diff < 0.003 else 'FAIL'}")
# Phase 1 = 21 sweep runs at 40% of the horizon + 21 main runs, in units of B12 runs.
# Split B12's wall-clock into training and the rest (compile, 11 val evals, CORE): the rest does not
# shrink with the horizon like training does, a sweep run still has 5 of the 11 evals (and no CORE).
# Costing every arm like B12 is on the safe side overall: B8 and B6 are much cheaper, only the LR arm
# is dearer (mean E = 14 instead of 12, so ~1.17x).
wall_hours = int(b12["wall_clock_s"]) / 3600
train_hours = int(b12["tokens"]) / int(b12["tokens_per_sec_median"]) / 3600
rest_hours = max(wall_hours - train_hours, 0)
main_run, sweep_run = wall_hours, 0.4 * train_hours + 0.5 * rest_hours
phase1_days = 21 * (main_run + sweep_run) / 24
print(f"B12 run: {wall_hours:.2f} h = {train_hours:.2f} h training + {rest_hours:.2f} h compile/evals/CORE")
print(f"Phase 1 estimate: 21 x {main_run:.2f} h + 21 x {sweep_run:.2f} h = {phase1_days:.1f} GPU-days. GATE <= 10 GPU-days: {'PASS' if phase1_days <= 10 else 'FAIL'}")
print("Not included: runs/looped_loopsweep.sh (42 evaluations of val+train bpb and CORE, at up to R=8).")
print("Record these numbers in dev/LOOPED_LOG.md. If both gates pass and all tests pass: git tag prereg-v1")
PY
