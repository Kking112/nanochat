"""
Test the pre-registered decision rule of scripts/looped_analyze.py on synthetic results.

python -m pytest tests/test_looped_analyze.py -v
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent

# arm -> (layout, U, E, val_bpb of seeds 0,1,2). Chosen so that every branch of the rule is hit:
# L8 beats B8 clearly; L6s "beats" B6 on the mean but one seed overlaps; L6p has a diverged run.
ARMS = {
    "B12": ("12", 12, 12, [0.848, 0.849, 0.847]),
    "B8":  ("8", 8, 8, [0.872, 0.873, 0.871]),
    "B6":  ("6", 6, 6, [0.893, 0.894, 0.880]),
    "L8":  ("2,4x2,2", 8, 12, [0.860, 0.861, 0.859]),
    "L6s": ("2,2x4,2", 6, 12, [0.876, 0.877, 0.881]),
    "L6p": ("0,6x2,0", 6, 12, [0.878, 0.879, None]), # None = diverged
    "LR":  ("2,4x2,2", 8, 12, [0.864, 0.865, 0.863]),
}


def write_results(results_dir):
    (results_dir / "logs").mkdir(parents=True)
    (results_dir / "selected_lr.json").write_text(json.dumps({arm: 2.0 if arm == "L8" else 1.0 for arm in ARMS}))
    rows, sweep = [], []
    for arm, (layout, U, E, values) in ARMS.items():
        for seed, v in enumerate(values):
            run_id = f"looped_{arm}_s{seed}"
            n_loop = 2 if "x" in layout else 1
            events = [{"kind": "config", "run_id": run_id, "model_config": {"n_loop": n_loop},
                       "param_counts": {"transformer_matrices": U * 7077888, "value_embeds": 0, "unique_layers": U, "effective_layers": E}}]
            for step in (0, 500, 2520):
                events.append({"kind": "eval", "step": step, "val_bpb": (v or 9.0) + 1.0 / (1 + step)})
                events.append({"kind": "diag", "step": step, "residual_rms": {str(n_loop): [float(i) for i in range(E)]}})
            (results_dir / "logs" / f"{run_id}.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
            rows.append(dict(arm=arm, layout=layout, seed=seed, lr_mult=2.0 if arm == "L8" else 1.0, horizon_frac=1.0,
                             final_val_bpb="nan" if v is None else v, core=0.15, tokens_per_sec_median=260000, peak_vram_mib=28000,
                             train_flops=1.0e18, diverged=int(v is None), train_loops="1,2,3,4" if arm == "LR" else "", run_id=run_id))
    rows.append(dict(rows[0], arm="L8", lr_mult=1.0, final_val_bpb=0.5))   # L8 at a multiplier that was NOT selected: must be ignored
    rows.append(dict(rows[0], arm="B8", horizon_frac=0.4, final_val_bpb=0.5)) # a sweep run: must be ignored
    rows.append(dict(rows[0], arm="stock12", final_val_bpb=0.5))             # a phase 0 run: must be ignored
    for seed in range(3):
        for r, v in zip((1, 2, 3, 4, 5, 6, 8), (0.880, 0.870, 0.866, 0.864, 0.865, 0.866, 0.868 if seed else 0.870)):
            sweep.append(dict(arm="LR", seed=seed, num_loops=r, val_bpb=v))
    for name, data in (("looped_results.csv", rows), ("looped_loopsweep.csv", sweep)):
        with open(results_dir / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)


def test_decision_rule_and_figures(tmp_path):
    results_dir = tmp_path / "results"
    write_results(results_dir)
    proc = subprocess.run([sys.executable, "-m", "scripts.looped_analyze", "--results-dir", str(results_dir)], cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-3000:]
    report = (results_dir / "analysis.md").read_text()
    line = lambda start: next(l for l in report.splitlines() if l.startswith(start))

    # only the selected multiplier, full horizon, study arms: none of the 0.5 decoys may leak into a mean
    assert "| 0.86000 |" in line("| L8 ") and "0.50000" not in report
    assert "SUPPORTED: L8 beats B8" in line("- L8 vs B8")
    # the mean of L6s is better than B6's, but one seed of B6 (0.880) beats one of L6s (0.881)
    assert "INCONCLUSIVE" in line("- L6s vs B6")
    assert "SUPPORTED: B12 beats L8" in line("- B12 vs L8")
    # diverged run: counted, excluded from the mean
    assert "| 1/3 |" in line("| L6p ") and "| 0.87850 |" in line("| L6p ")
    # rho(L8) = (0.872 - 0.860) / (0.872 - 0.848) = 0.5
    assert "rho(L8) = 0.500" in report and "27 seed combinations" in line("- rho(L8)") and "Inside the predicted range" in line("- rho(L8)")
    # H4: monotone over 1..4 for all seeds; R=8 is +0.006 for seed 0 (fails the 0.005 bound), +0.004 for the others
    assert "monotone over 1..4: **True**" in line("- LR seed 0") and "within 0.005: **False**" in line("- LR seed 0")
    assert "within 0.005: **True**" in line("- LR seed 1")
    for name in ("F1_bpb_vs_params", "F2_bpb_vs_flops", "F3_recovery_fraction", "F4_bpb_vs_inference_loops", "F5_training_curves", "F6_residual_rms"):
        assert (results_dir / "figures" / f"{name}.png").stat().st_size > 10000
