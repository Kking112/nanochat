"""
Analysis of the looped study (looped_nanochat_proposal.md sections 2, 3.4 and 4): the results
table, the pre-registered decisions on H1-H4, the recovery fraction rho, and figures F1-F6.

python -m scripts.looped_analyze

Reads <results-dir>/looped_results.csv, looped_loopsweep.csv, selected_lr.json and logs/*.jsonl.
Writes <results-dir>/analysis.md and <results-dir>/figures/F*.png. Reads only, never edits results.

Everything here was fixed before any main-matrix run existed:
- A run counts if it is a full-horizon run of one of the 7 arms at the arm's selected LR multiplier.
- Diverged runs are counted and reported per arm, and excluded from means (section 3.4).
- "a beats b" is SUPPORTED iff mean(b) - mean(a) > 2 x pooled SD and every seed of a beats every
  seed of b. Anything else is INCONCLUSIVE. Pooled SD = sqrt((sd_a^2 + sd_b^2) / 2), sample SDs.
- rho(L) = (bpb(B_U) - bpb(L)) / (bpb(B_U) - bpb(B_E)), from arm means, with the spread over all
  seed combinations (one seed per arm) as its interval. With 3 seeds that is 27 combinations:
  an interval over seed combinations, not a confidence interval.
"""

import os
import csv
import json
import glob
import argparse
import itertools
import statistics

parser = argparse.ArgumentParser(description="Analyze the looped study")
parser.add_argument("--results-dir", type=str, default="results")
args = parser.parse_args()

ARMS = ["B12", "B8", "B6", "L8", "L6s", "L6p", "LR"]
BASELINES = {"B12", "B8", "B6"}
# looped arm -> (equal-parameter baseline B_U, equal-compute baseline B_E)
CONTROLS = {"L8": ("B8", "B12"), "L6s": ("B6", "B12"), "L6p": ("B6", "B12")}
# Fixed color per arm (categorical slots in fixed order, validated for CVD separation). Shape is
# the second encoding: squares = untied baselines, circles = looped.
COLOR = dict(zip(ARMS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]))
MARKER = {arm: "s" if arm in BASELINES else "o" for arm in ARMS}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

def read_csv(name):
    path = os.path.join(args.results_dir, name)
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

# -----------------------------------------------------------------------------
# Load

with open(os.path.join(args.results_dir, "selected_lr.json"), encoding="utf-8") as f:
    selected_lr = json.load(f)
runs = {arm: [] for arm in ARMS} # arm -> rows of the main matrix, diverged ones included
for row in read_csv("looped_results.csv"):
    arm = row["arm"]
    if arm in runs and float(row["horizon_frac"]) == 1.0 and float(row["lr_mult"]) == float(selected_lr[arm]):
        runs[arm].append(row)
logs = {} # run_id -> events
for path in glob.glob(os.path.join(args.results_dir, "logs", "*.jsonl")):
    with open(path, encoding="utf-8") as f:
        events = [json.loads(line) for line in f]
    logs[events[0]["run_id"]] = events

bpb = {arm: {int(r["seed"]): float(r["final_val_bpb"]) for r in runs[arm] if r["diverged"] == "0"} for arm in ARMS}
mean = lambda arm: statistics.mean(bpb[arm].values())
sd = lambda arm: statistics.stdev(bpb[arm].values()) if len(bpb[arm]) > 1 else float("nan")

out = [] # lines of analysis.md
def emit(line=""):
    print(line)
    out.append(line)

# -----------------------------------------------------------------------------
# Results table

emit("# Looped study: analysis")
emit()
emit("Selected matrix LR multipliers: " + ", ".join(f"{arm} {selected_lr[arm]:g}x" for arm in ARMS))
emit()
emit("| arm | layout | U/E | block params | value-embed params | val_bpb per seed | mean | SD | CORE mean | diverged | tok/s median | peak VRAM MiB | train FLOPs |")
emit("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
unique_params, train_flops = {}, {}
for arm in ARMS:
    rows = runs[arm]
    if not rows:
        emit(f"| {arm} | no runs yet |" + " |" * 11)
        continue
    config = logs[rows[0]["run_id"]][0]
    counts = config["param_counts"]
    unique_params[arm] = counts["transformer_matrices"]
    train_flops[arm] = statistics.mean(float(r["train_flops"]) for r in rows)
    effective = f"{counts['effective_layers']}" if not rows[0]["train_loops"] else "8-20"
    cores = [float(r["core"]) for r in rows if r["diverged"] == "0" and r["core"]]
    seeds = ", ".join(f"s{s}: {v:.5f}" for s, v in sorted(bpb[arm].items()))
    emit(f"| {arm} | `{rows[0]['layout']}` | {counts['unique_layers']}/{effective} | {counts['transformer_matrices']/1e6:.1f}M | {counts['value_embeds']/1e6:.1f}M "
         f"| {seeds} | {mean(arm):.5f} | {sd(arm):.5f} | {statistics.mean(cores):.4f} | {sum(r['diverged'] == '1' for r in rows)}/{len(rows)} "
         f"| {statistics.median(int(r['tokens_per_sec_median']) for r in rows):,.0f} | {max(int(r['peak_vram_mib']) for r in rows):,} | {train_flops[arm]:.3e} |")
emit()

# -----------------------------------------------------------------------------
# Decisions

def compare(a, b):
    """Does arm a beat (= lower val_bpb than) arm b? Returns (verdict, detail)."""
    if len(bpb[a]) < 2 or len(bpb[b]) < 2:
        return "NOT ENOUGH RUNS", f"{a}: {len(bpb[a])} runs, {b}: {len(bpb[b])} runs"
    diff = mean(b) - mean(a)
    pooled = ((sd(a) ** 2 + sd(b) ** 2) / 2) ** 0.5
    detail = f"mean({b}) - mean({a}) = {diff:+.5f}, 2 x pooled SD = {2 * pooled:.5f}"
    if diff > 2 * pooled and max(bpb[a].values()) < min(bpb[b].values()):
        return f"SUPPORTED: {a} beats {b}", detail + f", all seeds of {a} beat all seeds of {b}"
    if -diff > 2 * pooled and max(bpb[b].values()) < min(bpb[a].values()):
        return f"SUPPORTED IN REVERSE: {b} beats {a}", detail + f", all seeds of {b} beat all seeds of {a}"
    return "INCONCLUSIVE", detail

def rho(looped):
    """Recovery fraction from arm means, and its spread over all seed combinations."""
    b_u, b_e = CONTROLS[looped]
    if not (bpb[looped] and bpb[b_u] and bpb[b_e]):
        return None
    point = (mean(b_u) - mean(looped)) / (mean(b_u) - mean(b_e))
    combos = sorted((u - l) / (u - e) for u, l, e in itertools.product(bpb[b_u].values(), bpb[looped].values(), bpb[b_e].values()))
    return point, combos[0], statistics.median(combos), combos[-1], len(combos)

emit("## Hypotheses (decision rule of section 3.4)")
emit()
emit("**H1, looping helps at equal parameters** (needs L8 < B8 and L6* < B6):")
for a, b in (("L8", "B8"), ("L6s", "B6"), ("L6p", "B6")):
    verdict, detail = compare(a, b)
    emit(f"- {a} vs {b}: **{verdict}**. {detail}")
emit()
emit("**H2, sharing has a cost at equal compute** (needs B12 < L8; prediction 0.3 <= rho(L8) <= 0.8):")
verdict, detail = compare("B12", "L8")
emit(f"- B12 vs L8: **{verdict}**. {detail}")
rhos = {arm: rho(arm) for arm in CONTROLS}
for arm, r in rhos.items():
    if r is not None:
        point, lo, med, hi, n = r
        note = ""
        if arm == "L8":
            note = " Inside the predicted range." if 0.3 <= point <= 0.8 else " OUTSIDE the predicted range [0.3, 0.8]: the prediction is falsified" + (" (rho >= 1: looping matches or beats the larger untied model)." if point >= 1 else " (rho <= 0: looping adds nothing)." if point <= 0 else ".")
        emit(f"- rho({arm}) = {point:.3f} (over {n} seed combinations: min {lo:.3f}, median {med:.3f}, max {hi:.3f}).{note}")
emit()
emit("**H3, layout at fixed parameters and compute** (weak prediction: sandwich `2,2x4,2` not worse than pure loop `0,6x2,0`):")
verdict, detail = compare("L6s", "L6p")
emit(f"- L6s vs L6p: **{verdict}**. {detail}")
emit()

# H4: test-time loops
sweep = {} # arm -> seed -> R -> val_bpb
for row in read_csv("looped_loopsweep.csv"):
    if row["arm"] in ("LR", "L8") and row["val_bpb"]:
        sweep.setdefault(row["arm"], {}).setdefault(int(row["seed"]), {})[int(row["num_loops"])] = float(row["val_bpb"])
emit("**H4, test-time loops** (LR arm: val_bpb decreases monotonically over R = 1..4, and R in {5,6,8} is no worse than R = 4 by more than 0.005):")
if "LR" not in sweep:
    emit("- no loop sweep yet")
for seed, curve in sorted(sweep.get("LR", {}).items()):
    trained = [curve.get(r) for r in (1, 2, 3, 4)]
    if None in trained:
        emit(f"- LR seed {seed}: incomplete")
        continue
    monotone = all(a > b for a, b in zip(trained, trained[1:]))
    beyond = {r: curve[r] - curve[4] for r in (5, 6, 8) if r in curve}
    holds = all(d <= 0.005 for d in beyond.values())
    emit(f"- LR seed {seed}: monotone over 1..4: **{monotone}** ({', '.join(f'{v:.5f}' for v in trained)}); "
         f"beyond the trained range vs R=4: {', '.join(f'R={r}: {d:+.5f}' for r, d in beyond.items())} => within 0.005: **{holds}**; "
         f"improves beyond R=4: {any(d < 0 for d in beyond.values())}")
emit()
emit("Note for H4: the backout tap is the residual after effective layer E // 2, so it moves with R (spec 9.3). The curve varies the backout point together with the loop count.")
emit()

# -----------------------------------------------------------------------------
# Figures

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def plain_log_axis(ax):
    """Log y axis with plain decimal tick labels (0.9, 2, 3 and not 9x10^-1, 2x10^0)."""
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.tick_params(axis="y", which="minor", labelsize=8)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "lines.linewidth": 2, "lines.markersize": 8, "legend.frameon": False,
})
figures_dir = os.path.join(args.results_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

def save(fig, name):
    path = os.path.join(figures_dir, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    emit(f"- `{path}`")

def mark(ax, x, arm, **kwargs):
    """One arm: every seed as a small dot, the mean as the marker with a surface ring, labeled directly."""
    values = list(bpb[arm].values())
    ax.scatter([x] * len(values), values, s=14, color=COLOR[arm], alpha=0.6, linewidths=0, zorder=2)
    ax.scatter([x], [mean(arm)], s=80, marker=MARKER[arm], color=COLOR[arm], edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.annotate(arm, (x, mean(arm)), xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=9, **kwargs)

have = [arm for arm in ARMS if bpb[arm] and arm in unique_params]
emit("## Figures")
emit()
if have:
    # F1: val_bpb vs unique parameters
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for arm in have:
        mark(ax, unique_params[arm] / 1e6, arm)
    ax.set_xlabel("unique block parameters (millions), embeddings excluded")
    ax.set_ylabel("final val_bpb (lower is better)")
    ax.set_title("F1  Loss vs stored parameters   ■ untied   ● looped")
    ax.margins(x=0.12)
    save(fig, "F1_bpb_vs_params.png")

    # F2: val_bpb vs training FLOPs
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for arm in have:
        mark(ax, train_flops[arm] / 1e18, arm)
    ax.set_xlabel("training FLOPs (1e18), same tokens for every arm")
    ax.set_ylabel("final val_bpb (lower is better)")
    ax.set_title("F2  Loss vs training compute   ■ untied   ● looped")
    ax.margins(x=0.12)
    save(fig, "F2_bpb_vs_flops.png")

if any(r is not None for r in rhos.values()):
    # F3: recovery fraction. A single measure against two reference lines: bars from the 0 baseline.
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    shown = [arm for arm in CONTROLS if rhos[arm] is not None]
    for i, arm in enumerate(shown):
        point, lo, _, hi, _ = rhos[arm]
        ax.bar(i, point, width=0.5, color=COLOR[arm], zorder=2)
        ax.plot([i, i], [lo, hi], color=INK2, linewidth=1.5, zorder=3) # spread over seed combinations
        ax.annotate(f"{point:.2f}", (i, max(hi, point)), xytext=(0, 5), textcoords="offset points", ha="center", color=INK, fontsize=9)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.axhline(1, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("1 = matches the untied 12-layer model", (1, 1), xycoords=("axes fraction", "data"), xytext=(0, 4), textcoords="offset points", ha="right", color=MUTED, fontsize=8)
    ax.set_ylim(top=max(1.12, ax.get_ylim()[1]))
    ax.set_xticks(range(len(shown)), [f"{arm}\n{runs[arm][0]['layout']}" for arm in shown])
    ax.set_ylabel("recovery fraction rho")
    ax.set_title("F3  Share of the missing-layers loss recovered by looping")
    ax.grid(axis="x", visible=False)
    save(fig, "F3_recovery_fraction.png")

if sweep:
    # F4: val_bpb vs inference loop count
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for arm in ("LR", "L8"):
        if arm not in sweep:
            continue
        loops = sorted({r for curve in sweep[arm].values() for r in curve})
        for curve in sweep[arm].values():
            ax.plot(sorted(curve), [curve[r] for r in sorted(curve)], color=COLOR[arm], linewidth=1, alpha=0.35, zorder=2)
        means = [statistics.mean(curve[r] for curve in sweep[arm].values() if r in curve) for r in loops]
        ax.plot(loops, means, color=COLOR[arm], marker=MARKER[arm], markeredgecolor=SURFACE, markeredgewidth=2, zorder=3,
                label=f"{arm}: trained with R {'~ U{1..4}' if arm == 'LR' else '= 2'}")
        ax.annotate(arm, (loops[-1], means[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.axvspan(0.5, 4.5, color=GRID, alpha=0.45, zorder=1, linewidth=0)
    ax.annotate("loop counts seen in LR training", (2.5, 1), xycoords=("data", "axes fraction"), xytext=(0, -12), textcoords="offset points", ha="center", color=MUTED, fontsize=8)
    plain_log_axis(ax)
    ax.set_xticks(sorted({r for arm in sweep for curve in sweep[arm].values() for r in curve})) # only the R that were evaluated
    ax.set_xlabel("loop count R at inference")
    ax.set_ylabel("val_bpb (log scale, lower is better)")
    ax.set_title("F4  Test-time loop scaling (thin lines: seeds)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2) # below the plot: the curves may be anywhere inside it
    ax.margins(x=0.1)
    save(fig, "F4_bpb_vs_inference_loops.png")

curves = {} # arm -> step -> [val_bpb per seed]
residual = {} # arm -> [rms per effective layer], seed mean at the final eval, at the arm's own R
for arm in ARMS:
    for row in runs[arm]:
        if row["diverged"] == "1":
            continue
        events = logs[row["run_id"]]
        for e in events:
            if e["kind"] == "eval":
                curves.setdefault(arm, {}).setdefault(e["step"], []).append(e["val_bpb"])
        last = [e for e in events if e["kind"] == "diag"][-1]["residual_rms"]
        own_r = str(events[0]["model_config"]["n_loop"])
        residual.setdefault(arm, []).append(last[own_r])

if curves:
    # F5: training curves. The arms end close together, so the legend carries identity and the
    # direct labels are left to F1, where the final values are spread out.
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for arm in ARMS:
        if arm in curves:
            steps = sorted(s for s in curves[arm] if s >= 500) # the first 500 steps would squash everything after them
            ax.plot(steps, [statistics.mean(curves[arm][s]) for s in steps], color=COLOR[arm], label=arm,
                    linestyle="-" if arm not in BASELINES else (0, (5, 2)))
    plain_log_axis(ax)
    ax.set_xlabel("optimizer step (524,288 tokens each), from step 500")
    ax.set_ylabel("val_bpb (log scale, seed mean)")
    ax.set_title("F5  Validation loss during training")
    ax.legend(ncols=4, loc="upper right", title="dashed: untied   solid: looped", title_fontsize=9)
    save(fig, "F5_training_curves.png")

if residual:
    # F6: residual stream RMS after every effective layer, end of training
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for arm in ARMS:
        if arm in residual:
            rms = [statistics.mean(layer) for layer in zip(*residual[arm])]
            ax.plot(range(1, len(rms) + 1), rms, color=COLOR[arm], marker=MARKER[arm], markersize=5, markeredgecolor=SURFACE, markeredgewidth=1,
                    label=arm, linestyle="-" if arm not in BASELINES else (0, (5, 2)))
    # no direct labels: with 7 series that end together they pile up, the legend carries identity
    ax.set_xlabel("effective layer (loops unrolled)")
    ax.set_ylabel("residual stream RMS, end of training")
    ax.set_title("F6  Residual growth through depth")
    ax.legend(ncols=4, loc="upper left", title="dashed: untied   solid: looped", title_fontsize=9)
    ax.margins(x=0.08)
    save(fig, "F6_residual_rms.png")

with open(os.path.join(args.results_dir, "analysis.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print(f"\nWrote {os.path.join(args.results_dir, 'analysis.md')}")
