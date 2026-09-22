# Looped study: analysis

Selected matrix LR multipliers: B12 2x, B8 2x, B6 2x, L8 2x, L6s 2x, L6p 2x, LR 2x

| arm | layout | U/E | block params | value-embed params | val_bpb per seed | mean | SD | CORE mean | diverged | tok/s median | peak VRAM MiB | train FLOPs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B12 | `12` | 12/12 | 84.9M | 151.0M | s0: 0.85060, s1: 0.85094, s2: 0.85108 | 0.85087 | 0.00025 | 0.1488 | 0/3 | 266,464 | 28,771 | 1.172e+18 |
| B8 | `8` | 8/8 | 56.6M | 100.7M | s0: 0.87034, s1: 0.86979, s2: 0.86921 | 0.86978 | 0.00057 | 0.1373 | 0/3 | 367,170 | 20,875 | 8.479e+17 |
| B6 | `6` | 6/6 | 42.5M | 75.5M | s0: 0.88592, s1: 0.88614, s2: 0.88525 | 0.88577 | 0.00046 | 0.1287 | 0/3 | 452,465 | 16,924 | 6.858e+17 |
| L8 | `2,4x2,2` | 8/12 | 56.6M | 100.7M | s0: 0.86001, s1: 0.85901, s2: 0.86005 | 0.85969 | 0.00059 | 0.1493 | 0/3 | 268,374 | 28,189 | 1.172e+18 |
| L6s | `2,2x4,2` | 6/12 | 42.5M | 75.5M | s0: 0.87129, s1: 0.87066, s2: 0.87057 | 0.87084 | 0.00039 | 0.1396 | 0/3 | 269,535 | 27,902 | 1.172e+18 |
| L6p | `0,6x2,0` | 6/12 | 42.5M | 75.5M | s0: 0.86979, s1: 0.87083, s2: 0.87116 | 0.87059 | 0.00071 | 0.1479 | 0/3 | 269,654 | 27,906 | 1.172e+18 |
| LR | `2,4x2,2` | 8/8-20 | 56.6M | 100.7M | s0: 0.86493, s1: 0.86523, s2: 0.86529 | 0.86515 | 0.00019 | 0.1385 | 0/3 | 265,285 | 42,827 | 1.329e+18 |

## Hypotheses (decision rule of section 3.4)

**H1, looping helps at equal parameters** (needs L8 < B8 and L6* < B6):
- L8 vs B8: **SUPPORTED: L8 beats B8**. mean(B8) - mean(L8) = +0.01009, 2 x pooled SD = 0.00115, all seeds of L8 beat all seeds of B8
- L6s vs B6: **SUPPORTED: L6s beats B6**. mean(B6) - mean(L6s) = +0.01493, 2 x pooled SD = 0.00086, all seeds of L6s beat all seeds of B6
- L6p vs B6: **SUPPORTED: L6p beats B6**. mean(B6) - mean(L6p) = +0.01518, 2 x pooled SD = 0.00120, all seeds of L6p beat all seeds of B6

**H2, sharing has a cost at equal compute** (needs B12 < L8; prediction 0.3 <= rho(L8) <= 0.8):
- B12 vs L8: **SUPPORTED: B12 beats L8**. mean(L8) - mean(B12) = +0.00882, 2 x pooled SD = 0.00090, all seeds of B12 beat all seeds of L8
- rho(L8) = 0.534 (over 27 seed combinations: min 0.492, median 0.523, max 0.588). Inside the predicted range.
- rho(L6s) = 0.428 (over 27 seed combinations: min 0.403, median 0.428, max 0.444).
- rho(L6p) = 0.435 (over 27 seed combinations: min 0.407, median 0.431, max 0.466).

**H3, layout at fixed parameters and compute** (weak prediction: sandwich `2,2x4,2` not worse than pure loop `0,6x2,0`):
- L6s vs L6p: **INCONCLUSIVE**. mean(L6p) - mean(L6s) = -0.00024, 2 x pooled SD = 0.00115

**H4, test-time loops** (LR arm: val_bpb decreases monotonically over R = 1..4, and R in {5,6,8} is no worse than R = 4 by more than 0.005):
- LR seed 0: monotone over 1..4: **False** (0.88097, 0.86494, 0.86249, 0.86281); beyond the trained range vs R=4: R=5: +0.00111, R=6: +0.00204, R=8: +0.00375 => within 0.005: **True**; improves beyond R=4: False
- LR seed 1: monotone over 1..4: **False** (0.88116, 0.86524, 0.86273, 0.86297); beyond the trained range vs R=4: R=5: +0.00088, R=6: +0.00168, R=8: +0.00304 => within 0.005: **True**; improves beyond R=4: False
- LR seed 2: monotone over 1..4: **False** (0.88105, 0.86530, 0.86278, 0.86315); beyond the trained range vs R=4: R=5: +0.00104, R=6: +0.00205, R=8: +0.00379 => within 0.005: **True**; improves beyond R=4: False

Note for H4: the backout tap is the residual after effective layer E // 2, so it moves with R (spec 9.3). The curve varies the backout point together with the loop count.

## Figures

- `results/figures/F1_bpb_vs_params.png`
- `results/figures/F2_bpb_vs_flops.png`
- `results/figures/F3_recovery_fraction.png`
- `results/figures/F4_bpb_vs_inference_loops.png`
- `results/figures/F5_training_curves.png`
- `results/figures/F6_residual_rms.png`
