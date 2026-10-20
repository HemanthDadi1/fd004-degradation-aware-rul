# Paper 2: results of the 60-cycle rerun (matched, five seeds)

2026-10-20. Question: does a 60-cycle input context (instead of 30) reduce the mid-life error that
Paper 1 identified, and does that improve engine-level conformal reliability under shift?

## Design

- Only the window length changes. Everything else is the Paper 1 pipeline: splits, preprocessing,
  BiLSTM-attention quantile model, pre-registered training rule (20% internal validation, early
  stopping, convergence check), calibration methods, metrics, seeds 42, 123, 456, 789, 1000.
- Notebooks 09 (IID), 10 (DS-A), 11 (DS-B) in `Notebooks/Paper 2`; results in `FD004_*_W60_Results/`.
- Convergence: 14 of 15 runs converged; IID seed 123 narrowly failed (0.61 vs 0.60, judged on the restored epoch) and is
  reported and flagged, not replaced.
- **Matched comparison:** 30- and 60-cycle models are compared only on endpoints both can score
  (same seed, engine, endpoint cycle ≥ 60): IID 4,703, DS-A 1,753, DS-B 1,408 windows per seed.
  Each model keeps its own calibration. Results: `FD004_Paper2_Matched_30v60/`.

## Matched results, five-seed mean ± SD

| | IID 30 | IID 60 | DS-A 30 | DS-A 60 | DS-B 30 | DS-B 60 |
|---|---|---|---|---|---|---|
| RMSE (cycles) | 23.81 ± 1.14 | 21.21 ± 3.66 | 26.49 ± 1.91 | 25.31 ± 4.47 | 26.60 ± 0.58 | 22.65 ± 4.74 |
| Mid-life RMSE | 33.15 ± 2.21 | 30.02 ± 4.82 | 39.05 ± 3.17 | 35.91 ± 6.64 | 40.75 ± 2.21 | 32.91 ± 4.51 |
| Mid-life bias (pred − true) | +8.06 ± 1.05 | +6.34 ± 2.89 | +17.42 ± 2.76 | +16.25 ± 6.80 | +19.82 ± 3.19 | +14.28 ± 1.59 |
| Asymmetric CQR coverage (%) | 80.88 ± 1.70 | 82.74 ± 1.38 | 71.91 ± 2.70 | 75.00 ± 1.55 | 70.99 ± 2.17 | 74.26 ± 2.09 |
| Balanced coverage (%) | 92.00 ± 0.90 | 92.56 ± 1.91 | 88.33 ± 1.52 | 90.70 ± 1.13 | 89.01 ± 2.80 | 89.50 ± 1.56 |
| Balanced width (cycles) | 78.50 ± 7.63 | 74.39 ± 17.90 | 70.15 ± 8.52 | 77.43 ± 13.86 | 75.89 ± 6.66 | 69.71 ± 17.39 |
| Balanced mid-life coverage (%) | 85.93 ± 3.06 | 86.96 ± 6.48 | 75.58 ± 6.18 | 86.03 ± 7.29 | 76.44 ± 8.18 | 79.87 ± 10.39 |
| Engines below 80% (Balanced) | 3.6 ± 1.1 | 3.6 ± 1.7 | 11.2 ± 1.8 | 8.4 ± 1.5 | 10.6 ± 3.4 | 8.4 ± 2.1 |
| Worst engine (Balanced, %) | 69.66 ± 1.95 | 63.16 ± 6.43 | 29.06 ± 5.41 | 43.92 ± 5.54 | 32.12 ± 9.81 | 38.48 ± 11.22 |

(30-cycle values here are restricted to the matched endpoints, so they differ from Paper 1's
full-test-set numbers.)

## What is consistent across seeds (paired 60 − 30, per seed)

- **Fewer under-covered engines under shift.** Engines below 80% fall in 9 of 10 shift seed pairs
  and never rise (DS-A −1, −2, −4, −4, −3; DS-B −1, 0, −2, −5, −3). Sign test over the 9 non-tied
  pairs: p ≈ 0.004. No effect under IID (+1, 0, 0, +1, −2).
- **DS-A mid-life coverage rises in all five seeds** (+3.6 to +20.3 points; mean +10.4).
- **Asymmetric CQR coverage rises** in 12 of 15 seed pairs (mean +1.9 to +3.3 points).

## What is not consistent

- **Point accuracy.** Mean RMSE falls (−2.6 IID, −1.2 DS-A, −4.0 DS-B), but per-seed changes go
  both ways (DS-A: +3.6, +6.6, −4.5, −9.3, −2.3). DS-B is the most consistent (4 of 5 lower).
- **Interval width.** 60-cycle widths vary much more across seeds (SD 14–18 cycles vs 7–9).
- **Seed 42 alone is misleading:** under DS-A, 60 cycles makes seed 42 worse (RMSE +3.6, width
  +24 cycles); under DS-B, much better (RMSE −6.6).

## Persistent engines (below 80% Balanced coverage in all five seeds)

| | 30 cycles | 60 cycles |
|---|---|---|
| IID | 36 | none (36, 49 in four seeds) |
| DS-A | 17, 105, 115, 156, 184, 240 | 17, 156, 240 |
| DS-B | 17, 39, 156, 184, 240 | 17, 156, 184 |

Engines 17 and 156 fail in every shift run at both context lengths.

## Comparison with the original Paper 2 claims

| Original claim | Clean rerun |
|---|---|
| DS-A RMSE 24.87 → 16.94 (−7.9) | 26.49 → 25.31 (−1.2 ± 6.3), matched |
| DS-B RMSE 24.00 → 17.13 (−6.9) | 26.60 → 22.65 (−4.0 ± 4.3), matched |
| DS-A mid-life Balanced coverage 70.88 → 81.56 | 75.58 → 86.03 (+10.4 ± 8.0), consistent sign |
| DS-B mid-life Balanced coverage 75.16 → 86.35 | 76.44 → 79.87 (+3.4 ± 9.4), not consistent |
| Persistent nine-engine subgroup unresolved | Core shrinks to three engines under each shift; 17 and 156 remain |

The original numbers came from a different model (deterministic LSTM), unmatched endpoints, and
early stopping on calibration engines (see `Paper2_Audit_and_Plan.md`).

## Defensible headline for Paper 2

A longer input context does not reliably improve point accuracy, but under operating-condition shift
it consistently reduces the number of engines whose conformal intervals under-cover, shrinks the
persistent failure core, and (under the held-out regime) raises mid-life coverage. The gain is in
engine-level reliability, not average error, and it is seed-dependent in size, so five-seed paired
reporting is required.

## P2-E1 vs P2-E8 clean rerun (notebook 12)

Point LSTM (Paper 2's original development architecture), Paper 1 training rule, five seeds, 8,862 matched IID
validation endpoints; all 10 runs converged. 60 − 30: RMSE −3.51 ± 0.71 cycles (lower in 5/5 seeds), mid-life RMSE
−6.09 ± 1.03 (5/5), late −3.46 ± 0.94 (5/5); seed-42 bootstrap −3.94 [−6.21, −1.78] and −5.07 [−8.23, −1.85].
For this simpler model the longer context is a consistent improvement, larger than the archived single-seed
study reported (−1.68 and −3.92). For the Part I BiLSTM-attention quantile model the gain is smaller and
seed-dependent, so the benefit of context length depends on the architecture.

Note on early stopping: Keras `start_from_epoch=30` restores the best weights from epoch 31 on; convergence is
judged on that restored epoch (`convergence_check_restored_epoch.csv`).
