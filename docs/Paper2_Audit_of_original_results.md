# Paper 2: audit and re-anchoring plan

2026-10-19. Paper 2 question: does a longer input context (60 cycles, P2-E8) or degradation-state
information reduce the mid-life error that Paper 1 identified, and does that improve engine-level
conformal reliability under the same two shifts?

## What Paper 2 currently compares

| Paper 2 claim (manuscript and `Paper2_final_headline_summary.csv`) | Paper 1 reference used | Updated Paper 1 value |
|---|---|---|
| DS-A point RMSE 24.869 → 16.942 | archived DS-A run (not reproducible) | 24.10 (seed 42), 25.00 ± 1.72 |
| DS-B point RMSE 23.999 → 17.130 | archived DS-B run (model overwritten) | 24.88 (seed 42), 25.20 ± 0.53 |
| DS-A Balanced coverage 90.21 → 85.05 | archived | 91.04, 89.86 ± 1.30 |
| DS-B Balanced coverage 90.63 → 89.21 | archived | 91.81, 90.38 ± 2.54 |
| DS-A Balanced mid-life coverage 70.88 → 81.56 | archived | 77.06, 75.52 ± 6.36 |
| DS-B Balanced mid-life coverage 75.16 → 86.35 | archived | 77.36, 76.60 ± 8.13 |
| "persistent nine-engine subgroup" (17, 105, 120, 156, 181, 184, 194, 240, 245) | archived | five engines: 17, 156, 184, 194, 240 |
| IID matched RMSE 19.116 → 17.432 (P2-E1 vs P2-E8) | P2-E1, a point LSTM | (development; see below) |

Every Paper 1 number Paper 2 cites has changed, and several of its conclusions depend on them
(for example "mid-life coverage improved from 70.88% to 81.56%" against a reference that is now 77.06%).

## Problems found in the Paper 2 pipeline

1. **Calibration engines used for model selection (shift runs).** `Shift_V1_DS_A` and `Shift_V2_DS_B`
   train P2-E8 with early stopping and checkpointing on `validation_data=(X_cal_ds, y_cal_ds)`, the
   calibration engines. Paper 1's rule is that calibration engines never touch model fitting.
2. **Development numbers are scored on the selection set.** P2-E1 to P2-E8 stop early and checkpoint
   on the 49 IID validation engines and then report RMSE on the same engines. Rankings between the
   experiments are still informative, but the reported RMSEs are optimistic.
3. **Unmatched comparisons.** 60-cycle windows need 60 cycles of history, so P2-E8 is scored on fewer
   endpoints than Paper 1 (DS-A 1,753 vs 2,043; DS-B 1,408 vs 1,623). The dropped endpoints are early
   in life. The shift headline (24.87 → 16.94) compares different window sets. Only the IID
   development comparison (8,862 endpoints) is matched.
4. **Different model families in one comparison.** The shift point results use a deterministic LSTM
   (MSE). The UQ results use a separately trained quantile LSTM with a fixed 60 epochs and no
   validation (DS-B RMSE 18.997 there vs 17.130 in the point run). Paper 1 uses a BiLSTM-attention
   quantile model. Point and interval claims in Paper 2 come from different models.
5. **Single seed, fixed or validation-chosen budgets.** No five-seed evidence for P2-E8 under shift,
   and the plateau that broke Paper 1's fixed-budget runs was never checked.

## Recommended plan

**Principle:** Paper 2 changes one thing relative to Paper 1, the input context (30 → 60 cycles),
and keeps everything else identical: splits, preprocessing, the BiLSTM-attention quantile model,
the pre-registered training rule, the five seeds, every calibration method and every engine-level
metric. Then any difference is attributable to the context length.

1. **Main experiment (new notebooks 09–11):** the Paper 1 rerun notebooks with `WINDOW = 60`, for
   IID, DS-A and DS-B, five seeds each, same protocol file and convergence check. Compute is about
   2× Paper 1's (longer sequences): roughly 3–4 hours here for all three, run in the background.
2. **Matched evaluation:** compare 30-cycle and 60-cycle models only on endpoints that both can
   score (endpoint cycle ≥ 60), using each model's own calibration. Report unmatched numbers as
   secondary. Paper 1's saved row-level predictions make this possible without retraining.
3. **Engine-level and mid-life focus:** the same tables as Paper 1 (engine-level coverage, persistent
   engines, life stage, bootstrap, five-seed summary) for both context lengths, side by side, with
   paired bootstrap differences (60 − 30) on matched endpoints.
4. **Development evidence (P2-E1 to P2-E8):** keep as the mechanism study that motivated 60 cycles,
   reported like Paper 1's Table 1 (development, selection-biased, disclosed). Optional: rerun only
   P2-E1 vs P2-E8 under the pre-registered rule on the IID split for a clean confirmation.
5. **Then rewrite the Paper 2 manuscript** around the matched results.

## Decisions for you

- Approve the plan above (main experiment = 60-cycle version of the Paper 1 pipeline), or keep
  Paper 2's deterministic LSTM as its model and only fix the leakage and matching.
- Whether to rerun the P2-E1 vs P2-E8 development comparison (step 4, optional, about 1 hour).
