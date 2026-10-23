# FD004 Paper 2: does a longer temporal context improve engine-level RUL reliability?

Code and results for *Engine-Level Reliability of Conformal RUL Uncertainty Under Operating-Condition
Distribution Shift in C-MAPSS FD004 (Part II): Does a Longer Temporal Context Help?*

Part I ([fd004-conformal-rul-reliability](https://github.com/HemanthDadi1/fd004-conformal-rul-reliability))
found that conformal RUL intervals keep pooled coverage under operating-condition shift but still fail for
some engines, mostly in mid-life. Part II changes one thing, the input window (60 cycles instead of 30),
and keeps everything else of the Part I pipeline: splits, preprocessing, BiLSTM-attention quantile model,
pre-specified training rule, calibration methods, metrics and five seeds.

## Main findings

- For a simple single-direction point LSTM, 60 cycles lowers IID error in all five seeds
  (RMSE −3.5 ± 0.7, mid-life −6.1 ± 1.0 cycles).
- For the Part I probabilistic model, the effect on average error is modest and seed-dependent, but under
  shift the number of engines below 80% Balanced Engine CQR coverage falls in 9 of 10 seed pairs (the tenth
  tied), and the set of always-failing engines shrinks. Engines 17 and 156 fail in every shift run.

Extensions (draft v2, Section 6):

- **Seed ensembles**: averaging the five seeds' quantiles makes the 60-cycle gain clear under IID, DS-A and DS-B
  (bootstrap 95% intervals for RMSE exclude zero).
- **Equal width**: at the same mean interval width, the 60-cycle ensemble has fewer or equally many engines below 80%
  in all nine condition-width cells.
- **FD002**: 60 cycles lowers IID and DS-B error in every seed.
- **Architecture factorial** (BiLSTM/LSTM × with/without attention, matched endpoints): without attention the model
  is 2–8 cycles more accurate, far more stable across seeds, and gains from 60 cycles in every converged seed pair;
  the count of under-covered engines barely moves.
- **45/90-cycle sweep** (matched endpoints): error falls with window length; under DS-A the 90-cycle model has fewer
  engines below 80% than the 30-cycle model in all five seeds.

Details: `docs/Paper2_Results_Summary.md`. Why the original Paper 2 numbers were replaced:
`docs/Paper2_Audit_of_original_results.md`.

## Layout

| Path | What it is |
|---|---|
| `notebooks/09_…IID_60cycle`, `10_…DS_A_60cycle`, `11_…DS_B_60cycle` | Part I pipeline with 60-cycle windows, five seeds each (executed) |
| `notebooks/12_…P2E1_vs_P2E8_Rerun` | Point LSTM, 30 vs 60 cycles, Part I training rule, five seeds (executed) |
| `notebooks/16_…`–`18_…` | FD002 with 60-cycle windows (IID, DS-A, DS-B) |
| `notebooks/19_…`–`30_…` | Architecture factorial: LSTM + attention, BiLSTM and LSTM without attention, at 30 and 60 cycles, IID and DS-A |
| `notebooks/31_…`–`36_…` | Part I model with 45- and 90-cycle windows, IID, DS-A and DS-B |
| `notebooks/archived_original_paper2/` | The original Paper 2 notebooks, kept for the record (superseded) |
| `FD004_*_W60_Results/` | Protocol, models, histories, convergence checks, row-level predictions, tables, SHA-256 manifest |
| `FD004_P2E1_vs_P2E8_Results/` | Same for notebook 12 |
| `FD004_Paper2_Matched_30v60/` | Matched 30 vs 60 comparison: per-seed metrics, paired differences, bootstrap, life stage, persistent engines |
| `scripts/matched_compare.py`, `scripts/p2figs.py` | Matched comparison and the paper's Figures 1–2 |
| `FD004_FD002_*_W60_Results/`, `FD004_*_{lstm_attn,bilstm_noattn,lstm_noattn}_Results/`, `FD004_*_W45_Results/`, `FD004_*_W90_Results/` | Full outputs of notebooks 16–36 |
| `FD004_FD002_Matched_30v60/` | FD002 matched 30 vs 60 comparison (30-cycle side from the Part I repository) |
| `FD004_Paper2_Ensembles/`, `FD004_Paper2_EqualWidth/` | Five-seed ensembles and equal-width curves; `figures/p2_fig3_equal_width.png` |
| `FD004_Paper2_Factorial_Sweep/` | Matched factorial and sweep tables, all seeds and collapsed seeds excluded (see README there) |
| `scripts/ensembles.py`, `equal_width_30v60.py`, `matched_compare_fd002.py`, `p2fig3.py`, `matched_factorial_sweep.py` | Analyses behind the extension tables and Figure 3 |
| `Supplementary_Data_S2.xlsx`, `scripts/build_s2.py` | Supplementary Data S2: convergence of all 120 Part II runs, FD002 splits, per-seed matched tables, ensembles, equal-width curves |
| `scripts/make_gen_nb.py` | Generates notebooks 16–36 (condition, window, dataset, architecture) |
| `colab/RUN_ON_COLAB.ipynb` | Runs notebooks on a Colab GPU, resumably (notebooks 24–36 ran this way) |
| `artifacts/`, `FD004_DS_B_Results/`, `FD004_DS_A_Rerun_Results/`, `models/` | Inputs the notebooks check against |

## Reproduce

1. Get the data (`DATA.md`) and create an environment from `requirements.txt` (Python 3.12, TensorFlow 2.20).
2. Run the notebooks with `FD004_ROOT` set to this folder:

```bash
export FD004_ROOT="$PWD"
jupyter nbconvert --to notebook --execute notebooks/10_FD004_Paper2_DS_A_60cycle_FiveSeed.ipynb
```

   `FD004_SMOKE=1` gives a one-seed, three-epoch pipeline check written to a separate folder.
3. The matched comparison needs the 30-cycle Part I results. Copy `FD004_IID_Rerun_Results`,
   `FD004_DS_A_Rerun_Results` and `FD004_DS_B_Rerun_Results` from the Part I repository into this folder, then:

```bash
python scripts/matched_compare.py . FD004_Paper2_Matched_30v60
```

4. Extensions, from this folder (with the Part I rerun and FD002 30-cycle folders copied in as above):

```bash
python scripts/ensembles.py . FD004_Paper2_Ensembles
python scripts/equal_width_30v60.py . FD004_Paper2_EqualWidth
python scripts/matched_compare_fd002.py . FD004_FD002_Matched_30v60
python scripts/matched_factorial_sweep.py . FD004_Paper2_Factorial_Sweep
```

## Early stopping note

The notebooks use Keras `EarlyStopping(start_from_epoch=30, restore_best_weights=True)`, which ignores epochs
1–30, so each saved model holds the best weights from epoch 31 on. Convergence is judged on that restored
epoch in `convergence_check_restored_epoch.csv`; `convergence_check.csv` used the minimum over all epochs.
One run (IID 60-cycle, seed 123, ratio 0.61) misses the 0.60 threshold and is reported, not replaced. In the
extensions, five runs collapsed to the constant-quantile solution (ratio 1.00): notebooks 17 (seed 789), 22 and 26
(seed 1000), 28 and 29 (seed 123). They are reported, not replaced.

## Citation and license

See `CITATION.cff`. Archive DOI: [add after the Zenodo release]. Code: MIT. The C-MAPSS data are NASA's and are
not included.
