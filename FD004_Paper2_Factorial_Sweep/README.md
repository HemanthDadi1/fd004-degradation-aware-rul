# Paper 2 extension: architecture factorial and context sweep (preliminary)

`factorial_sweep_five_seed.csv` is built by `factorial_sweep.py` from:
- local runs: `global_by_seed.csv` (notebooks 06–11, 19–23);
- Colab runs (24–36): the five-seed summary printed in the executed notebooks (per-seed files are still on Google Drive).

**Preliminary**: these are full-test-set means. 30/45/60/90-cycle models score different endpoint sets (a W-cycle window
needs W cycles of history), so window-length differences are not yet matched comparisons. Colab rows with a collapsed seed
(ratio ≈ 1.00: the model predicts the constant quantiles) include that seed; collapsed seeds:
22 DS-A 60 lstm_attn s1000, 26 DS-A 60 bilstm_noattn s1000, 28 DS-A 30 lstm_noattn s123, 29 IID 60 lstm_noattn s123.

To finish: copy the `FD004_*_Results` folders for 24–36 from Drive, then rerun matched endpoints, bootstrap and the
collapsed-excluded summaries.
