# Paper 2 extension: architecture factorial and context sweep

`matched_factorial_sweep.py` (run as `python matched_factorial_sweep.py <root> <this folder>`) compares models on
identical endpoints (same seed, engine, endpoint cycle), from each run's `test_predictions_rowlevel.csv`:

- factorial: BiLSTM/LSTM x with/without attention x 30/60 cycles x IID/DS-A, on the endpoints a 60-cycle window can
  score (IID 4,703, DS-A 1,753 per seed) -> `factorial_matched_*.csv` (draft v2 Tables 11-12);
- sweep: Part I model at 30/45/60/90 cycles x IID/DS-A/DS-B, on the endpoints a 90-cycle window can score
  (IID 3,983, DS-A 1,471, DS-B 1,170) -> `sweep_matched_*.csv` (Table 13).

Summaries are given for all seeds and with collapsed seeds excluded (loss ratio > 0.95: the model predicts the
constant quantiles): notebooks 22 and 26 seed 1000, 28 and 29 seed 123. Paired differences drop any pair with a
collapsed side.

`factorial_sweep_five_seed.csv` (from `factorial_sweep.py`) is the earlier unmatched full-test-set summary, kept
for the record.
