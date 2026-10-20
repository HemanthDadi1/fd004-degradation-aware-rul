"""Five-seed ensembles at 30 and 60 cycles: average quantiles, calibrate, compare on matched endpoints (Paper 2 extension)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
A = 0.20
SRC = {("IID", 30): "FD004_IID_Rerun_Results", ("DS-A", 30): "FD004_DS_A_Rerun_Results", ("DS-B", 30): "FD004_DS_B_Rerun_Results",
       ("IID", 60): "FD004_IID_W60_Results", ("DS-A", 60): "FD004_DS_A_W60_Results", ("DS-B", 60): "FD004_DS_B_W60_Results"}
SEEDS = [42, 123, 456, 789, 1000]
cq = lambda s, lvl: float(np.quantile(s, min(1.0, np.ceil((len(s) + 1) * lvl) / len(s)), method="higher"))
rmse = lambda e: float(np.sqrt(np.mean(np.square(e))))


def ensemble(cond, w):
    d = ROOT / SRC[(cond, w)]
    cal = [pd.read_csv(d / f"calibration_predictions_seed{s}.csv").rename(columns={"engine_id": "engine"}) for s in SEEDS]
    t = pd.read_csv(d / "test_predictions_rowlevel.csv").rename(columns={"engine_id": "engine"})
    tests = [t[t.seed == s].reset_index(drop=True) for s in SEEDS]
    for c in cal[1:]:
        assert np.array_equal(c.true_RUL.values, cal[0].true_RUL.values) and np.array_equal(c.engine.values, cal[0].engine.values)
    for x in tests[1:]:
        assert np.array_equal(x.true_RUL.values, tests[0].true_RUL.values) and np.array_equal(x.engine.values, tests[0].engine.values)
    C = cal[0][["engine", "true_RUL"]].copy(); T = tests[0][["engine", "cycle", "true_RUL", "stage"]].copy()
    for q in ("q10", "q50", "q90"):
        C[q] = np.mean([c[q].values for c in cal], axis=0); T[q] = np.mean([x[q].values for x in tests], axis=0)
    if cond == "IID" and w == 30:
        T["cycle"] = T["cycle"] + 30
    lc, uc, yc, ec = C.q10.values, C.q90.values, C.true_RUL.values, C.engine.values
    ql, qu = cq(lc - yc, 1 - A / 2), cq(yc - uc, 1 - A / 2)
    s2 = np.maximum(lc - yc, yc - uc)
    bq = np.array([np.quantile(s2[ec == e], 0.80) for e in np.unique(ec)]); qb = cq(bq, 1 - A)
    T["a_lo"], T["a_hi"] = np.maximum(0, T.q10 - ql), np.maximum(0, T.q90 + qu)
    T["b_lo"], T["b_hi"] = np.maximum(0, T.q10 - qb), np.maximum(0, T.q90 + qb)
    T["err"] = T.q50 - T.true_RUL
    T["a_cov"] = (T.true_RUL >= T.a_lo) & (T.true_RUL <= T.a_hi); T["b_cov"] = (T.true_RUL >= T.b_lo) & (T.true_RUL <= T.b_hi)
    T["b_w"] = T.b_hi - T.b_lo
    return T


def metrics(T):
    e = T.groupby("engine").b_cov.mean() * 100; mid = T[T.stage == "Mid"]
    return {"N": len(T), "RMSE": rmse(T.err), "Mid_RMSE": rmse(mid.err), "Asym_PICP": 100 * T.a_cov.mean(), "Bal_PICP": 100 * T.b_cov.mean(),
            "Bal_MPIW": float(T.b_w.mean()), "Bal_Mid_PICP": 100 * mid.b_cov.mean(), "Bal_below80": int((e < 80).sum()),
            "Bal_eng_worst": float(e.min())}


rows, boots = [], []
for cond in ("IID", "DS-A", "DS-B"):
    e30, e60 = ensemble(cond, 30), ensemble(cond, 60)
    rows.append({"condition": cond, "window": 30, "set": "full test set", **metrics(e30)})
    rows.append({"condition": cond, "window": 60, "set": "full test set", **metrics(e60)})
    m = e30.merge(e60, on=["engine", "cycle"], suffixes=("_30", "_60"))
    assert len(m) == len(e60)
    pick = lambda w: m[["engine"] + [c for c in m.columns if c.endswith(f"_{w}")]].rename(columns=lambda c: c[:-3] if c.endswith(f"_{w}") else c).assign(stage=m.stage_30)
    m30, m60 = pick(30), pick(60)
    rows.append({"condition": cond, "window": 30, "set": "matched", **metrics(m30)})
    rows.append({"condition": cond, "window": 60, "set": "matched", **metrics(m60)})
    engs = np.unique(m.engine); by = {e: np.where(m.engine.values == e)[0] for e in engs}; rng = np.random.default_rng(42); sims = []
    for _ in range(10000):
        idx = np.concatenate([by[e] for e in rng.choice(engs, len(engs))]); mid = idx[m.stage_30.values[idx] == "Mid"]
        sims.append((rmse(m.err_60.values[idx]) - rmse(m.err_30.values[idx]), rmse(m.err_60.values[mid]) - rmse(m.err_30.values[mid]),
                     100 * (m.b_cov_60.values[mid].mean() - m.b_cov_30.values[mid].mean())))
    sims = np.array(sims); mm = m.stage_30 == "Mid"
    pts = [rmse(m.err_60) - rmse(m.err_30), rmse(m.err_60[mm]) - rmse(m.err_30[mm]), 100 * (m.b_cov_60[mm].mean() - m.b_cov_30[mm].mean())]
    for j, name in enumerate(["RMSE 60 − 30", "Mid RMSE 60 − 30", "Bal mid PICP 60 − 30 (pp)"]):
        boots.append({"condition": cond, "stat": name, "point": pts[j], "ci_low": np.percentile(sims[:, j], 2.5), "ci_high": np.percentile(sims[:, j], 97.5)})
    e30.to_csv(OUT / f"ensemble_{cond}_W30_test.csv", index=False); e60.to_csv(OUT / f"ensemble_{cond}_W60_test.csv", index=False)
res = pd.DataFrame(rows); res.to_csv(OUT / "ensemble_metrics.csv", index=False)
bt = pd.DataFrame(boots); bt.to_csv(OUT / "ensemble_bootstrap_60_minus_30.csv", index=False)
pd.set_option("display.width", 250)
print(res.round(2).to_string(index=False)); print(bt.round(2).to_string(index=False))
