"""Matched 30- vs 60-cycle comparison on identical endpoints (same seed, engine, endpoint cycle) for IID, DS-A and DS-B."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
SRC = {"IID": ("FD004_IID_Rerun_Results", "FD004_IID_W60_Results"),
       "DS-A": ("FD004_DS_A_Rerun_Results", "FD004_DS_A_W60_Results"),
       "DS-B": ("FD004_DS_B_Rerun_Results", "FD004_DS_B_W60_Results")}


def load(path, cond, w):
    d = pd.read_csv(ROOT / path / "test_predictions_rowlevel.csv")
    d = d.rename(columns={"engine_id": "engine", "asym_lower": "a_lo", "asym_upper": "a_hi", "bal_lower": "b_lo", "bal_upper": "b_hi",
                          "asymmetric_lower": "a_lo", "asymmetric_upper": "a_hi", "balanced_lower": "b_lo", "balanced_upper": "b_hi"})
    if cond == "IID" and w == 30:
        d["cycle"] = d["cycle"] + 30          # 30-cycle IID rows store the window index; first endpoint is cycle 30
    d["err"] = d.q50 - d.true_RUL
    d["a_cov"] = (d.true_RUL >= d.a_lo) & (d.true_RUL <= d.a_hi)
    d["b_cov"] = (d.true_RUL >= d.b_lo) & (d.true_RUL <= d.b_hi)
    d["b_w"] = d.b_hi - d.b_lo
    return d[["seed", "engine", "cycle", "true_RUL", "stage", "err", "a_cov", "b_cov", "b_w"]]


rmse = lambda e: float(np.sqrt(np.mean(np.square(e))))


def metrics(g):
    e = g.groupby("engine").b_cov.mean() * 100
    mid = g[g.stage == "Mid"]
    return {"N": len(g), "RMSE": rmse(g.err), "Mid_RMSE": rmse(mid.err), "Mid_bias": float(mid.err.mean()),
            "Asym_PICP": 100 * g.a_cov.mean(), "Bal_PICP": 100 * g.b_cov.mean(), "Bal_MPIW": float(g.b_w.mean()),
            "Bal_Mid_PICP": 100 * mid.b_cov.mean(), "Bal_eng_mean": e.mean(), "Bal_eng_p10": np.percentile(e, 10),
            "Bal_eng_worst": e.min(), "Bal_below80": int((e < 80).sum())}


rows, diffs, boots, engines = [], [], [], []
for cond, (p30, p60) in SRC.items():
    a, b = load(p30, cond, 30), load(p60, cond, 60)
    m = a.merge(b, on=["seed", "engine", "cycle"], suffixes=("_30", "_60"))
    assert (m.true_RUL_30 == m.true_RUL_60).all()
    assert len(m) == len(b), f"{cond}: {len(b) - len(m)} 60-cycle endpoints have no 30-cycle partner"
    unmatched30 = len(a) - len(m)
    for seed, g in m.groupby("seed"):
        for w in (30, 60):
            gw = pd.DataFrame({"engine": g.engine, "stage": g.stage_30, "err": g[f"err_{w}"], "a_cov": g[f"a_cov_{w}"],
                               "b_cov": g[f"b_cov_{w}"], "b_w": g[f"b_w_{w}"]})
            r = metrics(gw); r.update(condition=cond, seed=seed, window=w, unmatched_30cycle_endpoints=unmatched30)
            rows.append(r)
            ec = gw.groupby("engine").b_cov.mean() * 100
            engines += [{"condition": cond, "seed": seed, "window": w, "engine": k, "bal_cov": v} for k, v in ec.items()]
    res = pd.DataFrame([r for r in rows if r["condition"] == cond])
    for seed in res.seed.unique():
        x30 = res[(res.seed == seed) & (res.window == 30)].iloc[0]; x60 = res[(res.seed == seed) & (res.window == 60)].iloc[0]
        d = {k: x60[k] - x30[k] for k in ["RMSE", "Mid_RMSE", "Mid_bias", "Asym_PICP", "Bal_PICP", "Bal_MPIW", "Bal_Mid_PICP",
                                           "Bal_eng_mean", "Bal_eng_p10", "Bal_eng_worst", "Bal_below80"]}
        d.update(condition=cond, seed=seed); diffs.append(d)
    # paired engine-cluster bootstrap, seed 42
    g = m[m.seed == 42].reset_index(drop=True)
    engs = np.unique(g.engine); by = {e: np.where(g.engine.values == e)[0] for e in engs}
    rng = np.random.default_rng(42); sims = []
    for _ in range(10000):
        idx = np.concatenate([by[e] for e in rng.choice(engs, len(engs))])
        mid = idx[g.stage_30.values[idx] == "Mid"]
        sims.append((rmse(g.err_60.values[idx]) - rmse(g.err_30.values[idx]),
                     rmse(g.err_60.values[mid]) - rmse(g.err_30.values[mid]),
                     100 * (g.b_cov_60.values[idx].mean() - g.b_cov_30.values[idx].mean()),
                     100 * (g.b_cov_60.values[mid].mean() - g.b_cov_30.values[mid].mean()),
                     g.b_w_60.values[idx].mean() - g.b_w_30.values[idx].mean()))
    sims = np.array(sims)
    mid = g.stage_30 == "Mid"
    pts = [rmse(g.err_60) - rmse(g.err_30), rmse(g.err_60[mid]) - rmse(g.err_30[mid]),
           100 * (g.b_cov_60.mean() - g.b_cov_30.mean()), 100 * (g.b_cov_60[mid].mean() - g.b_cov_30[mid].mean()),
           g.b_w_60.mean() - g.b_w_30.mean()]
    for j, name in enumerate(["RMSE", "Mid_RMSE", "Bal_PICP (pp)", "Bal_Mid_PICP (pp)", "Bal_MPIW"]):
        boots.append({"condition": cond, "stat": f"{name}: 60 - 30", "point": pts[j],
                      "ci_low": np.percentile(sims[:, j], 2.5), "ci_high": np.percentile(sims[:, j], 97.5)})

res = pd.DataFrame(rows); dif = pd.DataFrame(diffs); boot = pd.DataFrame(boots); eng = pd.DataFrame(engines)
res.to_csv(OUT / "matched_metrics_by_seed.csv", index=False)
dif.to_csv(OUT / "matched_differences_by_seed.csv", index=False)
boot.to_csv(OUT / "matched_bootstrap_seed42.csv", index=False)
eng.to_csv(OUT / "matched_engine_coverage.csv", index=False)
cols = ["N", "RMSE", "Mid_RMSE", "Mid_bias", "Asym_PICP", "Bal_PICP", "Bal_MPIW", "Bal_Mid_PICP", "Bal_eng_p10", "Bal_eng_worst", "Bal_below80"]
summ = res.groupby(["condition", "window"])[cols].agg(["mean", "std"]); summ.to_csv(OUT / "matched_five_seed_summary.csv")
pd.set_option("display.width", 250)
print(res[res.seed == 42].set_index(["condition", "window"])[cols].round(2).to_string())
print(dif.groupby("condition").agg(["mean", "std"]).round(2).T.to_string())
print(boot.round(2).to_string(index=False))
# persistent engines below 80% in all five seeds, per condition and window
pers = eng[eng.bal_cov < 80].groupby(["condition", "window", "engine"]).size().reset_index(name="seeds_below80")
print(pers[pers.seeds_below80 == 5].groupby(["condition", "window"]).engine.apply(list).to_string())
pers.to_csv(OUT / "matched_problem_engine_counts.csv", index=False)
