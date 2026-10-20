"""Equal-width comparison of 30- vs 60-cycle models: Window CQR level curves on matched endpoints, per seed and for the ensemble."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
SRC = {("IID", 30): "FD004_IID_Rerun_Results", ("DS-A", 30): "FD004_DS_A_Rerun_Results", ("DS-B", 30): "FD004_DS_B_Rerun_Results",
       ("IID", 60): "FD004_IID_W60_Results", ("DS-A", 60): "FD004_DS_A_W60_Results", ("DS-B", 60): "FD004_DS_B_W60_Results"}
SEEDS = [42, 123, 456, 789, 1000]
LEVELS = np.round(np.arange(0.80, 0.985, 0.005), 3)
cq = lambda s, lvl: float(np.quantile(s, min(1.0, np.ceil((len(s) + 1) * lvl) / len(s)), method="higher"))


def load(cond, w, seed):
    d = ROOT / SRC[(cond, w)]
    cal = pd.read_csv(d / f"calibration_predictions_seed{seed}.csv")
    t = pd.read_csv(d / "test_predictions_rowlevel.csv").rename(columns={"engine_id": "engine"})
    t = t[t.seed == seed][["engine", "cycle", "true_RUL", "stage", "q10", "q90"]].reset_index(drop=True)
    if cond == "IID" and w == 30:
        t["cycle"] = t["cycle"] + 30
    return cal, t


def curve(cal, t, keys):
    s2 = np.maximum(cal.q10 - cal.true_RUL, cal.true_RUL - cal.q90).values
    t = t.merge(keys, on=["engine", "cycle"])  # matched endpoints only
    out = []
    for lvl in LEVELS:
        d = cq(s2, lvl); lo, hi = np.maximum(0, t.q10 - d), np.maximum(0, t.q90 + d)
        cov = (t.true_RUL >= lo) & (t.true_RUL <= hi); e = cov.groupby(t.engine).mean() * 100
        out.append({"level": lvl, "MPIW": float((hi - lo).mean()), "below80": int((e < 80).sum()), "PICP": 100 * cov.mean(),
                    "mid_PICP": 100 * cov[t.stage == "Mid"].mean()})
    return pd.DataFrame(out)


rows, curves = [], []
for cond in ("IID", "DS-A", "DS-B"):
    keys = load(cond, 60, 42)[1][["engine", "cycle"]]
    per = {}
    for w in (30, 60):
        cals, tests = [], []
        for seed in SEEDS:
            cal, t = load(cond, w, seed); cals.append(cal); tests.append(t)
            c = curve(cal, t, keys).assign(condition=cond, window=w, model=f"seed {seed}"); curves.append(c); per[(w, seed)] = c
        ce = cals[0].copy(); te = tests[0].copy()
        for q in ("q10", "q90"):
            ce[q] = np.mean([c_[q].values for c_ in cals], axis=0); te[q] = np.mean([t_[q].values for t_ in tests], axis=0)
        c = curve(ce, te, keys).assign(condition=cond, window=w, model="ensemble"); curves.append(c); per[(w, "ens")] = c
    # Engines below 80% at equal width: interpolate each 60-cycle curve at the widths the 30-cycle curve reaches.
    for model in SEEDS + ["ens"]:
        c30, c60 = per[(30, model)], per[(60, model)]
        for target in (60.0, 70.0, 80.0):
            if c30.MPIW.min() <= target <= c30.MPIW.max() and c60.MPIW.min() <= target <= c60.MPIW.max():
                b30 = float(np.interp(target, c30.MPIW, c30.below80)); b60 = float(np.interp(target, c60.MPIW, c60.below80))
                m30 = float(np.interp(target, c30.MPIW, c30.mid_PICP)); m60 = float(np.interp(target, c60.MPIW, c60.mid_PICP))
                rows.append({"condition": cond, "model": "ensemble" if model == "ens" else f"seed {model}", "width": target,
                             "below80_30": b30, "below80_60": b60, "diff": b60 - b30, "midPICP_30": m30, "midPICP_60": m60})
pd.concat(curves).to_csv(OUT / "window_cqr_level_curves.csv", index=False)
r = pd.DataFrame(rows); r.to_csv(OUT / "equal_width_30v60.csv", index=False)
pd.set_option("display.width", 220)
seeds = r[r.model != "ensemble"].groupby(["condition", "width"]).agg(below80_30=("below80_30", "mean"), below80_60=("below80_60", "mean"),
        diff_mean=("diff", "mean"), diff_sd=("diff", "std"), n_better=("diff", lambda x: int((x < 0).sum())), n=("diff", "size"),
        mid30=("midPICP_30", "mean"), mid60=("midPICP_60", "mean"))
print("Per-seed (mean over seeds):"); print(seeds.round(2).to_string())
print("Ensemble:"); print(r[r.model == "ensemble"].round(2).to_string(index=False))
