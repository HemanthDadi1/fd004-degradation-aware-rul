"""Paper 2 factorial and context sweep on matched endpoints (same seed, engine, endpoint cycle), five seeds,
reported with all seeds and with collapsed seeds (constant-quantile solution, loss ratio > 0.95) excluded.

Factorial: 4 architectures x 30/60 cycles x IID/DS-A, all on the endpoints a 60-cycle window can score.
Sweep: Part I model at 30/45/60/90 cycles x IID/DS-A/DS-B, all on the endpoints a 90-cycle window can score."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
C = {"IID": "IID", "DS-A": "DS_A", "DS-B": "DS_B"}


def folder(cond, w, arch):
    if arch == "bilstm_attn":
        return f"FD004_{C[cond]}_Rerun_Results" if w == 30 else f"FD004_{C[cond]}_W{w}_Results"
    return f"FD004_{C[cond]}_W{w}_{arch}_Results"


def load(cond, w, arch):
    d = pd.read_csv(ROOT / folder(cond, w, arch) / "test_predictions_rowlevel.csv").rename(columns={
        "engine_id": "engine", "asym_lower": "asymmetric_lower", "asym_upper": "asymmetric_upper",
        "bal_lower": "balanced_lower", "bal_upper": "balanced_upper"})  # the Part I rerun files use the short names
    if folder(cond, w, arch) == "FD004_IID_Rerun_Results":
        d["cycle"] = d["cycle"] + 30  # this file stores the window index; the first endpoint is cycle 30
    d["err"] = d.q50 - d.true_RUL
    d["a_cov"] = (d.true_RUL >= d.asymmetric_lower) & (d.true_RUL <= d.asymmetric_upper)
    d["b_cov"] = (d.true_RUL >= d.balanced_lower) & (d.true_RUL <= d.balanced_upper)
    d["b_w"] = d.balanced_upper - d.balanced_lower
    return d[["seed", "engine", "cycle", "true_RUL", "stage", "err", "a_cov", "b_cov", "b_w"]]


def collapsed(cond, w, arch):
    c = pd.read_csv(ROOT / folder(cond, w, arch) / "convergence_check.csv")
    return set(c.loc[c.ratio > 0.95, "seed"])


rmse = lambda e: float(np.sqrt(np.mean(np.square(e))))
METRICS = ["RMSE", "Mid_RMSE", "Asym_PICP", "Bal_PICP", "Bal_MPIW", "Bal_Mid_PICP", "Bal_below80", "Bal_eng_worst"]
BETTER = {"RMSE": -1, "Mid_RMSE": -1, "Asym_PICP": 1, "Bal_PICP": 1, "Bal_MPIW": -1, "Bal_Mid_PICP": 1, "Bal_below80": -1, "Bal_eng_worst": 1}


def metrics(g):
    e = g.groupby("engine").b_cov.mean() * 100
    mid = g[g.stage == "Mid"]
    return {"N": len(g), "RMSE": rmse(g.err), "Mid_RMSE": rmse(mid.err), "Asym_PICP": 100 * g.a_cov.mean(),
            "Bal_PICP": 100 * g.b_cov.mean(), "Bal_MPIW": float(g.b_w.mean()), "Bal_Mid_PICP": 100 * mid.b_cov.mean(),
            "Bal_below80": int((e < 80).sum()), "Bal_eng_worst": float(e.min())}


def matched(models, min_cycle):
    """models: {label: (cond, w, arch)}; returns per-seed metrics on the endpoints every model scores (cycle >= min_cycle)."""
    rows = []
    data = {k: load(*v) for k, v in models.items()}
    keys = None
    for d in data.values():
        k = d.loc[d.cycle >= min_cycle, ["seed", "engine", "cycle"]]
        keys = k if keys is None else keys.merge(k)
    for label, d in data.items():
        m = keys.merge(d)
        assert len(m) == len(keys), f"{label}: missing endpoints"
        bad = collapsed(*models[label])
        for seed, g in m.groupby("seed"):
            rows.append({"model": label, "cond": models[label][0], "W": models[label][1], "arch": models[label][2],
                         "seed": seed, "collapsed": seed in bad, **metrics(g)})
    return pd.DataFrame(rows)


def summarize(r):
    out = []
    for subset, rr in [("all seeds", r), ("collapsed excluded", r[~r.collapsed])]:
        g = rr.groupby(["cond", "arch", "W"])
        s = g[METRICS].agg(["mean", "std"])
        s.columns = [f"{a}_{b}" for a, b in s.columns]
        s["n_seeds"] = g.size(); s["subset"] = subset
        out.append(s.reset_index())
    return pd.concat(out)


def paired(r, base_w, key):
    """Per-seed differences against base_w within each (cond, key); pairs with a collapsed side are dropped."""
    out = []
    for (cond, k), g in r.groupby(["cond", key]):
        base = g[g.W == base_w].set_index("seed")
        for w, gw in g[g.W != base_w].groupby("W"):
            gw = gw.set_index("seed")
            ok = ~(gw.collapsed | base.collapsed.reindex(gw.index))
            d = gw.loc[ok, METRICS] - base.loc[gw.index[ok], METRICS]
            row = {"cond": cond, key: k, "W": w, "base_W": base_w, "pairs": int(ok.sum())}
            for m in METRICS:
                row[m] = f"{d[m].mean():+.2f} ± {d[m].std():.2f} ({int((d[m] * BETTER[m] > 0).sum())}/{int(ok.sum())})"
            out.append(row)
    return pd.DataFrame(out)


ARCHS = ["bilstm_attn", "lstm_attn", "bilstm_noattn", "lstm_noattn"]
fac = pd.concat([matched({f"{c} {a} W{w}": (c, w, a) for a in ARCHS for w in (30, 60)}, 60) for c in ["IID", "DS-A"]])
swp = pd.concat([matched({f"{c} W{w}": (c, w, "bilstm_attn") for w in (30, 45, 60, 90)}, 90) for c in ["IID", "DS-A", "DS-B"]])

for name, r, pk in [("factorial", fac, "arch"), ("sweep", swp, "arch")]:
    r.to_csv(OUT / f"{name}_matched_by_seed.csv", index=False)
    s = summarize(r); s.to_csv(OUT / f"{name}_matched_summary.csv", index=False)
    p = paired(r, 30, pk); p.to_csv(OUT / f"{name}_matched_paired_vs30.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200); pd.set_option("display.max_columns", 30)
for name in ["factorial", "sweep"]:
    s = pd.read_csv(OUT / f"{name}_matched_summary.csv")
    print(f"\n=== {name}: matched, N per seed", pd.read_csv(OUT / f"{name}_matched_by_seed.csv").groupby("cond").N.first().to_dict())
    print(s[["subset", "cond", "arch", "W", "n_seeds", "RMSE_mean", "RMSE_std", "Mid_RMSE_mean", "Bal_PICP_mean", "Bal_MPIW_mean",
             "Bal_Mid_PICP_mean", "Bal_below80_mean", "Bal_below80_std", "Bal_eng_worst_mean"]].round(2).to_string(index=False))
    p = pd.read_csv(OUT / f"{name}_matched_paired_vs30.csv")
    print(p[["cond", "arch", "W", "pairs", "RMSE", "Mid_RMSE", "Bal_MPIW", "Bal_below80", "Bal_Mid_PICP"]].to_string(index=False))
