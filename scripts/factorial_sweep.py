"""Paper 2 factorial (BiLSTM vs LSTM x attention, 30 vs 60) and context sweep (30/45/60/90): five-seed tables.
Local runs from global_by_seed.csv; Colab runs from the five-seed summary printed in the executed notebooks."""
import json, re, sys
from pathlib import Path
import pandas as pd

R, NB, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
M = ["Asymmetric CQR", "Balanced Engine CQR"]
rows = []


def add(arch, W, cond, df, src, collapsed):
    for meth in M:
        d = df[df.method == meth]
        rows.append({"arch": arch, "W": W, "cond": cond, "method": meth, "n_seeds": len(d), "collapsed_seeds": collapsed, "source": src,
                     **{f"{k}_mean": d[k].mean() for k in ["RMSE", "PICP", "MPIW", "below80", "eng_worst"]},
                     **{f"{k}_sd": d[k].std() for k in ["RMSE", "PICP", "MPIW", "below80"]}})


local = {("bilstm_attn", 30, "IID"): "FD004_IID_Rerun_Results", ("bilstm_attn", 30, "DS-A"): "FD004_DS_A_Rerun_Results",
         ("bilstm_attn", 30, "DS-B"): "FD004_DS_B_Rerun_Results", ("bilstm_attn", 60, "IID"): "FD004_IID_W60_Results",
         ("bilstm_attn", 60, "DS-A"): "FD004_DS_A_W60_Results", ("bilstm_attn", 60, "DS-B"): "FD004_DS_B_W60_Results",
         ("lstm_attn", 30, "IID"): "FD004_IID_W30_lstm_attn_Results", ("lstm_attn", 30, "DS-A"): "FD004_DS_A_W30_lstm_attn_Results",
         ("lstm_attn", 60, "IID"): "FD004_IID_W60_lstm_attn_Results", ("lstm_attn", 60, "DS-A"): "FD004_DS_A_W60_lstm_attn_Results",
         ("bilstm_noattn", 30, "IID"): "FD004_IID_W30_bilstm_noattn_Results"}
for (a, W, c), d in local.items():
    g = pd.read_csv(R / d / "global_by_seed.csv")
    if "below80" not in g:  # the Paper 1 DS-A rerun kept engine metrics only in per_engine_by_seed.csv
        pe = pd.read_csv(R / d / "per_engine_by_seed.csv")
        for meth, col in [("Asymmetric CQR", "Asym_Coverage"), ("Balanced Engine CQR", "Bal_Coverage")]:
            e = pe.groupby("seed")[col].agg(below80=lambda x: (x < 80).sum(), eng_worst="min")
            for k in e:
                g.loc[g.method == meth, k] = g.loc[g.method == meth, "seed"].map(e[k])
    conv = pd.read_csv(R / d / "convergence_check.csv")
    # collapsed = predicts the constant quantiles (ratio ~1); mild misses like 0.61 stay in, as in Paper 2
    bad = conv.loc[conv.ratio > 0.95, "seed"].tolist()
    add(a, W, c, g, "local, all seeds", bad)
    if bad:
        add(a, W, c, g[~g.seed.isin(bad)], "local, collapsed excluded", bad)

pat = re.compile(r"^(Asymmetric CQR|Balanced Engine CQR)\s+([\d.\s]+)$", re.M)
for f in sorted(NB.glob("*.ipynb")):
    m = re.match(r"(\d+)_FD004_(IID|DS_A|DS_B)_(\d+)cycle(?:_(\w+))?\.ipynb", f.name)
    if not m:
        continue
    cond, W, arch = m.group(2).replace("_", "-"), int(m.group(3)), m.group(4) or "bilstm_attn"
    outs = ["".join("".join(o.get("text", "")) for o in c.get("outputs", [])) for c in json.load(open(f))["cells"]]
    conv = [t for t in outs if "constant_val_loss  ratio" in t][-1]
    bad = [int(s) for s, r in re.findall(r"^\s+(\d+)\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)", conv, re.M) if float(r) > 0.95]
    summ = [t for t in outs if "PICP_mean" in t][-1]
    cols = summ.split("\n")[0].split()
    for meth, vals in pat.findall(summ):
        v = dict(zip(cols, map(float, vals.split())))
        rows.append({"arch": arch, "W": W, "cond": cond, "method": meth, "n_seeds": 5, "collapsed_seeds": bad, "source": "colab, all seeds",
                     **{f"{k}_mean": v[f"{k}_mean"] for k in ["RMSE", "PICP", "MPIW", "below80", "eng_worst"]},
                     **{f"{k}_sd": v[f"{k}_std"] for k in ["RMSE", "PICP", "MPIW", "below80"]}})

t = pd.DataFrame(rows).sort_values(["cond", "arch", "W", "method", "source"])
t.to_csv(OUT / "factorial_sweep_five_seed.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
show = t[t.method == "Balanced Engine CQR"][["cond", "arch", "W", "source", "collapsed_seeds", "RMSE_mean", "RMSE_sd", "PICP_mean", "MPIW_mean", "below80_mean", "below80_sd", "eng_worst_mean"]]
print(show.round(2).to_string(index=False))
asy = t[t.method == "Asymmetric CQR"][["cond", "arch", "W", "source", "PICP_mean", "MPIW_mean", "below80_mean"]]
print("\nAsymmetric CQR\n", asy.round(2).to_string(index=False))
