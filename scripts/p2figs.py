"""Paper 2 figures: paired per-seed slope charts, 30 vs 60 cycles, matched endpoints."""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, outdir = sys.argv[1], sys.argv[2]
INK, MUTED, GRID, GREY, BLUE, RED = "#1A1A1A", "#5F5F5F", "#E3E3E3", "#B5B5A6", "#2878D7", "#C8553D"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 11})
m = pd.read_csv(src)
conds = ["IID", "DS-A", "DS-B"]


def slope(metric, ylabel, title, subtitle, fname, better_lower=True, fmt="{:.0f}"):
    fig, axes = plt.subplots(1, 3, figsize=(8.96, 4.0), dpi=150, sharey=True)
    for ax, c in zip(axes, conds):
        d = m[m.condition == c].pivot(index="seed", columns="window", values=metric)
        for seed, r in d.iterrows():
            improved = (r[60] < r[30]) if better_lower else (r[60] > r[30])
            col = BLUE if improved else (GREY if r[60] == r[30] else RED)
            ax.plot([0, 1], [r[30], r[60]], color=col, lw=1.8, marker="o", ms=5, zorder=2)
        ax.plot([0, 1], [d[30].mean(), d[60].mean()], color=INK, lw=3, zorder=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["30 cycles", "60 cycles"], color=INK)
        ax.set_xlim(-0.3, 1.3)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.tick_params(colors=MUTED, length=0)
        n_imp = int(((d[60] < d[30]) if better_lower else (d[60] > d[30])).sum())
        ax.set_title(f"{c}\n{n_imp} of 5 seeds improve", color=INK, fontsize=11.5)
    axes[0].set_ylabel(ylabel, color=MUTED)
    fig.suptitle(title, x=0.012, ha="left", fontsize=13.5, color=INK)
    fig.text(0.012, 0.88, subtitle, ha="left", color=MUTED, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(f"{outdir}/{fname}", dpi=150); plt.close(fig)


slope("Bal_below80", "Engines below 80% (Balanced CQR)",
      "Under shift, 60 cycles cuts under-covered engines in every seed but one",
      "Matched endpoints; one line per seed (blue = fewer engines, red = more, grey = no change); black = five-seed mean",
      "p2_fig1_engines_below80.png")
slope("Mid_RMSE", "Mid-life RMSE (cycles)",
      "Mid-life error falls on average, but not for every seed",
      "Matched endpoints; one line per seed (blue = lower error, red = higher); black = five-seed mean",
      "p2_fig2_midlife_rmse.png")
print("ok")
