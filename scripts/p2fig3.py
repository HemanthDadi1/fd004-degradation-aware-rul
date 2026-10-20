"""Paper 2 Figure 3: ensemble width-reliability curves, 30 vs 60 cycles, matched endpoints."""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
INK, MUTED, GRID, GREY, BLUE = "#1A1A1A", "#5F5F5F", "#E3E3E3", "#A3A394", "#2878D7"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 11})
c = pd.read_csv(src); c = c[c.model == "ensemble"]
fig, axes = plt.subplots(1, 3, figsize=(8.96, 3.9), dpi=150)
for ax, cond in zip(axes, ["IID", "DS-A", "DS-B"]):
    for w, col, lab in [(30, GREY, "30 cycles"), (60, BLUE, "60 cycles")]:
        d = c[(c.condition == cond) & (c.window == w)].sort_values("MPIW")
        ax.plot(d.MPIW, d.below80, color=col, lw=2.2, drawstyle="steps-post", label=lab)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID); ax.set_axisbelow(True); ax.tick_params(colors=MUTED, length=0)
    ax.set_title(cond, color=INK, fontsize=12); ax.set_xlabel("Mean interval width (cycles)", color=MUTED)
    ax.set_xlim(40, 110)
axes[0].set_ylabel("Engines below 80% coverage", color=MUTED)
axes[2].legend(frameon=False, loc="upper right", fontsize=10)
fig.suptitle("With seed noise averaged out, 60 cycles needs less width for the same reliability, most clearly under DS-B",
             x=0.012, ha="left", fontsize=13, color=INK)
fig.text(0.012, 0.875, "Five-seed ensembles, matched endpoints; Window CQR swept from 80% to 98% nominal level",
         ha="left", color=MUTED, fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.87)); fig.savefig(out, dpi=150); plt.close(fig)
print("ok")
