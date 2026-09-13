"""Fig. 2a: swap accuracy per model and property with 95% CIs (Exp. 1) and the LLM floor (Exp. 2).
Reads results/summary.json (python -m masb.summarize). Set BW=1 for a greyscale variant.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from style import (set_style, clean_axes, save, OKABE, BW,
                   MODEL_ORDER, MODEL_NAME, AXIS_COLOR, AXIS_LABEL, AXIS_HATCH, BAR_EDGE)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.environ.get("MASB_FIGDIR", os.path.join(HERE, "out"))
os.makedirs(FIGDIR, exist_ok=True)
SUMMARY = os.path.join(os.environ.get("MASB_RESULTS", os.path.join(HERE, "..", "results")), "summary.json")
_s = json.load(open(SUMMARY))
ACC = {m: {ax: (_s["swap_accuracy"][m][ax]["acc"], *_s["swap_accuracy"][m][ax]["ci95"]) for ax in "TRO"} for m in MODEL_ORDER}
FLOOR = {ax: (_s["floor"][ax]["acc"], *_s["floor"][ax]["ci95"]) for ax in "TRO"}


set_style()
fig, ax = plt.subplots(figsize=(3.5, 1.95))

axes3 = ["T", "R", "O"]
groups = MODEL_ORDER + ["floor"]
xlabels = [MODEL_NAME[m] for m in MODEL_ORDER] + ["Qwen2.5-7B"]
n = len(groups)
x = np.arange(n)
w = 0.26


def yerr(acc, lo, hi):
    return [[acc - lo], [hi - acc]]


for i, ax_name in enumerate(axes3):
    xs, ys, elo, ehi = [], [], [], []
    for gi, g in enumerate(groups):
        d = FLOOR if g == "floor" else ACC[g]
        acc, lo, hi = d[ax_name]
        xs.append(x[gi] + (i - 1) * w)
        ys.append(acc)
        elo.append(acc - lo)
        ehi.append(hi - acc)
    # models solid; the text-only group is hatched (colour) or white-faced (BW)
    for j, g in enumerate(groups):
        if g == "floor":
            fc = "white" if BW else AXIS_COLOR[ax_name]
            hh = AXIS_HATCH[ax_name] if BW else "///"
            ec = "#000000" if BW else "white"
        else:
            fc, hh, ec = AXIS_COLOR[ax_name], AXIS_HATCH[ax_name] or None, BAR_EDGE
        ax.bar(xs[j], ys[j], w, color=fc, edgecolor=ec, linewidth=0.4, hatch=hh,
               zorder=3, label=AXIS_LABEL[ax_name] if j == 0 else None)
    ax.errorbar(xs, ys, yerr=[elo, ehi], fmt="none", ecolor="#333333",
                elinewidth=0.7, capsize=1.6, zorder=4)

# chance line
ax.axhline(0.5, color=OKABE["black"], lw=0.8, ls=(0, (4, 3)), zorder=2)
ax.text(x[-1] + 0.5, 0.5, "chance", va="bottom", ha="left",
        fontsize=6.5, color="#333333",
        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))

# separator before floor group
ax.axvline(x[len(MODEL_ORDER)] - 0.5, color="#CCCCCC", lw=0.6, ls=":", zorder=1)

ax.set_ylim(0, 0.95)
ax.set_ylabel("swap accuracy")
ax.set_xticks(x)
ax.set_xticklabels(xlabels, rotation=30, ha='right', fontsize=6.5)
ax.set_xlim(-0.6, n - 0.15)
clean_axes(ax)

axis_handles = [Patch(facecolor=AXIS_COLOR[a], hatch=AXIS_HATCH[a] or None,
                      edgecolor=BAR_EDGE, linewidth=0.4, label=AXIS_LABEL[a]) for a in axes3]
if BW:
    axis_handles.append(Patch(facecolor="white", edgecolor="#000000", linewidth=0.4,
                              label="text-only (white)"))
else:
    axis_handles.append(Patch(facecolor="#DDDDDD", hatch="///", edgecolor="#888888",
                              label="LLM (hatched)"))
ax.legend(handles=axis_handles, frameon=False, loc="upper center",
          ncol=2, fontsize=6, handlelength=1.0, columnspacing=1.0, borderaxespad=0.2)


save(fig, os.path.join(FIGDIR, "fig_swapacc"))
print("wrote fig_swapacc")
