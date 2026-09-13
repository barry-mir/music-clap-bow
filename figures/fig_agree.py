"""Fig. 2b: Cohen's kappa between each model's per-pair choice and the LLM's (Exp. 2).
Reads results/summary.json (python -m masb.summarize). Set BW=1 for a greyscale variant.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from style import (set_style, clean_axes, save, OKABE,
                   MODEL_ORDER, MODEL_NAME, AXIS_COLOR, AXIS_LABEL, AXIS_HATCH, BAR_EDGE)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.environ.get("MASB_FIGDIR", os.path.join(HERE, "out"))
os.makedirs(FIGDIR, exist_ok=True)
SUMMARY = os.path.join(os.environ.get("MASB_RESULTS", os.path.join(HERE, "..", "results")), "summary.json")
KAPPA = json.load(open(SUMMARY))["kappa"]


set_style()
fig, ax = plt.subplots(figsize=(3.4, 1.75))
axes3 = ["T", "R", "O"]
x = np.arange(len(MODEL_ORDER))
w = 0.26
for i, a in enumerate(axes3):
    ax.bar(x + (i - 1) * w, [KAPPA[m][a] for m in MODEL_ORDER], w,
           color=AXIS_COLOR[a], edgecolor=BAR_EDGE, linewidth=0.4,
           hatch=AXIS_HATCH[a] or None, zorder=3)
ax.axhline(0, color="#333333", lw=1.0, zorder=4)
ax.set_ylim(-0.3, 0.55)
ax.set_ylabel("Cohen's $\\kappa$ vs. LLM")
ax.set_xticks(x)
ax.set_xticklabels([MODEL_NAME[m] for m in MODEL_ORDER], rotation=25, ha="right", fontsize=6.5)
clean_axes(ax)
handles = [Patch(facecolor=AXIS_COLOR[a], hatch=AXIS_HATCH[a] or None, edgecolor=BAR_EDGE,
                 linewidth=0.4, label=AXIS_LABEL[a]) for a in axes3]
ax.legend(handles=handles, frameon=False, loc="upper left", ncol=3,
          fontsize=6.5, handlelength=1.1, columnspacing=1.0, borderaxespad=0.2)
save(fig, os.path.join(FIGDIR, "fig_agree"))
print("wrote fig_agree")
