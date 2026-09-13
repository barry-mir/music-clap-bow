"""Fig. 3: per-pair cosine distance from c+ to the swapped caption, the paraphrase, and another
recording's caption, per text encoder (Exp. 3). Violins with medians, log y axis.
Reads results/text_text_<model>.json. Set BW=1 for a greyscale variant.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from style import set_style, clean_axes, save, OKABE, MODEL_NAME, BW

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.environ.get("MASB_FIGDIR", os.path.join(HERE, "out"))
os.makedirs(FIGDIR, exist_ok=True)
RES = os.environ.get("MASB_RESULTS", os.path.join(HERE, "..", "results"))

KEYS = [("laion-clap", "laion-clap-music"), ("ms-clap", "ms-clap-2023"),
        ("muq-mulan", "muq-mulan"), ("clamp3", "clamp3-saas")]
if BW:
    COLS = [("d_swap", "attribute swap", "#404040", ""),
            ("d_paraphrase", "paraphrase", "#9A9A9A", "///"),
            ("d_other", "other clip", "#E6E6E6", "...")]
else:
    COLS = [("d_swap", "attribute swap", OKABE["blue"], ""),
            ("d_paraphrase", "paraphrase", OKABE["orange"], ""),
            ("d_other", "other clip", "#C8C8C8", "")]  # light grey: luminance 0.78 vs orange 0.68, blue 0.38

set_style()
fig, ax = plt.subplots(figsize=(3.4, 1.05))
w = 0.26
x = np.arange(len(KEYS))
meds = {}
for j, (key, name) in enumerate(KEYS):
    rows = json.load(open(f"{RES}/text_text_{key}.json"))["rows"]
    meds[name] = {}
    for i, (col, lab, color, hatch) in enumerate(COLS):
        v = np.array([r[col] for r in rows], dtype=float)
        v = np.clip(v, 4e-4, None)  # log axis floor (min observed d_swap is 5.4e-4)
        pos = x[j] + (i - 1) * w
        parts = ax.violinplot([v], positions=[pos], widths=w * 0.95,
                              showmeans=False, showmedians=False, showextrema=True)
        for b in parts["bodies"]:
            b.set_facecolor(color); b.set_alpha(1.0 if BW else 0.85)
            if BW:
                b.set_edgecolor("#000000"); b.set_linewidth(0.4); b.set_hatch(hatch or None)
            else:
                b.set_edgecolor("none")
        for k in ("cbars", "cmins", "cmaxes"):  # min/max caps and the connecting stem
            parts[k].set_color("#333333"); parts[k].set_linewidth(0.6)
        m = float(np.median(v))
        mc = "white" if (BW and color == "#404040") else "black"
        ax.plot([pos - w * 0.35, pos + w * 0.35], [m, m], color=mc, lw=0.9, zorder=4)
        meds[name][col] = m
ax.set_yscale("log")
ax.set_ylim(3e-4, 6.0)
ax.set_yticks([0.001, 0.01, 0.1, 1.0])
ax.set_yticklabels(["0.001", "0.01", "0.1", "1"])
ax.set_ylabel("cosine distance")
ax.set_xticks(x)
ax.set_xticklabels([MODEL_NAME[n] for _, n in KEYS])
clean_axes(ax)
handles = [Patch(facecolor=c, hatch=h or None, edgecolor="#000000" if BW else "none",
                 linewidth=0.4, label=l) for _, l, c, h in COLS]
ax.legend(handles=handles, frameon=False, loc="upper left", ncol=3, fontsize=6.5,
          handlelength=1.1, columnspacing=1.0, borderaxespad=0.2)
save(fig, os.path.join(FIGDIR, "fig_textenc"))
print("medians (d_swap, d_paraphrase, d_other):")
for n, d in meds.items():
    print(f"  {n}: {d['d_swap']:.4f} {d['d_paraphrase']:.4f} {d['d_other']:.4f}")
print("wrote fig_textenc")
