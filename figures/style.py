"""Shared plotting style for the MIRB figures.

Colorblind-safe Okabe-Ito palette, IEEE 2-column sizing, thin dashed grid,
no top/right spines. Numbers come from results/summary.json.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette
OKABE = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermil":  "#D55E00",
    "purple":  "#CC79A7",
    "grey":    "#999999",
}

import os
# BW=1 -> greyscale + hatching variants, saved with a _bw suffix
BW = os.environ.get("BW") == "1"

# axis colors (T / R / O) reused across figures; Okabe-Ito blue / orange / green are
# pairwise distinguishable under deuteranopia, protanopia, and tritanopia
if BW:
    AXIS_COLOR = {"T": "#404040", "R": "#9A9A9A", "O": "#E6E6E6"}
    AXIS_HATCH = {"T": "", "R": "///", "O": "..."}
    BAR_EDGE = "#000000"
else:
    # luminance-separated so the three stay distinct in greyscale and under CVD
    AXIS_COLOR = {"T": "#08306B", "R": OKABE["orange"], "O": OKABE["green"]}  # navy / orange / green, luminance 0.12 / 0.68 / 0.50
    AXIS_HATCH = {"T": "", "R": "", "O": ""}
    BAR_EDGE = "white"
AXIS_LABEL = {"T": "Timbre", "R": "Lead/acc.", "O": "Onset"}

# canonical model order + display names
MODEL_ORDER = [
    "laion-clap-music",
    "ms-clap-2023",
    "muq-mulan",
    "clamp3-saas",
    "qwen2-audio-7b-instruct",
]
MODEL_NAME = {
    "laion-clap-music": "LAION-CLAP",
    "ms-clap-2023": "MS-CLAP",
    "muq-mulan": "MuQ-MuLan",
    "clamp3-saas": "CLaMP3",
    "qwen2-audio-7b-instruct": "Qwen2-Audio",
}

# text-encoder order (Exp. 3; generative Qwen excluded)
ENC_ORDER = ["laion-clap-music", "ms-clap-2023", "muq-mulan", "clamp3-saas"]


def set_style():
    mpl.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "pdf.fonttype": 42,        # embed TrueType, editable text
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
        "lines.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "grid.color": "#BBBBBB",
        "grid.linestyle": "--",
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "hatch.linewidth": 0.5,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def clean_axes(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def save(fig, path_noext):
    if BW:
        path_noext += "_bw"
    fig.savefig(path_noext + ".pdf")
    fig.savefig(path_noext + ".png", dpi=300)
    plt.close(fig)
