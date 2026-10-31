#!/usr/bin/env python3
"""Benchmark figure for the README (light and dark variants).

Recorded numbers only (see benchmarks/headline.json). Horizontal grouped bars, sorted by
speed-up, with the setup and the formats where .omnia is not faster stated underneath.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = Path(__file__).resolve().parent / "charts"
for f in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/HelveticaNeue.ttc"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"]

# label, original tiles/s, .omnia tiles/s
ROWS = sorted([
    ("Aperio SVS, JPEG, level 0", 1348, 2489),
    ("Leica SCN", 810, 2225),
    ("3DHISTECH MIRAX", 555, 2116),
    ("Aperio SVS, JPEG, level 1", 455, 1874),
    ("Aperio SVS, JPEG 2000, level 0", 292, 2491),
], key=lambda r: r[2] / r[1])

THEMES = {
    "light": dict(bg="#ffffff", ink="#1f2328", muted="#656d76", grid="#eaeef2", base="#afb8c1", accent="#0969da"),
    "dark":  dict(bg="#0d1117", ink="#e6edf3", muted="#8d96a0", grid="#21262d", base="#484f58", accent="#4493f8"),
}


def draw(theme, name, size=(10, 5.4)):
    t = THEMES[theme]
    fig = plt.figure(figsize=size, dpi=200, facecolor=t["bg"])
    ax = fig.add_axes([0.27, 0.17, 0.63, 0.66]); ax.set_facecolor(t["bg"])
    h = 0.34
    for i, (label, a, b) in enumerate(ROWS):
        ax.barh(i + h / 2, a, h, color=t["base"], linewidth=0)
        ax.barh(i - h / 2, b, h, color=t["accent"], linewidth=0)
        ax.text(a + 30, i + h / 2, f"{a:,}", va="center", fontsize=8.5, color=t["muted"])
        ax.text(b + 30, i - h / 2, f"{b:,}", va="center", fontsize=8.5, color=t["ink"])
        ax.text(3150, i, f"{b / a:.1f}×", va="center", ha="right", fontsize=10.5, color=t["ink"], fontweight="bold")
    ax.set_yticks(range(len(ROWS)), [r[0] for r in ROWS], fontsize=9.5, color=t["ink"])
    ax.set_xlim(0, 3150)
    ax.set_xticks([0, 500, 1000, 1500, 2000, 2500])
    ax.tick_params(axis="x", colors=t["muted"], labelsize=8.5, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=t["grid"], lw=0.8); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Training throughput (tiles/s), higher is better", fontsize=9, color=t["muted"], labelpad=8)
    ax.text(3150, len(ROWS) - 0.45, "speed-up", ha="right", fontsize=8.5, color=t["muted"])

    fig.text(0.03, 0.935, "Training throughput: original slide file vs .omnia", fontsize=13, color=t["ink"], fontweight="bold")
    fig.text(0.03, 0.885, "ResNet-18 (AMP), NVIDIA A40, 4 DataLoader workers, shuffled tiles. Lossless .omnia.",
             fontsize=9, color=t["muted"])
    # legend
    lx = 0.03
    for color, text in ((t["base"], "Original file (OpenSlide)"), (t["accent"], ".omnia")):
        fig.patches.append(plt.Rectangle((lx, 0.045), 0.012, 0.022, transform=fig.transFigure, color=color))
        fig.text(lx + 0.018, 0.048, text, fontsize=8.5, color=t["ink"])
        lx += 0.22
    fig.text(0.97, 0.048, "Not faster on Hamamatsu NDPI or DICOM. Full results: docs/FORMATS.md",
             ha="right", fontsize=8, color=t["muted"])
    fig.savefig(OUT / name, facecolor=t["bg"])
    plt.close(fig)


draw("light", "hero_speedup_light.png")
draw("dark", "hero_speedup_dark.png")
draw("light", "hero_speedup_square.png", size=(8, 6))
print("ok")
