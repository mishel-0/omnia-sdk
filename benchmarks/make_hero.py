#!/usr/bin/env python3
"""Hero benchmark graphic for the README and social posts (light + dark).

Same recorded numbers as benchmarks/headline.json, sorted from smallest to largest
speed-up. Footer states where .omnia is not faster.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "charts"

# label, original tiles/s, .omnia tiles/s
ROWS = sorted([
    ("Aperio SVS\nJPEG · full-res", 1348, 2489),
    ("Leica\nSCN", 810, 2225),
    ("3DHISTECH\nMIRAX", 555, 2116),
    ("Aperio SVS\nJPEG · 20x", 455, 1874),
    ("Aperio SVS\nJPEG 2000 · full-res", 292, 2491),
], key=lambda r: r[2] / r[1])

THEMES = {
    "light": dict(bg="#ffffff", ink="#0b1220", muted="#5b6577", grid="#e6e9ef", base="#c7cdd8",
                  c0="#60a5fa", c1="#1d4ed8"),
    "dark": dict(bg="#0b1020", ink="#f1f5f9", muted="#94a3b8", grid="#1e2638", base="#3a4458",
                 c0="#38bdf8", c1="#818cf8"),
}


def rounded_bar(ax, x, h, w, color):
    ax.add_patch(FancyBboxPatch((x - w / 2, 0), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                mutation_aspect=1 / 1.6, linewidth=0, facecolor=color))


def draw(theme, name, size=(12, 6.6)):
    t = THEMES[theme]
    cmap = LinearSegmentedColormap.from_list("g", [t["c0"], t["c1"]])
    fig = plt.figure(figsize=size, dpi=150, facecolor=t["bg"])
    ax = fig.add_axes([0.06, 0.31, 0.90, 0.45]); ax.set_facecolor(t["bg"])
    n = len(ROWS)
    ups = [b / a for _, a, b in ROWS]
    top = max(ups) * 1.18
    for i, (label, a, b) in enumerate(ROWS):
        up = b / a
        ax.bar(i, up, .56, color=cmap(i / (n - 1)), linewidth=0)
        ax.text(i, up + top * .03, f"{up:.1f}×", ha="center", va="bottom", fontsize=24, fontweight="bold", color=t["ink"])
        ax.text(i, -top * .07, label, ha="center", va="top", fontsize=10.5, color=t["ink"], linespacing=1.35)
        ax.text(i, -top * .25, f"{a:,} → {b:,} tiles/s", ha="center", va="top", fontsize=8.6, color=t["muted"])
    ax.axhline(1, color=t["base"], lw=1.6, ls=(0, (4, 3)))
    ax.text(-.58, 1 - top * .015, "original file = 1×", ha="left", va="top", fontsize=9, color=t["muted"])
    ax.set_xlim(-.6, n - .4); ax.set_ylim(0, top); ax.axis("off")
    fig.text(0.06, 0.90, "Train up to 8.5× faster by reading .omnia instead of the slide file",
             fontsize=19, fontweight="bold", color=t["ink"])
    fig.text(0.06, 0.845, "Same training loop, same GPU — only the file format changes. Higher is better.",
             fontsize=11.5, color=t["muted"])
    fig.text(0.06, 0.035,
             "NVIDIA A40 · ResNet-18 (AMP) · 4 data workers · shuffled tiles · lossless .omnia · public OpenSlide sample slides · mean of runs.\n"
             "Not faster on Hamamatsu NDPI or DICOM CT/MR. Full results, raw JSON and the script to reproduce on your data: github.com/mishel-0/omnia-sdk",
             fontsize=8, color=t["muted"], linespacing=1.6)
    fig.savefig(OUT / name, facecolor=t["bg"])
    plt.close(fig)


draw("light", "hero_speedup_light.png")
draw("dark", "hero_speedup_dark.png")
draw("light", "hero_speedup_square.png", size=(8, 8))
print("ok")
