#!/usr/bin/env python3
"""Social-media graphics (LinkedIn) from the recorded benchmark numbers.

Shows the formats where .omnia clearly speeds up training (>= 1.8x) and says, in the
footer, where it does not. Numbers are the same ones as benchmarks/headline.json.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "charts"
INK, MUTED, GRAY, BLUE, BG = "#0f172a", "#64748b", "#cbd5e1", "#1d4ed8", "#ffffff"

# label, original file items/s, .omnia items/s
ROWS = [
    ("Aperio SVS  (JPEG 2000, full resolution)", 292, 2491, "8.5x"),
    ("Aperio SVS  (JPEG, 20x)",                  455, 1874, "4.1x"),
    ("3DHISTECH MIRAX",                          555, 2116, "3.8x"),
    ("Leica SCN",                                810, 2225, "2.7x"),
    ("Aperio SVS  (JPEG, full resolution)",     1348, 2489, "1.8x"),
]
FOOT = ("NVIDIA A40 · ResNet-18 + AMP · 4 data workers · shuffled tiles · lossless .omnia · public OpenSlide sample slides.\n"
        "Not faster: Hamamatsu NDPI, DICOM CT/MR (ties). Marginal (<=1.3x): generic/Philips TIFF. Two runs differed by up to 2x on some rows.\n"
        "Full results, JSON and code: github.com/mishel-0/omnia-sdk")


def draw(w_px, h_px, name, title_size, bar_h):
    fig = plt.figure(figsize=(w_px / 100, h_px / 100), dpi=100, facecolor=BG)
    ax = fig.add_axes([0.30 if w_px > h_px else 0.34, 0.20, 0.50 if w_px > h_px else 0.46, 0.50])
    n = len(ROWS)
    for i, (label, a, b, mult) in enumerate(ROWS[::-1]):
        ax.barh(i + bar_h / 2, a, bar_h, color=GRAY)
        ax.barh(i - bar_h / 2, b, bar_h, color=BLUE)
        ax.text(b + 60, i - bar_h / 2, f"{b:,.0f}/s", va="center", fontsize=title_size * .55, color=BLUE, fontweight="bold")
        ax.text(a + 60, i + bar_h / 2, f"{a:,.0f}/s", va="center", fontsize=title_size * .5, color=MUTED)
        ax.text(3650, i, mult, va="center", ha="right", fontsize=title_size * 1.35, color=INK, fontweight="bold")
    ax.set_yticks(range(n), [r[0] for r in ROWS[::-1]], fontsize=title_size * .62, color=INK)
    ax.set_xlim(0, 3700); ax.set_xticks([])
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRAY); ax.tick_params(axis="y", length=0)
    fig.text(0.04, 0.90, "Train from .omnia instead of the original slide file", fontsize=title_size * 1.1, color=INK, fontweight="bold")
    fig.text(0.04, 0.835, "Training throughput, tiles per second. Gray = original file, blue = .omnia. Right = speed-up.",
             fontsize=title_size * .6, color=MUTED)
    fig.text(0.04, 0.045, FOOT, fontsize=title_size * .42, color=MUTED, va="bottom", linespacing=1.5)
    fig.savefig(OUT / name, dpi=100, facecolor=BG)
    plt.close(fig)


draw(1200, 627, "linkedin_1200x627.png", 15, .36)
draw(1080, 1080, "linkedin_1080x1080.png", 15, .36)
print("ok")
