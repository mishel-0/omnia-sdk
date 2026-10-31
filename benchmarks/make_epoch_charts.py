#!/usr/bin/env python3
"""Epoch-time charts and JSON.

Epoch time is derived from measured training throughput (benchmarks/headline.json):
seconds per epoch = tiles per epoch / tiles per second. Nothing new is measured here.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
OUT = HERE / "charts"
for f in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Georgia.ttf"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
SANS = ["Helvetica", "Arial", "DejaVu Sans"]
SERIF = ["Georgia", "DejaVu Serif"]
plt.rcParams["font.family"] = SANS

GPU = "NVIDIA A40 (48 GB), RunPod, 9 vCPU"
MODEL = "ResNet-18, mixed precision (AMP), batch 64, 4 DataLoader workers, shuffled tiles"
TILES_PER_EPOCH = 100_000
EPOCHS = 20

# label, original tiles/s, .omnia tiles/s, colour
ROWS = [
    ("Aperio SVS · JPEG 2000 · level 0", 292, 2491, "#6d3fc0"),
    ("Aperio SVS · JPEG · level 1", 455, 1874, "#d97706"),
    ("3DHISTECH MIRAX", 555, 2116, "#0f8b8d"),
    ("Leica SCN", 810, 2225, "#c2410c"),
    ("Aperio SVS · JPEG · level 0", 1348, 2489, "#2563eb"),
]
INK, MUTED, GRID = "#1f2328", "#5b6470", "#e3e8ee"


def minutes(rate):
    return TILES_PER_EPOCH / rate / 60


# ── 1. Line chart: cumulative training time over 20 epochs ───────────────
fig, ax = plt.subplots(figsize=(11, 6.4), dpi=200)
fig.patch.set_facecolor("white")
ep = list(range(0, EPOCHS + 1))
for label, a, b, c in ROWS:
    orig = [e * minutes(a) for e in ep]
    omni = [e * minutes(b) for e in ep]
    ax.plot(ep, orig, color=c, lw=1.6, ls=(0, (4, 3)), marker="o", ms=3.2, markevery=4, mfc="white", mew=1.2)
    ax.plot(ep, omni, color=c, lw=2.2, marker="o", ms=3.6, markevery=4, mfc=c, mec="white", mew=.8)
    ax.text(EPOCHS + .3, orig[-1], f"{label}  {orig[-1]:.0f} min", va="center", fontsize=8.6, color=c)
ax.text(EPOCHS + .3, max(EPOCHS * minutes(b) for _, _, b, _ in ROWS) - 3, ".omnia: 13–18 min for all five",
        va="top", fontsize=8.6, color=INK, fontweight="bold")
ax.set_xlim(0, EPOCHS); ax.set_ylim(0, 120)
ax.set_xticks(range(0, EPOCHS + 1, 4))
ax.set_xlabel("Epoch", fontsize=10.5, color=INK, fontweight="bold")
ax.set_ylabel("Cumulative training time (minutes)", fontsize=10.5, color=INK, fontweight="bold")
ax.grid(color=GRID, lw=.9); ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(colors=MUTED, length=0, labelsize=9)
fig.subplots_adjust(left=.07, right=.70, top=.82, bottom=.16)
fig.text(.07, .93, "Training time with the original slide file vs .omnia", fontsize=19, color=INK, family=SERIF)
fig.text(.07, .875, "Dashed: original file read with OpenSlide.   Solid: the same tiles read from .omnia.", fontsize=9.5, color=MUTED)
fig.text(.07, .045, f"{GPU}. {MODEL}. {TILES_PER_EPOCH:,} tiles per epoch.\n"
         "Times are projected from measured throughput (tiles/s) on public OpenSlide sample slides. "
         "Not faster on Hamamatsu NDPI or DICOM; see docs/FORMATS.md.", fontsize=8, color=MUTED, linespacing=1.5)
fig.savefig(OUT / "epoch_time_lines.png", facecolor="white")
plt.close(fig)

# ── 2. Bar charts: epoch time and throughput, values on top ──────────────
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6), dpi=200, gridspec_kw={"wspace": .22})
fig.patch.set_facecolor("white")
short = ["SVS\nJPEG 2000\nlevel 0", "SVS\nJPEG\nlevel 1", "MIRAX", "Leica\nSCN", "SVS\nJPEG\nlevel 0"]
x = range(len(ROWS)); w = .38
for ax, title, key, unit in ((a1, "Time per epoch (minutes, lower is better)", "min", ""),
                             (a2, "Throughput (tiles per second, higher is better)", "rate", "")):
    for i, (label, a, b, c) in enumerate(ROWS):
        va, vb = (minutes(a), minutes(b)) if key == "min" else (a, b)
        ax.bar(i - w / 2, va, w, color="#b6bec9")
        ax.bar(i + w / 2, vb, w, color="#2563eb")
        fmt = (lambda v: f"{v:.1f}") if key == "min" else (lambda v: f"{v:,.0f}")
        ax.text(i - w / 2, va, fmt(va), ha="center", va="bottom", fontsize=9, color=MUTED, fontweight="bold")
        ax.text(i + w / 2, vb, fmt(vb), ha="center", va="bottom", fontsize=9, color=INK, fontweight="bold")
    ax.set_xticks(list(x), short, fontsize=9, color=INK)
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, fontweight="bold", pad=12)
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c9d1d9")
    ax.tick_params(axis="y", colors=MUTED, length=0, labelsize=8.5); ax.tick_params(axis="x", length=0)
a1.set_ylim(0, 6.6); a2.set_ylim(0, 2900)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#b6bec9"), plt.Rectangle((0, 0), 1, 1, color="#2563eb")],
           labels=["Original file (OpenSlide)", ".omnia"], loc="lower left", bbox_to_anchor=(.06, .02), ncol=2, frameon=False, fontsize=9.5)
fig.text(.06, .94, "Epoch time and throughput: original slide file vs .omnia", fontsize=19, color=INK, family=SERIF)
fig.text(.98, .035, f"{GPU} · {MODEL} · {TILES_PER_EPOCH:,} tiles/epoch", ha="right", fontsize=8, color=MUTED)
fig.subplots_adjust(left=.06, right=.98, top=.82, bottom=.2)
fig.savefig(OUT / "epoch_time_bars.png", facecolor="white")
plt.close(fig)

# ── JSON ─────────────────────────────────────────────────────────────────
rows = []
for label, a, b, _ in ROWS:
    rows.append({
        "format": label.replace(" · ", ", "),
        "throughput_tiles_per_s": {"original_file": a, "omnia": b},
        "speedup": round(b / a, 2),
        "epoch_time_seconds": {"original_file": round(TILES_PER_EPOCH / a, 1), "omnia": round(TILES_PER_EPOCH / b, 1)},
        "time_for_20_epochs_minutes": {"original_file": round(EPOCHS * minutes(a), 1), "omnia": round(EPOCHS * minutes(b), 1)},
    })
(HERE / "epoch_times.json").write_text(json.dumps({
    "gpu": GPU,
    "training": MODEL,
    "original_file_reader": "OpenSlide read_region, decoding on every read",
    "omnia": "lossless .omnia (zstd level 5, one tile per block)",
    "tiles_per_epoch": TILES_PER_EPOCH,
    "method": "epoch time = tiles per epoch / measured training throughput (tiles/s). Throughput measured over 2 timed epochs after a warm-up epoch; mean of two runs where two exist.",
    "samples": "public OpenSlide test slides: JP2K-33003-1.svs, CMU-1.svs, CMU-1.mrxs, Leica-1.scn",
    "not_faster_on": ["Hamamatsu NDPI (tie)", "DICOM CT and MR series (tie)", "DICOM whole-slide (slower on a 57-tile sample)"],
    "results": rows,
}, indent=2))
print("ok")
