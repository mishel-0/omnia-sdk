#!/usr/bin/env python3
"""Build the consolidated benchmark file and charts from the recorded RunPod runs.

    python benchmarks/build_report.py

Writes benchmarks/results.json (every number, both runs, methodology, caveats) and
benchmarks/charts/*.png|svg. Nothing here is measured: it only reshapes numbers that were
recorded by `omnia benchmark` (see docs/FORMATS.md, docs/CODECS.md).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "charts"
OUT.mkdir(exist_ok=True)

# items/s through a ResNet-18 training loop, 4 data workers, NVIDIA A40. (run 1, run 2)
TRAIN = {
    "Aperio SVS (JPEG)":      {"openslide": (435.9, 474.3),   "omnia_zstd": (1616.4, 2130.7), "omnia_jpeg90": (2059.4, 1986.7)},
    "Aperio SVS (JPEG 2000)": {"openslide": (2020.3, 1025.9), "omnia_zstd": (2106.1, 1507.9), "omnia_jpeg90": (2275.4, 1304.5)},
    "Hamamatsu NDPI":         {"openslide": (2484.8, 3438.4), "omnia_zstd": (2482.7, 3168.1), "omnia_jpeg90": (2264.0, 2490.2)},
    "Leica SCN":              {"openslide": (821.6, 797.7),   "omnia_zstd": (2166.3, 2283.2), "omnia_jpeg90": (2061.2, 2019.7)},
    "Generic TIFF":           {"openslide": (1535.0, 1715.6), "omnia_zstd": (2041.2, 2344.2), "omnia_jpeg90": (2095.7, 2174.6)},
    "Philips TIFF":           {"openslide": (480.7, 469.6),   "omnia_zstd": (612.7, 620.5),   "omnia_jpeg90": (None, 621.6)},
    "3DHISTECH MIRAX":        {"openslide": (544.5, 566.2),   "omnia_zstd": (1940.8, 2291.1), "omnia_jpeg90": (1926.3, 2075.5)},
}
# MB on disk (second pyramid level and below). source, zstd level 5, jpeg q90
SIZE = {
    "Aperio SVS (JPEG)": (177.55, 60.25, 12.43), "Aperio SVS (JPEG 2000)": (63.85, 27.11, 4.95),
    "Hamamatsu NDPI": (198.03, 518.51, 282.43), "Leica SCN": (291.81, 143.31, 44.61),
    "Generic TIFF": (204.12, 352.98, 72.67), "Philips TIFF": (326.61, 623.5, 114.13),
    "3DHISTECH MIRAX": (565.1, 1214.48, 340.33),
    "NIfTI MRI": (8.68, 7.70, None), "DICOM CT": (60.0, 22.52, None), "DICOM MR": (40.01, 10.93, None),
    "DICOM whole-slide": (40.02, 8.42, None),
}
# single-thread random reads, items/s (native, omnia). (run 1, run 2)
READ = {
    "Aperio SVS (JPEG)": ((137.9, 134.2), (3363.6, 3680.3)), "Aperio SVS (JPEG 2000)": ((91.8, 107.8), (2287.9, 2387.0)),
    "Hamamatsu NDPI": ((6553.0, 7476.8), (29524.3, 40072.5)), "Leica SCN": ((202.7, 205.8), (3086.1, 3541.8)),
    "Generic TIFF": ((517.2, 498.0), (3113.5, 3495.6)), "Philips TIFF": ((141.6, 137.3), (912.0, 1083.2)),
    "3DHISTECH MIRAX": ((117.4, 144.4), (3398.0, 3955.2)),
    "NIfTI (gzip slice)": ((15.3, 15.8), (19005.8, 21656.5)),
    "DICOM CT": ((722.7, 965.6), (12642.4, 20202.5)), "DICOM MR": ((701.7, 940.6), (22730.4, 32548.9)),
    "DICOM whole-slide": ((None, 3180.9), (None, 3248.3)),
}
# radiology / DICOM training, items/s at 4 workers (run 2; run 1 where noted)
RADIO = {
    "NIfTI MRI": {"native (gzip slice)": 1694.8, "native (volume in RAM)": 2157.5, "omnia_zstd": 2096.4},
    "DICOM CT": {"native (pydicom)": 1049.4, "omnia_zstd": 921.3},
    "DICOM MR": {"native (pydicom)": 1697.0, "omnia_zstd": 1617.5},
    "DICOM whole-slide": {"native (wsidicom)": 1191.0, "omnia_zstd": 563.9},
}
# codec study (CMU-1 level-0 tiles, 3000 tiles, 4 workers): size MB, GPU share
CODEC = [("raw", 589.8, 0.970), ("zstd L1 / 1 tile", 542.6, 0.975), ("zstd L3 / 1 tile", 541.8, 0.972),
         ("zstd L3 / 16 tiles", 541.6, 0.332), ("zstd L9 / 16 tiles", 543.1, 0.330),
         ("JPEG q75", 48.2, 0.971), ("JPEG q90", 75.6, 0.963), ("JPEG q95", 104.4, 0.964), ("openslide", 177.6, 0.527)]

C_NATIVE, C_ZSTD, C_JPEG, C_BAD = "#8a8f98", "#1f6feb", "#e8710a", "#c0392b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "axes.axisbelow": True})
FOOT = ("NVIDIA A40, ResNet-18 + AMP, shuffled DataLoader, 4 workers, 9 vCPU. Bars = mean of two runs; whiskers = the two runs.\n"
        "Source: github.com/mishel-0/omnia-sdk, docs/FORMATS.md. Public sample slides (OpenSlide test data).")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bars(ax, x, vals, width, color, label):
    mean = [np.mean([v for v in r if v is not None]) if any(v is not None for v in r) else 0 for r in vals]
    lo = [min(v for v in r if v is not None) if any(v is not None for v in r) else 0 for r in vals]
    hi = [max(v for v in r if v is not None) if any(v is not None for v in r) else 0 for r in vals]
    ax.bar(x, mean, width, color=color, label=label)
    ax.errorbar(x, mean, yerr=[np.array(mean) - lo, np.array(hi) - mean], fmt="none", ecolor="#222", capsize=2.5, lw=1)
    return mean


# 1 ─ training throughput on slides
fig, ax = plt.subplots(figsize=(11, 5.2))
names = list(TRAIN)
x = np.arange(len(names)); w = .27
m_open = bars(ax, x - w, [TRAIN[n]["openslide"] for n in names], w, C_NATIVE, "openslide (native)")
m_zstd = bars(ax, x, [TRAIN[n]["omnia_zstd"] for n in names], w, C_ZSTD, ".omnia  zstd (lossless)")
bars(ax, x + w, [TRAIN[n]["omnia_jpeg90"] for n in names], w, C_JPEG, ".omnia  JPEG q90 (lossy)")
for i, n in enumerate(names):
    r = m_zstd[i] / m_open[i]
    ax.text(x[i], max(m_zstd[i], m_open[i]) * 1.12 + 60, f"{r:.1f}x" if abs(r - 1) > .15 else "≈ tie",
            ha="center", fontsize=9.5, fontweight="bold", color=C_ZSTD if r > 1.15 else "#555")
ax.set_xticks(x, [n.replace(" (", "\n(") for n in names], fontsize=9)
ax.set_ylabel("training throughput  (tiles / second)")
ax.set_title("Training throughput by slide format — .omnia vs reading the slide directly", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="upper right"); ax.set_ylim(0, 4100)
fig.text(0.01, -0.06, FOOT + "\nPhilips: GPU-limited at ~640 tiles/s (large tiles); its JPEG value is one run. NDPI is already fast to read.", fontsize=7.5, color="#555")
save(fig, "01_training_throughput_slides")

# 2 ─ single-thread read
fig, ax = plt.subplots(figsize=(11, 5.2))
names = list(READ); x = np.arange(len(names)); w = .38
bars(ax, x - w / 2, [READ[n][0] for n in names], w, C_NATIVE, "native reader")
bars(ax, x + w / 2, [READ[n][1] for n in names], w, C_ZSTD, ".omnia zstd")
ax.set_yscale("log"); ax.set_xticks(x, [n.replace(" (", "\n(") for n in names], fontsize=8.2, rotation=0)
ax.set_ylabel("random-order read, 1 thread  (items / second, log)")
ax.set_title("Raw read speed — single thread, shuffled order, cold file", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="upper left")
for i, n in enumerate(names):
    a = [v for v in READ[n][0] if v]; b = [v for v in READ[n][1] if v]
    if a and b:
        ax.text(x[i], max(b) * 1.35, f"{np.mean(b) / np.mean(a):.0f}x" if np.mean(b) / np.mean(a) >= 3 else f"{np.mean(b) / np.mean(a):.1f}x",
                ha="center", fontsize=9, fontweight="bold", color=C_ZSTD)
ax.set_ylim(8, 1.2e5)
fig.text(0.01, -0.07, "Native readers: openslide, nibabel (slice from .nii.gz), pydicom (one file per slice), wsidicom. Two runs on the same GPU model; whiskers = the two runs.", fontsize=7.5, color="#555")
save(fig, "02_single_thread_read")

# 3 ─ size
fig, ax = plt.subplots(figsize=(11, 5.2))
names = list(SIZE); x = np.arange(len(names)); w = .27
ax.bar(x - w, [SIZE[n][0] for n in names], w, color=C_NATIVE, label="source file")
ax.bar(x, [SIZE[n][1] for n in names], w, color=C_ZSTD, label=".omnia zstd (lossless)")
ax.bar(x + w, [SIZE[n][2] or 0 for n in names], w, color=C_JPEG, label=".omnia JPEG q90 (lossy)")
ax.set_yscale("log"); ax.set_xticks(x, [n.replace(" (", "\n(") for n in names], fontsize=8)
ax.set_ylabel("size on disk, MB (log)"); ax.legend(frameon=False)
ax.set_title("Disk size — lossless is often larger than an already-compressed slide", loc="left", fontweight="bold")
fig.text(0.01, -0.07, "Slide containers keep the second pyramid level and below (min_level=1), so the sources include a level that .omnia drops. Radiology/DICOM are full series.", fontsize=7.5, color="#555")
save(fig, "03_size_on_disk")

# 4 ─ radiology training (honest: mostly ties)
fig, ax = plt.subplots(figsize=(9.5, 4.8))
names = list(RADIO); x = np.arange(len(names)); w = .27
for i, n in enumerate(names):
    d = RADIO[n]; ks = list(d); nat = [k for k in ks if k.startswith("native")]; om = d["omnia_zstd"]
    for j, k in enumerate(nat):
        ax.bar(i - w / 2 - j * w * .0 + (j - (len(nat) - 1) / 2) * w * .9 - w * .5 * 0, d[k], w * .85, color=[C_NATIVE, "#b5b9c0"][j % 2])
    ax.bar(i + w * 1.05, om, w * .85, color=C_ZSTD)
    ax.text(i + w * 1.05, om + 40, f"{om:,.0f}", ha="center", fontsize=8)
ax.set_xticks(x, names); ax.set_ylabel("training throughput  (slices / second)")
ax.set_title("Radiology & DICOM training — .omnia does not beat the native reader here", loc="left", fontweight="bold")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_NATIVE, label="native reader"), Patch(color="#b5b9c0", label="native, volume loaded in RAM"), Patch(color=C_ZSTD, label=".omnia zstd")], frameon=False, loc="upper right")
fig.text(0.01, -0.07, "Run 2. DICOM whole-slide: 57 tiles only (3 batches); treat as 'not shown to help'. Bit-exact round trips for NIfTI, CT, MR.", fontsize=7.5, color="#555")
save(fig, "04_radiology_training")

# 5 ─ codec study: GPU share (the batching finding)
fig, ax = plt.subplots(figsize=(10, 4.9))
labs = [c[0] for c in CODEC]; share = [c[2] * 100 for c in CODEC]; size = [c[1] for c in CODEC]
colors = [C_BAD if "16" in l else (C_NATIVE if l in ("openslide", "raw") else (C_JPEG if "JPEG" in l else C_ZSTD)) for l in labs]
ax.barh(labs[::-1], share[::-1], color=colors[::-1])
for i, (s, mb) in enumerate(zip(share[::-1], size[::-1])):
    ax.text(s + 1, i, f"{s:.0f}%   {mb:,.0f} MB", va="center", fontsize=9)
ax.set_xlim(0, 125); ax.set_xlabel("share of the GPU kept busy (4 workers)")
ax.set_title("Which codec feeds the GPU?  Tiles per block matters more than the zstd level", loc="left", fontweight="bold")
fig.text(0.01, -0.07, "CMU-1.svs, 3000 level-0 tiles of 256 px, A40, ResNet-18. 16 tiles per compressed block made shuffled reads 6-14x slower and saved no space.", fontsize=7.5, color="#555")
save(fig, "05_codec_study")
print("charts:", sorted(p.name for p in OUT.glob("*.png")))


# ── consolidated JSON ────────────────────────────────────────────────────
def pair(v):
    return {"run1": v[0], "run2": v[1]}

doc = {
    "title": "omnia-sdk benchmarks: .omnia container vs each format's usual reader",
    "date": "2026-10-30",
    "hardware": {"gpu": "NVIDIA A40 (RunPod secure cloud)", "vcpu_allotted": 9, "runs": 2,
                 "note": "the two runs landed on different physical hosts; same GPU model"},
    "workload": {"model": "ResNet-18, AMP, channels_last", "batch": 64, "data_workers": 4,
                 "loader": "shuffled torch DataLoader, tiles/slices read on demand",
                 "metric": "items per second through the training loop (after one warm-up epoch)",
                 "slides": "256 px tiles (Philips: larger) from the second pyramid level, tissue tiles only",
                 "omnia_defaults": "zstd level 5, one tile per block; JPEG quality 90"},
    "slides_training_items_per_s": {n: {k: pair(v) for k, v in TRAIN[n].items()} for n in TRAIN},
    "single_thread_read_items_per_s": {n: {"native": pair(READ[n][0]), "omnia_zstd": pair(READ[n][1])} for n in READ},
    "size_mb": {n: {"source": v[0], "omnia_zstd_l5": v[1], "omnia_jpeg_q90": v[2]} for n, v in SIZE.items()},
    "radiology_training_items_per_s_run2": RADIO,
    "codec_study_cmu1": [{"variant": c[0], "size_mb": c[1], "gpu_share": c[2]} for c in CODEC],
    "zstd_level_size_mb": json.loads((HERE / "format_results_a40.json").read_text())["results"],
    "samples": {
        "slides": "OpenSlide public test data (https://openslide.org/demo/): CMU-1.svs, JP2K-33003-1.svs, CMU-1.ndpi, Leica-1.scn, generic-CMU-1.tiff, Philips-1.tiff, CMU-1.mrxs",
        "nifti": "OpenNeuro ds000114 sub-01 ses-test T1w (CC0)",
        "dicom": "NCI Imaging Data Commons, CC BY 4.0: an NLST CT series, a 192-slice MR series, a GTEx whole-slide series",
    },
    "what_is_proven": [
        "Round trips are exact: slide tiles identical to the native reader; NIfTI, CT and MR slices bit-identical.",
        ".omnia reads are 17-35x faster single-threaded than pydicom, and ~1000x faster than slicing a gzip NIfTI.",
        "With 4 data workers .omnia kept the GPU 2.6-4.5x busier than openslide on SVS (JPEG), Leica SCN and MIRAX.",
        "Higher zstd levels do not speed reads; level 5 is within 1-3% of level 9 on size at a third of the write time.",
        "Putting 16 tiles in one compressed block made shuffled reads 6-14x slower and saved no space.",
    ],
    "what_is_not_proven": [
        "Faster training in general: NDPI is a tie, JPEG 2000 is 1.0-1.5x, DICOM CT/MR training is a tie, DICOM whole-slide was slower (57 tiles).",
        "That JPEG re-encoding does not hurt model accuracy: only PSNR and ResNet-18 feature similarity (0.96-0.99) were measured.",
        "Anything on other GPUs, other models, 8+ workers (measured only in run 1), or datasets larger than a few thousand tiles.",
        "Lossless size: for already-compressed slides .omnia zstd is often larger than the source (NDPI 519 vs 198 MB).",
    ],
    "caveats": [
        "Two runs, up to 2x apart on single rows (different hosts). Treat gaps under ~30% as ties.",
        "One GPU model and one network. Public sample slides only; no patient data.",
        "Slide containers keep the second pyramid level and below, so slide sizes are not like-for-like with the source file.",
    ],
}
(HERE / "results.json").write_text(json.dumps(doc, indent=2))
print("wrote results.json")
