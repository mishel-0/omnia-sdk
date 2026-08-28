# omnia-20x

**The data layer for tile-based whole-slide-image training. Up to 20x faster data feeding than openslide — and we show you exactly when that's true.**

## The honest headline

We don't sell magic. We sell fewer idle GPU-hours for the common case in
tile-based classification (Gleason grading, detection, PANDA-style models on
ResNet/EfficientNet-class architectures). On that workload, openslide keeps
your GPU at ~21% utilization while it decodes JPEG-2000 every epoch.
omnia-20x preloads once and feeds the GPU at ~97-100%.

| Regime (full-train speedup) | Typical workload | Data feeding | End-to-end epoch |
|---|---|---|---|
| **Data-bound** (≥ 10x) | Tile classification, grading, detection — small/fast models | **15-25x** | **10x+** |
| **Mixed** (~2-10x) | Medium models, high-throughput pipelines | 5-20x (hidden) | ~2-10x |
| **Model-bound** (~1-2x) | Foundation-model fine-tuning (UNI, Virchow), huge ViTs | 5-20x (hidden) | ~1-2x |

The regime thresholds are exactly 10x and 2x full-train speedup — the same
numbers the benchmark writes into every JSON receipt, so a run's own
classification never contradicts this table.

The benchmark reports BOTH numbers (data-only and full-train) so you — or
your technical diligence — can see which regime you're in. If your model is
the bottleneck, the savings shrink toward break-even, and we say so before
you ask.

## Measured results

Reproduced end-to-end on a Colab **Tesla T4** with a public 178 MB Aperio
slide (CMU-1, 46,000 x 32,914 px), ResNet-18, batch 64, 1,485 tiles/epoch,
4 epochs, epoch 1 discarded:

| | `.svs` + openslide | `.omnia` | |
|---|---|---|---|
| **Epoch time (fp32)** | 18.91s | **5.17s** | **3.66x** |
| Data loading | 18.46s (97.6%) | 0.52s (10.2%) | **35.2x** |
| On disk | 177.6 MB | 61.1 MB | 2.9x smaller |
| Preload (one-off) | — | 1.0s | repaid in 0.1 epochs |

### End-to-end depends on your training config, not just the container

`.svs` is **data-bound** — 97.6% of its epoch is JPEG-2000 decoding, so a
faster GPU changes nothing. `.omnia` is **compute-bound**. Every improvement to
model throughput therefore *increases* the omnia-20x advantage:

| Training config (T4, ResNet-18) | `.omnia` epoch | End-to-end |
|---|---|---|
| fp32 | 5.16s | 3.66x |
| + AMP (fp16 autocast) | 3.26s | **5.81x** |
| + AMP + `channels_last` | 2.62s | **7.23x** |

The `.svs` baseline stays at 18.91s in every row. That asymmetry is the whole
point: this is not a fixed multiplier, it is the removal of a fixed cost.

### The container is not the bottleneck

Profiled on the same T4, isolating each component:

```
pure GPU compute, zero data movement    4.891s   <- the real floor
current .omnia data path                5.087s   <- only 4% overhead
entire slide resident in GPU memory     5.094s   <- no faster
```

Loading the whole slide into VRAM does not beat the container. There is no
data-path win left; what limits end-to-end is how fast your GPU runs the model.

### For reference: a model-bound machine

The same code on an Apple M5 (MPS), where data was only 37% of the epoch:
15.2x data feeding, **1.31x end-to-end**. Low end-to-end numbers mean there was
little I/O left to remove, not that the container underperformed.

## Reproduce it yourself — one command

```bash
git clone https://github.com/mishel-0/omnia-20x.git
cd omnia-20x
pip install -e .
python -m omnia_sdk.benchmark            # downloads a public slide, measures, writes JSON
```

Or one click, no install, no uploads:

[**Open the auditable benchmark in Google Colab**](https://colab.research.google.com/github/mishel-0/omnia-20x/blob/main/colab/omnia_vs_svs_benchmark.ipynb)

The benchmark writes `benchmarks/benchmark_results.json` with machine info,
package versions, slide SHA-256, per-epoch times (data-only and full-train),
the speedups, and GPU utilization samples. Nothing is hardcoded — every
number is measured on the machine that runs it. See
[docs/VERIFICATION.md](docs/VERIFICATION.md).

## Why it matters, in money terms

Pathology labs train on rented A100s/T4s. If a training pipeline is
data-bound (the common tile-classification case), data feeding is the
idle-GPU tax. Cutting it 10-20x means either fewer GPU-hours per experiment
or the same hours producing proportionally more epochs. This is the same
thesis as the dataset-infrastructure category (e.g. Activeloop/Deep Lake):
the format + loading layer is where training efficiency is won or lost.

## What the .omnia container is

One file replaces .svs + openslide: lossless Zstd tiles, CRC-verified,
zero-copy random access, no openslide dependency at training time.
`OmniaDataset(cache_mode="ram")` decodes everything once at init (~2-5s),
then every epoch is zero-copy tensor views.

## Other honest numbers

- **Compression**: zstd is lossless, ~5x vs raw pixels — the file stays
  LARGER than the .svs, because .svs is already JPEG-2000-compressed ~127x.
  `--codec jpeg --min-level 1` gives ~4.6x smaller than the .svs (lossy,
  fine for training). `svs-native --min-level 1` gives ~7.9x smaller,
  lossless (storage; keeps JP2K bytes).
- **num_workers must be 0** for preloaded data — workers only add IPC
  overhead (measured 1.12x slower). Use `pin_memory=True` on GPU.
- **RAM is the limit, not disk.** `cache_mode="ram"` decodes every tile to
  raw RGB and holds it, so memory scales with tiles retained, not with file
  size. Measured on one 61 MB slide (15,374 x 17,497, 4,569 tiles):

  | Config | .omnia on disk | RAM preloaded | 1,000 slides |
  |---|---|---|---|
  | Lossless zstd, all levels | 542 MB | ~3.3 GB | 529 GB disk · 3.3 TB RAM |
  | `--codec jpeg --min-level 1` | 3.8 MB | ~343 MB | 3.7 GB disk · 335 GB RAM |

  The recommended config is 16x smaller than the .svs on disk, but a
  full-cohort preload still will not fit in RAM. For anything past a few
  dozen slides, use `cache_mode="mmap"`, or preload one slide at a time and
  iterate slides in the outer loop.

- **`cache_mode="none"` is slower than openslide, not faster.** Tiles are
  compressed in batches (default 16), so an uncached random read decompresses
  16 tiles to return 1. Measured on 300 random level-0 tiles: openslide
  2.15 ms/tile vs .omnia 2.49 ms/tile — **0.9x**. The speedup comes from
  preloading or sequential access, not from the container being faster per
  isolated random read:

  | Access pattern | vs openslide |
  |---|---|
  | Random, `cache_mode="none"` | **0.9x** (slower) |
  | Sequential, cold | 8.8x |
  | Sequential, warm batch cache | 139x |
  | Shuffled DataLoader, `cache_mode="ram"` | **87x** |

  The last row is the one that matters for training, and it is the number the
  headline refers to.

## Usage

```bash
# Storage + speed in one: JPEG tiles at 20x, ~4.6x smaller than .svs
python -m omnia_sdk.cli svs-convert slide.svs slide.train.omnia --codec jpeg --quality 85 --min-level 1

# Lossless training format — Zstd RGB
python -m omnia_sdk.cli svs-convert slide.svs slide.train-full.omnia

# Lossless storage — drop the 40x level: ~8x smaller than .svs
python -m omnia_sdk.cli svs-native slide.svs slide.store.omnia --min-level 1

# Verify integrity (CRC check)
python -m omnia_sdk.cli verify slide.omnia

# PyTorch training
from omnia_sdk import OmniaDataset
from torch.utils.data import DataLoader

ds = OmniaDataset("slide.train.omnia", cache_mode="ram")  # preload once
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
for images, labels in loader:
    ...
```

## Package layout

```
omnia-20x/
├── omnia_sdk/
│   ├── __init__.py
│   ├── container.py       # .omnia read/write (lossless Zstd, CRC per tile)
│   ├── dataset.py         # OmniaDataset — RAM/mmap/none cache modes
│   ├── cli.py             # svs-convert / svs-native / verify / info
│   ├── svs_to_omnia.py    # .svs -> .omnia (zstd or jpeg, --min-level)
│   ├── native_convert.py  # JP2K passthrough (lossless storage)
│   └── benchmark.py       # auditable benchmark -> benchmark_results.json
├── benchmarks/
│   ├── run_all.sh         # one-command run
│   └── benchmark_results.json  # generated receipt (gitignored)
├── colab/
│   └── omnia_vs_svs_benchmark.ipynb  # one-click colab verification
├── docs/
│   ├── BENCHMARK.md       # methodology (incl. regime analysis)
│   └── VERIFICATION.md    # skeptic's audit guide
├── requirements.txt
└── pyproject.toml
```

## License

MIT — see LICENSE. The benchmark downloads a public test slide at runtime;
no slide files are shipped in the repo.
