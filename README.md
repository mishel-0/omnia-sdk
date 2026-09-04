# omnia-sdk

**The data layer for tile-based whole-slide-image training.**
**2.5x faster than NVIDIA cuCIM at loading a cohort into RAM, 2.9x smaller on
disk — and we show you exactly where it wins and where it loses.**

## The honest headline

We don't sell magic. We sell fewer idle GPU-hours for the common case in
tile-based classification (Gleason grading, detection, PANDA-style models on
ResNet/EfficientNet-class architectures). On that workload, **97.6% of an
openslide epoch is spent decoding JPEG-2000 rather than training** — against
10.2% once the slide is an `.omnia` container. Both figures are from the T4
run in the table below.

That share of time is what the format removes. How much of it converts into
GPU utilisation depends on your model and your machine, and the committed
`benchmarks/benchmark_results.json` was produced on a CPU-only box, so it
carries no utilisation samples to point at. Run
[the notebook](colab/omnia_vs_svs_benchmark.ipynb) on a GPU if you want that
number for your own hardware — it is the honest way to get one.

The multipliers in the table below are **against openslide**, because that is
what most pipelines still use. If you already run cuCIM your baseline is far
higher and the honest number is **2.5x**, not 15-25x — see
[Against cuCIM](#against-cucim-not-just-openslide). The large numbers here are
real but they measure the distance from the slowest common starting point, not
from the best available alternative.

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
| **Epoch time (AMP + `channels_last`)** | 20.20s | **2.60s** | **7.76x** |
| Data loading | 18.46s (97.6%) | 0.52s (10.2%) | **35.2x** |
| On disk | 177.6 MB | 61.1 MB | 2.9x smaller |
| Preload (one-off) | — | 1.0s | repaid in 0.1 epochs |

### Against cuCIM, not just openslide

openslide is the wrong baseline to judge this on. Anyone doing serious WSI
training already uses [cuCIM](https://github.com/rapidsai/cucim), NVIDIA's
GPU-accelerated image I/O library. It is free, actively developed, and reads
`.svs` directly. So that is the comparison that matters.

**Filling RAM with a whole slide** — 1,408 tiles at level 1, Tesla T4:

| | Time | Per tile | |
|---|---|---|---|
| openslide | 14.74s | 10.47 ms | 1.0x |
| **cuCIM** | 2.46s | 1.75 ms | 6.0x |
| **omnia-sdk** | **0.99s** | **0.70 ms** | **14.9x** |

**2.5x faster than cuCIM**, from a file 2.9x smaller on disk (177.6 MB `.svs`
-> 61.1 MB `.omnia`). This is the number to judge the project by.

#### Where cuCIM wins

Single random tiles, no preloading:

| | Per tile |
|---|---|
| cuCIM | **1.66 ms** |
| omnia-sdk (uncached) | 1.92 ms |

**cuCIM is 15% faster here.** Tiles are stored in batches of 16, so an uncached
random read decompresses 16 tiles to return one. If your workload is sparse
random access rather than epoch-style iteration over a cohort, cuCIM is the
better tool and this is documented so you can make that call before adopting
anything.

#### What has not been tested

cuCIM's `device="cuda"` path uses GPUDirect Storage, which is **not available on
Colab** — it failed with `cuFileHandleRegister ... internal error` and fell back
to compatibility mode, so the 14.26 ms/tile it recorded is meaningless. On real
NVIDIA hardware with GPUDirect working, cuCIM may be considerably faster than
measured here. Treat the 2.5x as an upper bound until someone reproduces it on
a DGX-class machine.

### End-to-end depends on your training config, not just the container

`.svs` is **data-bound** — 97.6% of its epoch is JPEG-2000 decoding. `.omnia` is
**compute-bound**. So every improvement to model throughput lands entirely on the
`.omnia` side, and the gap widens:

| Training config (T4, ResNet-18) | `.svs` | `.omnia` | End-to-end |
|---|---|---|---|
| fp32 | 18.91s | 5.17s | **3.66x** |
| AMP + `channels_last` | 20.20s | 2.60s | **7.76x** |

Both rows measured head-to-head in a single run, 4 epochs, epoch 1 discarded.

Note the `.svs` column: enabling AMP made it **slower**, 18.91s to 20.20s. Mixed
precision adds autocast and GradScaler overhead, and on a pipeline that spends
97.6% of its time decoding there is no compute to accelerate in return — so you
pay the cost and collect nothing. Optimising your model actively penalises the
`.svs` path while rewarding `.omnia`.

That asymmetry is the point. This is not a fixed multiplier applied to your
training time; it is the removal of a fixed cost. The faster your model gets,
the larger the ratio becomes.

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
git clone https://github.com/mishel-0/omnia-sdk.git
cd omnia-sdk
pip install -e .
python -m omnia_sdk.benchmark            # downloads a public slide, measures, writes JSON
```

Or one click, no install, no uploads:

[**Open the auditable benchmark in Google Colab**](https://colab.research.google.com/github/mishel-0/omnia-sdk/blob/main/colab/omnia_vs_svs_benchmark.ipynb)

The benchmark writes `benchmarks/benchmark_results.json` with machine info,
package versions, slide SHA-256, per-epoch times (data-only and full-train),
the speedups, and — on a machine with a GPU — utilisation samples. Nothing is
hardcoded: every number is measured on the machine that runs it, and the file
records which of its own claims that machine could and could not support. The
committed one is a CPU run, so its utilisation and end-to-end fields are
empty. See
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

  The last row is the one that matters for training. Note again that these are
  against openslide; measured against cuCIM the uncached case is 0.87x — cuCIM
  is the faster reader for sparse random access.

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
omnia-sdk/
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
