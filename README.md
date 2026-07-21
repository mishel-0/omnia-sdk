# .omnia SDK — Medical Image Container for AI Training

**One file instead of 277. Lossless. 2x faster training.**

```bash
pip install "omnia-sdk[ml] @ git+https://github.com/mishel-0/omnia-sdk.git"
```

---

## Quick Start — 5 Minutes

### 1. Convert your DICOM studies to .omnia

```bash
python -m omnia_sdk.convert /path/to/raw_dicom/ /path/to/compressed/
```

This takes a folder of DICOM studies like:

```
raw_dicom/
├── LIDC-IDRI-0001/
│   ├── slice_001.dcm
│   ├── slice_002.dcm
│   └── ... (277 files)
├── LIDC-IDRI-0002/
│   └── ...
```

And outputs:

```
compressed/
├── LIDC-IDRI-0001.omnia   ← 1 file replaces 277
├── LIDC-IDRI-0002.omnia
└── conversion_summary.json
```

**Result:** ~2.17x lossless compression. 0 pixel loss. Verified.

### 2. Train your model with .omnia

```python
from omnia_sdk.dataset import OmniaDataset
from torch.utils.data import DataLoader

# Same as DICOM — just point to your .omnia folder
ds = OmniaDataset("/path/to/compressed/")

loader = DataLoader(
    ds,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

for images, labels in loader:
    # images shape: [B, 1, 512, 512]
    # Same pixels as DICOM — identical loss curve
    output = model(images)
    loss = criterion(output, labels)
```

### 3. Compare the difference

```bash
# Benchmark DICOM vs .omnia on your data
python benchmarks/bench_training.py /path/to/raw_dicom/ /path/to/compressed/
```

---

## What changed?

| | Before (DICOM) | After (.omnia) |
|---|---|---|
| Files per study | 277 | **1** |
| Dataset loading | 127 seconds | **0.7 seconds** |
| GPU utilization | 48% | **93%** |
| Epoch time | 40.9s | **21.9s (1.87x faster)** |
| Storage | 1,819 MB | **837 MB (2.17x less)** |
| Pixel quality | — | ✅ Lossless (0 errors) |

## How it works

.omnia bundles all slices of a CT study into a single Zstd-compressed file with a pre-computed offset table. Any slice can be read in O(1) time — seek to offset, decompress 0.3ms, done. No full-file decompression, no header parsing per slice, no repeated open/close system calls.

## File format

```
[OMN2:4][VERSION:1][CODEC:1][RESERVED:2]
[META_SIZE:8][zstd(JSON metadata with offsets, shapes, CRC)]
[CRC:4+zstd(slice_0)] [CRC:4+zstd(slice_1)] ... [CRC:4+zstd(slice_N)]
```

Each slice chunk has its own CRC32 checksum for corruption detection. The format spec is at `FORMAT_SPEC.md` — open, no proprietary encoding.

## Verified on real hardware

| Hardware | Setup | Speedup |
|----------|-------|---------|
| RTX A4000 | 3,387 slices, ResNet-18 | **1.87x** |
| 100 epochs | Full training run | **46% time saved** |
| Cold start | First epoch, no cache | **3.11x faster** |

## Requirements

- Python 3.9+
- `pip install pydicom zstd numpy` (or `pip install "omnia-sdk[ml]"` for PyTorch)

---

Built at **Vilnius Gediminas Technical University (VGTU)** — Vilnius, Lithuania  
License: MIT — free for any use.  
```
