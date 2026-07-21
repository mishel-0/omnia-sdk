# .omnia SDK — Medical Image Container for AI Training

**One file instead of 277. Lossless. 2x faster training.**

```bash
pip install pydicom zstd torch torchvision numpy
python omnia_sdk/convert.py ./raw_dicom/ ./compressed/
```

Then in your training loop:

```python
from omnia_sdk.dataset import OmniaDataset
from torch.utils.data import DataLoader

ds = OmniaDataset("./compressed/")
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)
for images, labels in loader:
    # images: [B, 1, 512, 512] — same pixels, just faster
    ...
```

## Why

Every CT study is stored as **277 separate DICOM files**. Training a model means opening and parsing all 277 files every epoch. The GPU sits idle at **48% utilization** waiting on file I/O.

.omnia bundles each study into **1 file** with Zstd compression and persistent handle access. GPU utilization jumps to **93%**. Training finishes **1.87x faster**.

## Benchmark (ResNet-18, 3,387 CT slices, RTX A4000)

| Metric | Raw DICOM | .omnia Zstd |
|--------|-----------|-------------|
| Epoch time | 40.9s | **21.9s** |
| GPU utilization | 48% | **93%** |
| Dataset load | 127.6s | **0.7s** |
| Storage | 1,819 MB | **837 MB (2.17x)** |
| Lossless | — | ✅ 0 errors / 3,387 slices |

## Format

```
[OMN2:4][VER:1][CODEC:1][RSV:2]
[META_SIZE:8][zstd(JSON metadata)]
[CRC:4+zstd(slice_0)]...[CRC:4+zstd(slice_N)]
```

- Open specification: `FORMAT_SPEC.md`
- Codec: Zstd (ISO standard)
- Lossless: CRC32 verified per slice
- License: MIT

## SDK Structure

```
├── src/
│   ├── container.py      ← .omnia read/write (Zstd, CRC-verified)
│   ├── convert.py         ← DICOM → .omnia batch converter
│   └── dataset.py         ← PyTorch Dataset
├── benchmarks/            ← Reproducible benchmark scripts
├── FORMAT_SPEC.md         ← Open format specification
└── requirements.txt       ← pip install
```

## Quick Start

```bash
# Convert DICOM studies to .omnia
python omnia_sdk/convert.py /path/to/lidc_raw/ /path/to/output/

# Verify conversion
python benchmarks/debug_test.py

# Run benchmark
python benchmarks/bench_resnet.py /path/to/lidc_raw /path/to/output/
```

## License

MIT — free for any use. The format is open. The code is reference. Build your own tools on top.

---

Built at **Vilnius Gediminas Technical University (VGTU)** — Vilnius, Lithuania
