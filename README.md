# .omnia SDK v1.01 — Training Edition

Fast, lossless medical image containers for ML training.

**2.17x storage savings. 1.86x faster cold starts. Verified on 3,387 real CT slices.**

---

## Quick Start

```bash
pip install omnia-sdk

# Convert DICOM to .omnia
omnia compress /path/to/raw_dicom/ /path/to/output/

# Verify integrity
omnia verify /path/to/study.omnia

# Extract back to DICOM
omnia extract /path/to/study.omnia /path/to/restored/

# Train with PyTorch
from omnia_sdk import OmniaDataset
from torch.utils.data import DataLoader

ds = OmniaDataset("/path/to/omnia/files/")
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)
```

---

## Benchmark Results (RTX A4000, 3,387 LIDC CT slices, ResNet-18)

### Cold Start (Cache Empty — Real-World Scenario)

| Metric | DICOM | .omnia Zstd | Improvement |
|--------|-------|-------------|-------------|
| Epoch 1 | 40.6s | **21.9s** | **1.86x faster** |
| GPU util | 56.7% | **83.3%** | **+26.6 pp** |

### Steady State (OS Cache Warm — Best Case)

| Metric | DICOM | .omnia Zstd | Improvement |
|--------|-------|-------------|-------------|
| Epoch (avg 51-100) | 18.1s | **17.8s** | **1.02x faster** |
| GPU util | 96.4% | **98.3%** | **+1.9 pp** |
| Std dev | 0.064s | **0.037s** | **1.74x more stable** |

### Storage

| Metric | DICOM | .omnia | Improvement |
|--------|-------|--------|-------------|
| Size | 1,819 MB | **837 MB** | **2.17x smaller** |
| Files | 3,663 | **15** | **244x fewer** |

> **Note:** Cold-start numbers represent real-world behavior for datasets that do not fit in RAM. Steady-state numbers show the best-case scenario when the entire dataset is cached by the OS. For production datasets exceeding available RAM, cold-start behavior dominates.

---

## Container Format

```
[OMN2 magic:4] [version:1] [codec:1] [reserved:2]
[meta_size:8] [zstd(JSON metadata)]
[CRC:4+zstd(slice_0)] [CRC:4+zstd(slice_1)] ... [CRC:4+zstd(slice_N)]
```

- **Magic:** `OMN2` (0x324E4D4F)
- **Version:** 3 (Training Edition)
- **Codec:** 0 = Zstd (fast decode), 1 = JPEG 2000 (higher compression)
- **Integrity:** Per-slice CRC32 — corruption detected on read
- **Access:** O(1) random access via precomputed offset table

---

## Project Structure

```
omnia-sdk/
├── src/omnia_sdk/
│   ├── __init__.py       # Package exports
│   ├── container.py      # Read/write .omnia containers
│   ├── dataset.py        # PyTorch Dataset (DICOM + .omnia)
│   ├── convert.py        # Batch DICOM → .omnia converter
│   └── cli.py            # Command-line interface
├── tests/
│   └── test_omnia_sdk.py # Comprehensive test suite
├── benchmarks/
│   └── benchmark.py      # Fixed benchmark script
├── docs/
│   └── FORMAT_SPEC.md    # Binary format specification
├── pyproject.toml        # Package configuration
└── README.md             # This file
```

---

## Installation

```bash
# From PyPI (when published)
pip install omnia-sdk

# From source
git clone https://github.com/mishel-0/omnia-sdk.git
cd omnia-sdk
pip install -e ".[dev]"
```

---

## Testing

```bash
pytest tests/ -v
```

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Contact

Mishel Adnan — misheladnan35@gmail.com
