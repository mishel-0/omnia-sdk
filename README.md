# .omnia SDK

**Train 2x faster on CT scans. One file instead of 277. Lossless.**

```bash
pip install "omnia-sdk[ml] @ git+https://github.com/mishel-0/omnia-sdk.git"
```

---

## Usage

**1. Compress your DICOM folder:**

```bash
python -m omnia_sdk.convert ./ct_scans/ ./compressed/
```

**2. Train with it:**

```python
from omnia_sdk.dataset import OmniaDataset
from torch.utils.data import DataLoader

ds = OmniaDataset("./compressed/")
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)

for images, labels in loader:
    out = model(images)  # same pixels, faster
```

**3. Done.** Epochs go from 40s → 22s. GPU goes from 48% → 93%.

---

## Why

Every CT study is stored as 277 files. Opening them all every epoch keeps the GPU waiting. .omnia bundles each study into one Zstd file — one open, one seek, 0.3ms decompress. Your GPU stops waiting.

## Benchmark

| Metric | DICOM | .omnia |
|--------|-------|--------|
| Epoch | 40.9s | **21.9s** |
| GPU util | 48% | **93%** |
| Storage | 1,819 MB | **837 MB** |
| Load time | 127s | **0.7s** |
| Quality | — | ✅ lossless |

*ResNet-18, 3,387 CT slices, RTX A4000. Full results in `benchmarks/`.*

---

Built at **VGTU** — Vilnius, Lithuania · MIT license
