# .omnia SDK

**Train 2x faster on CT scans. One file instead of 277.**

```bash
pip install "omnia-sdk[ml] @ git+https://github.com/mishel-0/omnia-sdk.git"
```

---

## Usage

**1. Compress:**

```bash
omnia compress ./ct_scans/ ./compressed/
```

**2. Train:**

```python
from omnia_sdk.dataset import OmniaDataset
from torch.utils.data import DataLoader

ds = OmniaDataset("./compressed/")
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)
for images, labels in loader:
    out = model(images)  # same pixels, 2x faster
```

**3. Done.** Epochs go 40s → 22s. GPU 48% → 93%.

---

## CLI

```bash
# Convert DICOM to .omnia
omnia compress ./ct_scans/ ./compressed/

# Show file info
omnia info ./study.omnia

# Verify all slices (CRC check)
omnia verify ./study.omnia
```

## Why

277 files per study → 1 file. Persistent handle, no open/close per slice. GPU stops waiting.

## Benchmark

| Metric | DICOM | .omnia |
|--------|-------|--------|
| Epoch | 40.9s | **21.9s** |
| GPU util | 48% | **93%** |
| Storage | 1,819 MB | **837 MB** |
| Load time | 127s | **0.7s** |

*ResNet-18, 3,387 CT slices, RTX A4000. 100 epochs verified.*

---

VGTU · Vilnius · MIT
