# .omnia

Bundles a CT study into one file.  
AI training runs **1.87× faster**. GPU utilization **48% → 93%**.  
Storage drops **2.17×**. Zero pixel loss.

---

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Install

```
pip install omnia-sdk
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Usage

```bash
# Convert your DICOM dataset
omnia convert ./ct_scans/ ./compressed/
```

```python
from omnia_sdk.dataset import OmniaDataset
from torch.utils.data import DataLoader

ds = OmniaDataset("./compressed/")
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)

for images, labels in loader:
    out = model(images)  # same pixels, 2× faster
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Benchmark

| | DICOM | .omnia |
|---|---|---|
| Epoch | 40.9s | **21.9s** |
| GPU util | 48% | **93%** |
| Storage | 1,819 MB | **837 MB** |
| Load | 127s | **0.7s** |
| Quality | — | ✅ lossless |

*ResNet‑18 · 3,387 real CT slices · RTX A4000 · 100 epochs*

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Problem

Every CT study is 277 files. Each training epoch opens and parses all 277 — 16,000+ syscalls, 127 seconds of overhead. Your GPU waits.

**.omnia** bundles each study into one file. One seek, one read, 0.7 seconds. Your GPU stays fed.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## License

Proprietary. All rights reserved.  
Contact for SDK access.
