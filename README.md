<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://capsule-render.vercel.app/api?type=rounded&color=0:1a1a2e,100:16213e&height=180&section=header&text=.omnia&fontSize=70&fontColor=00d4aa&animation=fadeIn&desc=Medical%20Image%20Container%20for%20AI&descSize=16&descAlignY=60">
    <img src="https://capsule-render.vercel.app/api?type=rounded&color=0:0066CC,100:00AA55&height=180&section=header&text=.omnia&fontSize=70&fontColor=fff&animation=fadeIn&desc=Medical%20Image%20Container%20for%20AI&descSize=16&descAlignY=60">
  </picture>
</p>

<p align="center">
  <a href="https://img.shields.io/badge/training-1.87×_faster-00d4aa?style=flat-square"><img src="https://img.shields.io/badge/training-1.87×_faster-00d4aa?style=flat-square" alt="Training"></a>
  <a href="https://img.shields.io/badge/gpu-93%25_utilization-00d4aa?style=flat-square"><img src="https://img.shields.io/badge/gpu-93%25_utilization-00d4aa?style=flat-square" alt="GPU"></a>
  <a href="https://img.shields.io/badge/storage-2.17×_lossless-00d4aa?style=flat-square"><img src="https://img.shields.io/badge/storage-2.17×_lossless-00d4aa?style=flat-square" alt="Storage"></a>
  <a href="https://img.shields.io/badge/license-proprietary-666?style=flat-square"><img src="https://img.shields.io/badge/license-proprietary-666?style=flat-square" alt="License"></a>
</p>

<p align="center">
  Bundles a CT study into one file.  
  Training runs <strong>1.87× faster</strong>. GPU utilization doubles.  
  Storage drops <strong>2.17×</strong>. Zero pixel loss.
</p>

<br/>

---

## Install

```bash
pip install omnia-sdk
```

---

## Quickstart

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

---

## Benchmarks

| Metric | Raw DICOM | .omnia |
|--------|-----------|--------|
| **Epoch time** | 40.9 s | **21.9 s** |
| **GPU utilization** | 48% | **93%** |
| **Storage (15 studies)** | 1,819 MB | **837 MB** |
| **Dataset loading** | 127.6 s | **0.7 s** |
| **Lossless** | — | ✅ Verified |

<sub>ResNet‑18 · 3,387 real CT slices · NVIDIA RTX A4000 · 100 epochs</sub>

---

## Why

Every CT study is stored as **277 individual DICOM files**. Every training epoch opens and parses all 277 — over 16,000 syscalls, 127 seconds of overhead. The GPU sits idle at **48% utilization** waiting on file I/O.

.omnia bundles each study into **one file** with fast random access. One seek, one read, delivered in under a millisecond. GPU utilization jumps to **93%**. Training finishes in half the time.

These numbers are measured on real patient data. All benchmarks are reproducible.

---

## Problem it solves

277 files per study → **1 file**.  
16,000 syscalls per epoch → **~30 syscalls**.  
127 seconds dataset loading → **0.7 seconds**.  
48% GPU utilization → **93%**.

Same pixels. Same model. Same loss curve. **Half the time.**

---

## License

Proprietary — All rights reserved.  
Contact for SDK access and licensing.
