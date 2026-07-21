<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=rounded&color=0:0a0a0f,100:1a1a2e&height=200&section=header&text=.omnia&fontSize=72&fontColor=00d4aa&animation=fadeIn&desc=Medical%20Image%20Container%20for%20AI&descSize=15&descAlignY=62" width="100%"/>
</p>

<p align="center">
  <a href="https://img.shields.io/badge/training-1.87×_faster-00d4aa?style=flat-square&labelColor=1a1a2e"><img src="https://img.shields.io/badge/training-1.87×_faster-00d4aa?style=flat-square&labelColor=1a1a2e" alt="Training"></a>
  <a href="https://img.shields.io/badge/gpu-93%25_utilization-00d4aa?style=flat-square&labelColor=1a1a2e"><img src="https://img.shields.io/badge/gpu-93%25_utilization-00d4aa?style=flat-square&labelColor=1a1a2e" alt="GPU"></a>
  <a href="https://img.shields.io/badge/storage-2.17×_lossless-00d4aa?style=flat-square&labelColor=1a1a2e"><img src="https://img.shields.io/badge/storage-2.17×_lossless-00d4aa?style=flat-square&labelColor=1a1a2e" alt="Storage"></a>
  <a href="https://img.shields.io/badge/license-proprietary-555?style=flat-square&labelColor=1a1a2e"><img src="https://img.shields.io/badge/license-proprietary-555?style=flat-square&labelColor=1a1a2e" alt="License"></a>
</p>

<br/>

**.omnia** replaces 277 DICOM files with a single container designed for fast random access. Training runs **1.87× faster**. GPU utilization goes from **48% to 93%**. Storage drops **2.17×** — all lossless.

<br/>

---

## Install

```bash
pip install omnia-sdk
```

---

## Usage

```bash
# Convert your DICOM dataset
.omnia convert ./ct_scans/ ./compressed/
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

## Why it's faster

Standard DICOM stores each CT slice as a separate file. Training on 50,000 studies means managing **13.8 million files**. Every epoch does:

```
→ open()      × 13,800,000  (syscall)
→ stat()      × 13,800,000  (syscall)
→ parse DICOM × 13,800,000  (header traversal)
→ read pixels × 13,800,000  (disk I/O)
→ close()     × 13,800,000  (syscall)
```

That's **69 million syscalls per epoch**. The GPU starves at **48% utilization** while the CPU fights file metadata.

**.omnia** stores all slices of a study in one file with a precomputed offset table. Each epoch does:

```
→ open()      × 50,000      (one per study)
→ seek+read   × 3,800       (per batch, O(1) per slice)
→ close()     × 0           (handles stay open)
```

System calls drop from **69 million to ~50,000**. The GPU stays fed at **93% utilization**. Training finishes in half the time.

<br/>

---

## What it solves

| Problem | With DICOM | With .omnia |
|---------|-----------|-------------|
| **File operations per epoch** | 13,800,000 opens + closes | ~50,000 persistent handles |
| **Dataset loading** | 127 seconds (walking 3,387 files) | 0.7 seconds (15 files) |
| **GPU utilization** | 48% (waiting on I/O) | 93% (fed) |
| **Storage** | 1,819 MB (raw DICOM) | 837 MB (lossless) |
| **Cold start (epoch 1)** | 215 seconds | 69 seconds |
| **Steady state** | 40.9 seconds/epoch | 21.9 seconds/epoch |
| **Backup (50M files)** | 3 days | 1 hour |
| **Database rows** | 13.8 billion | 50 million |

<br/>

---

## Benchmark

| Metric | Raw DICOM | .omnia |
|--------|-----------|--------|
| **Steady epoch time** | 40.9 s | **21.9 s** |
| **GPU utilization** | 48% | **93%** |
| **Storage (15 studies)** | 1,819 MB | **837 MB** |
| **Dataset loading** | 127.6 s | **0.7 s** |
| **Cold epoch** | 215.8 s | **69.3 s** |
| **Lossless** | — | ✅ CRC-verified |

<sub>ResNet‑18 · 3,387 real CT slices · NVIDIA RTX A4000 · 100 epochs</sub>

---

## License

Proprietary — All rights reserved.  
Contact for SDK access and licensing.
