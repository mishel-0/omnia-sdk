<div align="center">
  <br/>
  <h1 align="center">.omnia</h1>
  <p align="center"><b>Medical Image Container for AI</b></p>
  <br/>
</div>

<p align="center">
  <a href="https://img.shields.io/badge/license-MIT-blue">
    <img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="MIT">
  </a>
  <a href="https://img.shields.io/badge/status-alpha-orange">
    <img src="https://img.shields.io/badge/status-alpha-orange?style=flat-square" alt="Alpha">
  </a>
  <a href="https://img.shields.io/badge/GPU-93%25%20util-brightgreen">
    <img src="https://img.shields.io/badge/GPU-93%25%20util-brightgreen?style=flat-square" alt="GPU">
  </a>
  <a href="https://img.shields.io/badge/training-1.87x%20faster-success">
    <img src="https://img.shields.io/badge/training-1.87x%20faster-success?style=flat-square" alt="Training">
  </a>
  <a href="https://img.shields.io/badge/storage-2.17x%20lossless-success">
    <img src="https://img.shields.io/badge/storage-2.17x%20lossless-success?style=flat-square" alt="Storage">
  </a>
</p>

<br/>

> **One file instead of 277.**  
> .omnia bundles a CT study into a single compressed container.  
> AI training runs 1.87x faster. GPU utilization doubles. Storage drops 2.17x.  
> Zero pixel loss. Same model weights. No pipeline changes.

<br/>

---

## Benchmark

ResNet-18 · 3,387 real CT slices · NVIDIA RTX A4000 · 100 epochs

<div align="center">

| Metric | Raw DICOM | .omnia |
|--------|-----------|--------|
| **Epoch time** | 40.9 s | **21.9 s** |
| **GPU utilization** | 48% | **93%** |
| **Storage (15 studies)** | 1,819 MB | **837 MB** |
| **Dataset loading** | 127.6 s | **0.7 s** |
| **Lossless** | — | ✅ CRC-verified |

</div>

---

## Why

Every CT study today is stored as **277 individual DICOM files**. Every AI training epoch opens and parses all 277 files. The GPU sits idle at **48% utilization** while the CPU fights the filesystem.

.omnia eliminates the I/O bottleneck at the storage layer — one seek, one read, delivered. GPU utilization jumps to **93%**. Training finishes in half the time.

These numbers are measured on real patient data from the LIDC-IDRI dataset on production hardware. All benchmarks are reproducible and documented.

---

## For ML Engineers

The SDK is available as a Python package.

```bash
pip install omnia-sdk
```

```python
from omnia_sdk.dataset import OmniaDataset
from torch.utils.data import DataLoader

ds = OmniaDataset("./compressed/")
loader = DataLoader(ds, batch_size=64, shuffle=True)
for images, labels in loader:
    out = model(images)  # same pixels, 2x faster
```

The C SDK for PACS integration is available under license.

---

## For PACS Vendors

Embed .omnia into your PACS backend and eliminate the O(n) tax at every layer:

- **Filesystem inodes:** 50 million files → 180 thousand
- **Backup windows:** 3 days → 1 hour
- **Database indexes:** billions of rows → millions
- **Ingestion atomicity:** 1 write per study — no orphan cleanup

Six functions, one header file, 50 lines of integration.  
Compiled binary — no source code required.

[Contact us](#) for SDK licensing.

---

## Research

Built at **Vilnius Gediminas Technical University (VGTU).**  
Vilnius, Lithuania.

---

<div align="center">
  <sub>
    Vilnius, Lithuania 🇱🇹
  </sub>
  <br/>
  <sub>© 2026 Mishel Adnan. All rights reserved.</sub>
</div>
