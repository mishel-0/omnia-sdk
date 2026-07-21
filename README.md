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
> .omnia bundles a CT study into a single Zstd-compressed container.  
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

## Architecture

.omnia collapses the 277-file DICOM study into a single container with O(1) random access:

```
┌─ .omnia container ──────────────────────────────┐
│ [OMN2] [v3] [zstd] [reserved]                    │
│ [offset table → slice 0, slice 1, ..., slice N]  │
│ [CRC + zstd(slice_0)] [CRC + zstd(slice_1)] ...  │
└──────────────────────────────────────────────────┘
```

Any slice is accessible in a single seek + 0.3 ms decompress.  
No full-file extraction. No DICOM header parsing per slice. No repeated open/close.

The format is open — see [`FORMAT_SPEC.md`](./FORMAT_SPEC.md).

---

## Why

Every CT study today is stored as **277 individual DICOM files**. Every AI training epoch opens and parses all 277 files. The GPU sits idle at **48% utilization** while the CPU fights the filesystem.

.omnia eliminates the I/O bottleneck at the storage layer — one open, one seek, 0.3 ms decompress. GPU utilization jumps to **93%**. Training finishes in half the time.

This is not synthetic. These numbers are measured on real patient data from the LIDC-IDRI dataset on production hardware.

---

## For ML Engineers

```bash
pip install omnia-sdk
```

```python
from omnia_sdk.dataset import OmniaDataset
from torch.utils.data import DataLoader

ds = OmniaDataset("./compressed/")
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)

for images, labels in loader:
    out = model(images)  # same pixels, 2x faster
```

The SDK is available as a Python package.  
The C SDK for PACS integration is available under license.

---

## For PACS Vendors

Embed .omnia into your PACS backend:

- **6 functions, 1 header file** — `omnia_open`, `omnia_get_slice`, `omnia_get_volume`, `omnia_get_tag`, `omnia_get_info`, `omnia_close`
- **50 lines of integration code**
- **Compiled binary** — no source code required
- **Eliminates inode exhaustion, backup windows, stat() storms**

[Contact us](#) for SDK licensing.

---

## Research

Built at **Vilnius Gediminas Technical University (VGTU)**.  
Vilnius, Lithuania.

The format spec is open (MIT). The Python SDK is open (MIT).  
The C SDK and enterprise features are proprietary.

---

<div align="center">
  <sub>
    <a href="./FORMAT_SPEC.md">Format Spec</a> · 
    <a href="./LICENSE">MIT License</a> · 
    Vilnius, Lithuania 🇱🇹
  </sub>
  <br/>
  <sub>© 2026 Mishel Adnan</sub>
</div>
