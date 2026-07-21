<p align="center">
  <br/><br/><br/>
  <span style="font-size: 64px; font-weight: 700; color: #0066CC; letter-spacing: -2px;">.omnia</span>
  <br/><br/>
  <span style="font-size: 18px; color: #8899aa; letter-spacing: 1px;">Medical Image Container for AI</span>
  <br/><br/><br/>
</p>

<p align="center">
  277 DICOM files → <strong>1</strong>. &nbsp; Training <strong>1.87× faster</strong>. &nbsp; GPU <strong>48% → 93%</strong>. &nbsp; Storage <strong>2.17× less</strong>.
</p>

<br/>

<p align="center">
  <a href="https://img.shields.io/badge/training-1.87×_faster-0066CC?style=flat-square&labelColor=0a1628"><img src="https://img.shields.io/badge/training-1.87×_faster-0066CC?style=flat-square&labelColor=0a1628" alt="Training"></a>
  <a href="https://img.shields.io/badge/gpu-93%25-0066CC?style=flat-square&labelColor=0a1628"><img src="https://img.shields.io/badge/gpu-93%25-0066CC?style=flat-square&labelColor=0a1628" alt="GPU"></a>
  <a href="https://img.shields.io/badge/storage-2.17×-0066CC?style=flat-square&labelColor=0a1628"><img src="https://img.shields.io/badge/storage-2.17×-0066CC?style=flat-square&labelColor=0a1628" alt="Storage"></a>
  <a href="https://img.shields.io/badge/license-proprietary-445566?style=flat-square&labelColor=0a1628"><img src="https://img.shields.io/badge/license-proprietary-445566?style=flat-square&labelColor=0a1628" alt="License"></a>
</p>

<br/><br/>

---

<br/>

## Install

```bash
pip install omnia-sdk
```

<br/>

## Usage

```bash
.omnia convert ./ct_scans/ ./compressed/
```

```python
from omnia_sdk.dataset import OmniaDataset
from torch.utils.data import DataLoader

ds = OmniaDataset("./compressed/")
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)

for images, labels in loader:
    out = model(images)
```

<br/>

## Why .omnia is faster

DICOM stores every CT slice as a separate file. Training on 50,000 studies means **13.8 million files**.

Each epoch opens, parses, reads, and closes every file — **69 million syscalls**. The CPU spends more time managing file handles than feeding the GPU. GPU utilization stalls at **48%**.

.omnia bundles each study into one file with a precomputed offset table. Files open once and stay open. A slice is a single seek plus a fast decompress — under a millisecond. GPU utilization reaches **93%**.

<p align="center">
  <br/>
  13.8M files → 50K files &nbsp;·&nbsp; 69M syscalls → 50K syscalls &nbsp;·&nbsp; 127s load → 0.7s load
  <br/><br/>
</p>

<br/>

## Results

| | Raw DICOM | .omnia |
|---|---|---|
| Steady epoch | 40.9 s | **21.9 s** |
| GPU utilization | 48% | **93%** |
| Storage | 1,819 MB | **837 MB** |
| Dataset loading | 127.6 s | **0.7 s** |
| Cold epoch | 215.8 s | **69.3 s** |
| Lossless | — | ✅ Verified |

<sub>ResNet‑18 · 3,387 real CT slices · RTX A4000 · 100 epochs</sub>

<br/>

---

<br/>

<p align="center">
  <span style="font-size: 13px; color: #556677;">Proprietary — All rights reserved.</span>
</p>

<br/><br/><br/>
