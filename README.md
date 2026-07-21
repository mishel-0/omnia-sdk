<p align="center">
  <br/>
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0a1628,100:0d1b2a&height=200&section=header&text=.omnia&fontSize=70&fontColor=0066CC&animation=fadeIn" width="100%"/>
</p>

<p align="center" style="font-size: 14px; color: #8899aa; max-width: 560px; line-height: 1.8;">
  A study&rsquo;s 277 DICOM files generate 3,387 files per dataset we tested. Each epoch opens, parses, reads, and closes every file.
  <strong style="color: #99aabb;">.omnia proposes a single-file container that eliminates this per-file overhead in medical AI pipelines.</strong>
</p>

<p align="center">
  <a href="https://img.shields.io/badge/status-research--prototype-0066CC?style=flat-square&labelColor=0a1628"><img src="https://img.shields.io/badge/status-research--prototype-0066CC?style=flat-square&labelColor=0a1628" alt="Status"></a>
  <a href="https://img.shields.io/badge/license-proprietary-445566?style=flat-square&labelColor=0a1628"><img src="https://img.shields.io/badge/license-proprietary-445566?style=flat-square&labelColor=0a1628" alt="License"></a>
  <a href="https://img.shields.io/badge/version-1.1.2-0066CC?style=flat-square&labelColor=0a1628"><img src="https://img.shields.io/badge/version-1.1.2-0066CC?style=flat-square&labelColor=0a1628" alt="Version"></a>
</p>

---

## Installation

```bash
pip install https://github.com/mishel-0/omnia-sdk/releases/download/v1.1.2/omnia_sdk-1.1.2-py3-none-any.whl
```

## License

A license key is required. Generate one instantly (no account, no email):

```bash
# Download the key generator
curl -O https://raw.githubusercontent.com/mishel-0/omnia-sdk/main/gen_key.py

# Generate a key valid for 365 days
python3 gen_key.py

# Activate
mkdir -p ~/.omnia && echo "your-key-here" > ~/.omnia/license.key
```

The key expires 365 days from the date it was generated. After expiry, run `gen_key.py` again for a new key.

For commercial licensing: **misheladnan35@gmail.com**

---

## Quickstart

```bash
# Compress a DICOM study
omnia compress ./ct_study/ ./output/

# Extract back to DICOM
omnia extract ./study.omnia ./restored/

# Verify integrity (CRC check on all slices)
omnia verify ./study.omnia
```

```python
from omnia_sdk import OmniaContainer

# Open a .omnia file
study = OmniaContainer("ct_scan.omnia")
study.open()

# Random access to any slice — O(1)
slice_47 = study.get_slice(47)

# Bulk read
for i in range(study.num_slices):
    vol = study.get_slice(i)

study.close()
```

```python
# PyTorch training (requires torch)
from omnia_sdk.dataset import OmniaDataset
from torch.utils.data import DataLoader

ds = OmniaDataset("/path/to/compressed/")
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)
for images, labels in loader:
    out = model(images)
```

---

## The problem

Medical imaging built its storage architecture around DICOM in 1993. The standard made sense at the time — each CT slice was a separate file, networks were slow, and studies had 20 slices.

Three decades later, a single CT study routinely produces **277 files**. A training dataset of 50,000 studies represents **13.8 million files**. Every epoch of training opens, stats, reads, and closes every file:

```
for each of 13.8M files:
    stat()      → metadata lookup       (syscall)
    open()      → file descriptor       (syscall)
    parse()     → DICOM header          (CPU)
    read()      → pixel data            (I/O)
    close()     → release descriptor    (syscall)
```

That's **69 million syscalls per epoch** — before a single pixel reaches the GPU. The operating system spends more time resolving file paths and managing descriptors than transferring data. GPU utilization stalls at **48%**.

---

## The insight

The I/O tax exists because the industry accepted a 1:1 mapping between slices and files. There is no technical reason for it. A CT study is a single logical volume — 277 slices that form a contiguous 3D block. Storing them as 277 independent files is a historical artifact.

> **277 files is not a feature of the data. It is a limitation of the format.**

Collapsing 277 files into one eliminates the I/O tax at every layer: filesystem metadata, system calls, DICOM header parsing, and storage allocation.

---

## Pipeline

```
  CT Study (277 DICOM files)
         │
         ▼
  ┌──────────────────┐
  │     Indexer       │  Scan DICOM headers, collect metadata
  └────────┬─────────┘
           │
  ┌────────▼─────────┐
  │  Chunk Compressor │  Compress each slice independently
  └────────┬─────────┘
           │
  ┌────────▼─────────┐
  │   Offset Table    │  Build O(1) index: slice N → byte offset + CRC32
  └────────┬─────────┘
           │
  ┌────────▼─────────┐
  │ Container Writer  │  Serialize header + table + chunks → 1 file
  └────────┬─────────┘
           │
           ▼
     study.omnia (1 file)
           │
           ▼
  ┌──────────────────┐
  │ PyTorch DataLoader │  O(1) random access, persistent handles
  └────────┬─────────┘
           │
           ▼
          GPU
```

---

## Container format

```
[OMN2 magic:4] [version:1] [codec:1] [reserved:2]
[meta_size:8] [zstd(JSON metadata)]
[CRC:4+zstd(slice_0)] [CRC:4+zstd(slice_1)] ... [CRC:4+zstd(slice_N)]
```

Each slice independently addressable via the offset table. Accessing slice 147 does not require decompressing slices 0-146. No full-file extraction. CRC32 per chunk for corruption detection.

---

## Benchmark

Measured on: NVIDIA RTX A4000 · ResNet-18 · 3,387 real LIDC-CT slices · Batch 64 · 4 workers

### GPU utilization

```
Epoch  DICOM  .omnia
─────────────────────
 1      ████░░░░░  9%   ████████████░░ 31%
 2      ███████░░ 17%   ████████████████████████████████░░ 89%
 3      ██████████ 24%   ████████████████████████████████████████ 95%
 4      ██████████████████ 44%   ████████████████████████████████████████ 95%
 5      ████████████████████ 52%   ██████████████████████████████████████ 91%
```

DICOM averages **29% GPU utilization**. .omnia averages **80% GPU utilization**.
The GPU feeds on DICOM's idle time — 3,387 file opens per epoch keep the CPU busy, not the GPU.

### Cold start (cache empty)

| Epoch | Raw DICOM | .omnia |
|-------|-----------|--------|
| 1 | 215.8 s | 69.3 s |
| 2 | 133.6 s | 21.9 s |
| 3 | 95.6 s | 21.9 s |
| 4 | 41.0 s | 21.9 s |
| 5 | 40.9 s | 21.9 s |

### Steady state (100 epochs, last 50 averaged)

| Metric | Raw DICOM | .omnia |
|--------|-----------|--------|
| Mean epoch | 18.1 s | 17.8 s |
| GPU utilization | 96% | 98% |
| Storage | 1,819 MB | 837 MB |
| Dataset load | 127.6 s | 0.7 s |
| Lossless | — | 0 errors / 3,387 slices |

> **Why two tables?** The test system has 440 GB RAM. Our 1.8 GB dataset fits entirely in OS page cache after a few epochs. The steady-state numbers show that once everything is cached, both run at similar speed. The cold-start numbers show the real-world difference when the cache is empty. For production datasets exceeding available RAM, cold-start behavior is the dominant regime.

---

## What it eliminates

| Layer | Conventional DICOM | .omnia |
|-------|-------------------|--------|
| **Filesystem metadata** | 13.8M inodes | 50K inodes |
| **System calls per epoch** | ~17,000 | ~30 |
| **DICOM parsing** | Per-slice header traversal | Once per container |
| **File descriptors** | 3,387 per epoch | 15, opened once |
| **Storage overhead** | Repeated DICOM headers × 277 | Single minimal header |
| **Ingestion atomicity** | 277 writes — partial on crash | 1 write — atomic |
| **Backup** | 50M files → 3 days | 50K files → 1 hour |

---

## Comparison

| Format | Random access | Lossless | Single file | DICOM-aware | Per-slice CRC |
|--------|:---:|:---:|:---:|:---:|:---:|
| Raw DICOM | ✅ | ✅ | ❌ | ✅ | ❌ |
| ZIP | ❌ | ✅ | ✅ | ❌ | ❌ |
| tar.gz | ❌ | ✅ | ✅ | ❌ | ❌ |
| Multi-page TIFF | O(n) | ✅ | ✅ | ❌ | ❌ |
| NIfTI (.nii) | ❌ | ✅ | ✅ | ❌ | ❌ |
| H.265 / AV1 | ❌ | ❌ | ✅ | ❌ | ❌ |
| **.omnia** | **O(1)** | **✅** | **✅** | **✅** | **✅** |

---

## Limitations

- Single-series CT only. Multi-series and multi-modality not yet supported.
- Not a DICOM transfer syntax — complements DICOM, does not replace it.
- Requires custom reader — no native PACS or viewer support.
- Research prototype. Not clinically validated. Not for diagnostic use.
- Designed as a cache / training format alongside existing PACS infrastructure.

---

## License & Contact

**Proprietary** — All rights reserved. License key required.

```
misheladnan35@gmail.com
```

---

<p align="center">
  <span style="font-size: 11px; color: #445566;">© 2026</span>
</p>
