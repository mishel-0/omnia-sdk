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
  <a href="https://img.shields.io/badge/version-0.1.0--alpha-0066CC?style=flat-square&labelColor=0a1628"><img src="https://img.shields.io/badge/version-0.1.0--alpha-0066CC?style=flat-square&labelColor=0a1628" alt="Version"></a>
</p>

---

## Installation

```bash
# Request an evaluation build:
# contact@omnia-sdk.com

# Supported platforms:
# • Linux (Ubuntu 22.04+, RHEL 8+)
# • Windows (Server 2019+, 10+)
# • Python 3.11+

# Once licensed:
pip install omnia-sdk
```

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
from omnia import Study

study = Study("ct_scan.omnia")
slice_47 = study[47]              # O(1) random access
volume = study.volume()           # bulk read
```

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
  │   Offset Table    │  Build index: slice N → byte offset + CRC32
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

## Benchmark

Measured on: NVIDIA RTX A4000 · ResNet-18 · 3,387 real LIDC-CT slices · Batch 64 · 4 workers

Two benchmarks were run:

**A) Truly cold system** (first run after boot, no cached data):

| Metric | Raw DICOM | .omnia |
|--------|-----------|--------|
| Epoch 1 (cold) | 215.8 s | 69.3 s |
| Epoch 2 | 133.6 s | 21.9 s |
| Epoch 3 | 95.6 s | 21.9 s |
| Epoch 4 | 41.0 s | 21.9 s |
| Epoch 5 | 40.9 s | 21.9 s |
| GPU util (avg) | 29% | 80% |

**B) Steady state** (100 epochs, last 50 averaged, fully cached):

| Metric | Raw DICOM | .omnia |
|--------|-----------|--------|
| Mean epoch (last 50) | 18.1 s | 17.8 s |
| GPU util (last 50) | 96% | 98% |
| Storage | 1,819 MB | 837 MB |
| Dataset load time | 127.6 s | 0.7 s |
| Lossless verification | — | 0 errors / 3,387 slices |

> **Why two tables?** The test system has 440 GB RAM. Our 1.8 GB dataset fits entirely in OS page cache after a few epochs. The steady-state numbers show that once everything is cached, both DICOM and .omnia run at similar speed. The cold-start numbers show the real-world difference: when the cache is empty (or the dataset is too large to cache), .omnia avoids the per-file overhead that slows DICOM down. For production datasets exceeding available RAM, the cold-start behavior is the dominant regime.

---

## Validation

| Check | Result |
|-------|--------|
| **Lossless** | Byte-for-byte verified across 3,387 slices. 0 errors. |
| **CRC integrity** | Per-slice CRC32 — corruption detected on read. |
| **Benchmark reproducibility** | Scripts and methodology in `benchmarks/`. |

---

## Supported modalities

| Modality | Status |
|----------|--------|
| **CT** | ✅ Supported |
| **MRI** | 🔄 Planned |
| **PET / Ultrasound** | 🔄 Planned |

---

## FAQ

### Why not ZIP?
ZIP requires decompressing the entire archive to access any single slice. Reading slice 200 means decompressing slices 0-199 first.

### Why not HDF5?
HDF5 has no DICOM ingestion path and its concurrency model is fragile under high worker counts.

### Why not Zarr?
Zarr has no DICOM understanding — no modality tags, no study hierarchy, no patient metadata.

### Why not NIfTI?
NIfTI stores a single volume with minimal metadata. No DICOM tag preservation. Gzip requires full decompression.

### Why not WebDataset?
WebDataset shards tar files but still requires DICOM parsing per sample. No random access within a shard.

### Why not JPEG 2000?
JPEG 2000 per-slice decode is ~16 ms, too slow for training pipelines.

### Why not MONAI CacheDataset?
Caches in RAM — works for datasets under 10 GB but does not scale to multi-TB archives.

### Why not NVIDIA DALI?
DALI has no DICOM decoder and requires custom preprocessing operators.

### Why not AV1 / H.265?
Lossy — not acceptable for diagnostic CT where 1 HU differences matter.

---

## Limitations

- Single-series CT only. Multi-series and multi-modality not yet supported.
- Not a DICOM transfer syntax — complements DICOM, does not replace it.
- Requires custom reader — no native PACS or viewer support.
- Research prototype. Not clinically validated. Not for diagnostic use.
- Designed as a cache / training format alongside existing PACS infrastructure.

---

## Roadmap

| Version | Focus | Status |
|---------|-------|--------|
| v0.1 | CT support, PyTorch Dataset, CLI, benchmark | ✅ Complete |
| v0.2 | MRI support, streaming reads | 🔄 Planned |
| v0.3 | PET / Ultrasound, batch compression | 📅 Planned |
| v1.0 | Production SDK, Windows build | 📅 Planned |

---

## Docs

```
docs/
├── container.md    — Format specification, offset table, CRC
├── cli.md          — Command reference
├── sdk.md          — Python & C SDK reference
├── benchmark.md    — Methodology, reproduction guide
└── faq.md          — Frequently asked questions
```

---

## Changelog

### v0.1.0-alpha (2026-07-21)
- CT study compression and decompression
- PyTorch Dataset for training pipelines
- CLI (compress, extract, verify)
- Per-slice CRC32 integrity verification
- Benchmark suite on real LIDC data

---

## License & Contact

**Proprietary** — All rights reserved.

```
Commercial licensing / Enterprise evaluation:
contact@omnia-sdk.com
```

---

<p align="center">
  <span style="font-size: 11px; color: #445566;">© 2026</span>
</p>
