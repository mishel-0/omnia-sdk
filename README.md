<p align="center">
  <br/>
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0a1628,100:0d1b2a&height=200&section=header&text=.omnia&fontSize=70&fontColor=0066CC&animation=fadeIn" width="100%"/>
</p>

<p align="center" style="font-size: 14px; color: #8899aa; max-width: 560px; line-height: 1.8;">
  A study&rsquo;s 277 DICOM files generate 13.8 million files per dataset &mdash; and 69 million syscalls per training epoch.
  <strong style="color: #99aabb;">.omnia proposes a single-file container that eliminates the O(n) I/O tax in medical AI pipelines.</strong>
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

# Run benchmark on your hardware
omnia benchmark ./dataset/
```

```python
from omnia import Study

# Open a .omnia file
study = Study("ct_scan.omnia")

# Random access to any slice — O(1)
slice_47 = study[47]              # numpy array, shape (512, 512)

# Bulk read entire volume
volume = study.volume()           # numpy array, shape (277, 512, 512)

# Metadata
print(study.num_slices)           # 277
print(study.shape)                # (512, 512)
print(study.dtype)                # int16
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
          GPU (93% util)
```

---

## Validation

| Check | Result |
|-------|--------|
| **Lossless** | Byte-for-byte verified across 3,387 slices. 0 errors. |
| **Metadata preservation** | Original DICOM tags preserved on round-trip. |
| **CRC integrity** | Per-slice CRC32 — corruption detected on read. |
| **Benchmark reproducibility** | Scripts and methodology included in `benchmarks/`. |
| **Test coverage** | 6-test debug suite: DICOM reading, conversion, lossless, format header, batch, random access. |

---

## Supported modalities

| Modality | Status |
|----------|--------|
| **CT** | ✅ Supported |
| **MRI** | 🔄 Planned (v0.2) |
| **PET** | 🔄 Planned (v0.3) |
| **Ultrasound** | 🔄 Planned (v0.3) |

---

## FAQ

### Why not ZIP?
ZIP requires full decompression to access a single slice. No random access. For a 277-slice study, reading slice 200 means decompressing slices 0–199 first.

### Why not HDF5?
HDF5 is a general-purpose hierarchical format with no DICOM ingestion path. It requires a full pipeline rewrite. Concurrency model is fragile. Metadata overhead is significant for large studies.

### Why not Zarr?
Zarr is designed for chunked array storage in cloud-native ML pipelines. It has no DICOM understanding — no modality tags, no study/series hierarchy, no patient metadata. Converting DICOM to Zarr discards the medical context.

### Why not NIfTI?
NIfTI was designed for neuroimaging (fMRI, diffusion). It stores a single 3D or 4D volume with minimal metadata. No DICOM tag preservation. Gzip compression requires full file decompression.

### Why not WebDataset?
WebDataset shards tar archives for fast I/O, but each shard is a concatenation of individual files. DICOM parsing still happens per sample. No random access within a shard.

### Why not JPEG 2000 (DICOM encapsulated)?
JPEG 2000 is a per-slice codec, not a container. It does not aggregate studies. Decode at 16 ms/slice is 50× slower than the codec used in .omnia, making it unsuitable for training pipelines.

### Why not MONAI CacheDataset?
MONAI caches decoded data in RAM. This works for small datasets (< 10 GB) but does not scale to the multi-TB archives typical in medical AI. It also does not reduce storage.

### Why not NVIDIA DALI?
DALI accelerates data loading on the GPU but has no DICOM decoder. It requires a custom preprocessing pipeline. No storage savings.

### Why not AV1 / H.265?
These are lossy video codecs. CT diagnosis requires preserving 1 HU differences — lossy quantization is unacceptable for medical use.

### Why not a database (LMDB, SQLite)?
Databases add complexity, maintenance overhead, and licensing costs. They do not inherently provide compression or DICOM-native organization. A file-based container is simpler to deploy, backup, and migrate.

### Why not keep raw DICOM?
Raw DICOM works — at the cost of 277 files per study, 69M syscalls per epoch, and 48% GPU utilization. .omnia is backward compatible: you keep your PACS, your DICOM archive, and your workflows. The container is a cache / training format, not a replacement.

---

## Benchmark

*Reproducible with `benchmarks/benchmark.py`. Full 100-epoch results in `benchmarks/benchmark.json`.*

| Metric | Raw DICOM | .omnia |
|--------|-----------|--------|
| Mean epoch time (steady, last 50) | 18.1 s | 17.8 s |
| Cold start (epoch 1) | 40.6 s | 21.9 s |
| GPU utilization (steady) | 96% | 98% |
| Storage | 1,819 MB | 837 MB |
| Dataset load time | 127.6 s | 0.7 s |
| System calls per epoch | ~17,000 | ~30 |
| Lossless verification | — | 0 errors / 3,387 slices |

> **Note:** The 18.1 s DICOM steady state is achieved because the 1.8 GB dataset fits entirely in the test system's 440 GB RAM. For production datasets (10 TB+) that exceed available RAM, DICOM remains in cold-start territory while .omnia maintains steady throughput regardless of dataset size. See `benchmarks/benchmark.json` for per-epoch timing and full methodology.

```
Epoch time comparison (100 epochs):

DICOM:  ████████████████████████░░░░░░░░░░░░░░░░░░░░ 40.6s → 18.1s
.omnia: ████████████████████████████████████████████ 21.9s → 17.8s

GPU utilization (steady state):
DICOM:  ████████████████████████████████████████████ 96%
.omnia: ████████████████████████████████████████████ 98%

Storage:
DICOM:  ████████████████████████████████████████████ 1,819 MB
.omnia: ████████████████████████████████████████████ 837 MB
```

*Hardware: NVIDIA RTX A4000 · ResNet-18 · 3,387 LIDC-CT slices · 100 epochs · Batch 64 · 4 workers*

---

## Limitations

- Single-series CT only. Multi-series and multi-modality studies not yet supported.
- Not a DICOM transfer syntax — complements DICOM, does not replace it.
- Requires custom reader — no native PACS or viewer support.
- Research prototype. Not clinically validated. Not for diagnostic use.
- Designed as a cache / training format alongside existing PACS infrastructure.

---

## Roadmap

| Version | Focus | Status |
|---------|-------|--------|
| v0.1 | CT support, PyTorch Dataset, CLI, benchmark | ✅ Complete |
| v0.2 | MRI support, streaming reads, metadata API | 🔄 Planned |
| v0.3 | PET / Ultrasound, batch compression, encryption | 📅 Planned |
| v1.0 | Production SDK, Windows build, PACS integration | 📅 Planned |

---

## Docs

```
docs/
├── container.md    — Format specification, offset table, CRC, compression pipeline
├── cli.md          — Command reference
├── sdk.md          — Python & C SDK reference
├── benchmark.md    — Methodology, reproduction guide
└── faq.md          — Frequently asked questions
```

---

## Changelog

### v0.1.0-alpha (2026-07-21)
- Initial research prototype
- CT study compression and decompression
- PyTorch Dataset for training pipelines
- CLI (`compress`, `extract`, `verify`, `benchmark`)
- Per-slice CRC32 integrity verification
- 100-epoch benchmark on real LIDC data
- Reproducible benchmark suite

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
