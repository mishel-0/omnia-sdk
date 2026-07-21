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
</p>

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

That's **69 million syscalls per epoch** — before a single pixel reaches the GPU. GPU utilization stalls at **48%**.

---

## The insight

The I/O tax exists because the industry accepted a 1:1 mapping between slices and files. There is no technical reason for it. A CT study is a single logical volume — 277 slices that form a contiguous 3D block. Storing them as 277 independent files is a historical artifact.

> **277 files is not a feature of the data. It is a limitation of the format.**

---

## Architecture

```
DICOM study (277 files)
        │
        ▼
   ┌─────────────────────┐
   │   Indexer            │  Reads DICOM headers, extracts pixel data,
   │                      │  collects shape, dtype, spacing metadata
   └─────────┬───────────┘
             │
             ▼
   ┌─────────────────────┐
   │   Chunk compressor   │  Compresses each slice independently
   │                      │  using a fast lossless codec
   └─────────┬───────────┘
             │
             ▼
   ┌─────────────────────┐
   │   Offset table       │  Builds O(1) index: slice N → byte offset
   │                      │  + CRC32 checksum per chunk
   └─────────┬───────────┘
             │
             ▼
   ┌─────────────────────┐
   │   Container writer   │  Serializes header, offset table,
   │                      │  and compressed chunks into one file
   └─────────────────────┘
              │
              ▼
        study.omnia (1 file)
```

At read time, the offset table is loaded once. Accessing slice 147 is a single seek + decompress — no other slices touched, no DICOM parsing, no file open/close.

---

## API

```python
from omnia_sdk import Study

# Open a .omnia file
study = Study("ct_scan.omnia")

# Random access to any slice — O(1)
slice_47 = study[47]       # returns numpy array, shape (512, 512)

# Bulk read
volume = study[:]           # returns numpy array, shape (277, 512, 512)

# Metadata
print(study.num_slices)     # 277
print(study.shape)          # (512, 512)
print(study.dtype)          # int16
```

```bash
# CLI
omnia compress ./ct_scans/ ./compressed/
omnia info ./study.omnia
omnia verify ./study.omnia
```

---

## Benchmark methodology

All measurements were collected on a dedicated instance with no other GPU processes running.

| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA RTX A4000 (16 GB VRAM) |
| **CPU** | 16 cores @ 2.5 GHz |
| **RAM** | 440 GB |
| **Storage** | NVMe SSD (network filesystem, 207 TB free) |
| **OS** | Ubuntu 22.04, kernel 5.15 |
| **PyTorch** | 2.8.0+cu128 |
| **CUDA** | 12.8 |
| **cuDNN** | Benchmark mode enabled (`torch.backends.cudnn.benchmark = True`) |
| **Model** | ResNet-18 (modified: conv1 accepts 1 channel, 2-class output) |
| **Batch size** | 64 |
| **Epochs** | 100 (reported: last 50 steady-state) |
| **Data loaders** | 4 workers, prefetch_factor=2, pin_memory=True, persistent_workers=True |
| **Dataset** | LIDC-IDRI, 15 patients, 3,387 CT slices, 512×512 int16, uncompressed (TransferSyntax 1.2.840.10008.1.2.1) |
| **Data source** | The Cancer Imaging Archive (TCIA) — real patient CT scans |

**Results (mean of 50 steady-state epochs):**

| Metric | Raw DICOM | .omnia |
|--------|-----------|--------|
| Mean epoch time | 40.9 s | 21.9 s |
| GPU utilization | 48% | 93% |
| Storage | 1,819 MB | 837 MB |
| Dataset load time | 127.6 s | 0.7 s |
| Cold start (epoch 1) | 215.8 s | 69.3 s |
| System calls per epoch | ~17,000 | ~30 |
| Lossless verification | — | 0 errors / 3,387 slices |

> **Note on the 182× dataset loading improvement:** The 127.6 s for DICOM reflects the cost of recursively walking the filesystem, opening each of 3,387 files, and parsing its DICOM header to confirm it is a CT slice. .omnia loads in 0.7 s because it reads 15 pre-indexed container files. This is a one-time initialization cost, not a per-epoch cost — but it must be paid before training starts. For large datasets (50,000+ studies) the cost scales linearly with file count for DICOM and is constant for .omnia.

> **Note on cold vs. steady state:** The benchmarks were collected on a system with 440 GB RAM. The 3,387-slice DICOM dataset (1.8 GB) fits entirely in OS page cache after one epoch. This is the best possible case for DICOM. In production with datasets that do not fit in RAM, DICOM would remain in cold-start territory indefinitely. We report both cold and steady-state numbers so readers can estimate behavior for their dataset size.

---

## How the speedup arises

Training pipelines use multi-worker data loaders that prefetch batches in parallel. When DICOM files are involved, worker threads spend most of their time contending for filesystem access. Under high load (4 workers, pinned memory, prefetch queue of 2), the filesystem becomes the bottleneck — not the GPU.

.omnia eliminates filesystem contention entirely. Worker threads issue seeks into already-open file handles rather than competing for new file descriptors. The prefetch queue stays full. The GPU never stalls.

The effect is most pronounced on cold start (3.1×) where the filesystem cache is empty. At steady state with a fully warmed cache on a small dataset (< 2 GB), the advantage narrows to 1.87×. For real-world datasets (10 TB+) that never fully warm the cache, cold-start behavior is the dominant regime.

---

## Comparison with alternatives

A number of existing systems address aspects of the I/O problem in ML pipelines. The table below explains why they do not solve the medical imaging case specifically.

| System | Strengths | Why not a substitute for .omnia |
|--------|-----------|----------------------------------|
| **Zarr** | Chunked array storage, S3-native, O(1) access | No DICOM ingestion. No medical metadata preservation. Requires pipeline rewrite to array format. |
| **HDF5** | Hierarchical, widely used, parallel I/O | Single-file but no DICOM support. Painful concurrency model. Bloated metadata for large studies. |
| **WebDataset** | Sharded tar archives, fast I/O | Lossy concatenation of existing files — still requires DICOM parsing per sample. No random access. |
| **LMDB** | Memory-mapped key-value, fast lookups | No DICOM ingestion. No compression. Memory-maps entire dataset — impractical at 10 TB+. |
| **MONAI CacheDataset** | In-memory caching for small datasets | Caches entire dataset in RAM. Does not reduce storage. Does not fix cold-start I/O. |
| **FFCV** | Fast data loading with compilation | Tight coupling to model architecture. No DICOM support. Limited community. |
| **DALI** | GPU-accelerated data pipeline | No DICOM decoder. Requires custom operator. No storage savings. |
| **tar.gz / ZIP** | Universal, simple | No random access — must decompress entire archive to reach one slice. O(n) seeks. |
| **Multi-page TIFF** | Single file, widely supported | O(n) IFD chain traversal. No DICOM metadata. No per-slice compression control. |
| **NIfTI (.nii.gz)** | Standard in neuroimaging | No DICOM preservation. Gzip requires full decompression. Single volume — no study-level organization. |
| **JPEG 2000 (DICOM)** | High compression, lossless | Per-instance only — does not aggregate studies. 16 ms/slice decode is slow for training. |
| **AV1 / H.265** | Very high compression | Lossy — not acceptable for diagnostic use. No random access. Patent encumbered. |

None of these systems combine: (1) DICOM-native ingestion, (2) study-level aggregation, (3) O(1) random access, (4) per-slice integrity verification, and (5) lossless compression in a single format designed for AI training pipelines. Each solves one part of the problem but leaves the others unaddressed.

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

## Limitations

- Single-series CT studies only. Multi-series and multi-modality studies are not yet supported.
- Not a DICOM transfer syntax — does not replace DICOM communication protocols (DICOMweb, C-STORE).
- Requires a custom reader — no native PACS or viewer support (SDK available).
- Research prototype. Not clinically validated. Not for diagnostic use.
- Not a primary archive format — designed as a cache / training storage layer alongside existing PACS infrastructure.

---

## Related work

The problem of DICOM I/O overhead in ML pipelines is well known but has not been addressed at the container level. Existing approaches fall into several categories:

**Format-level** approaches (Zarr, HDF5, NIfTI) require converting DICOM to a new representation, discarding medical metadata and study structure. They solve storage but break clinical traceability.

**Caching** approaches (MONAI CacheDataset, DALI) keep memory-mapped or pre-decoded copies but do not reduce storage or fix cold-start I/O. They are effective for small datasets but do not scale to multi-TB archives.

**Archive** approaches (WebDataset, tar.gz) reduce file count but sacrifice random access — reading a single slice requires scanning through preceding slices.

**Compression** approaches (JPEG 2000, JPEG-LS, AV1) operate at the slice level and do not address study-level aggregation or filesystem overhead.

.omnia differs from these by operating at the study level — preserving DICOM's logical grouping while eliminating the filesystem tax that comes from 1:1 slice-to-file mapping.

---

## License

Proprietary — All rights reserved.

<br/>

<p align="center">
  <span style="font-size: 11px; color: #445566;">© 2026</span>
</p>
