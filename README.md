<p align="center">
  <br/><br/>
  <code style="background: #0a1628; padding: 20px 40px; border-radius: 12px; border: 1px solid #1a2a3e; font-size: 48px; color: #0066CC; letter-spacing: 4px; font-weight: 300;">.omnia</code>
  <br/><br/>
  <span style="font-size: 15px; color: #667788; letter-spacing: 6px; font-weight: 300;">A CONTAINER FORMAT FOR MEDICAL IMAGING</span>
  <br/><br/><br/>
</p>

<p align="center" style="font-size: 14px; color: #778899; max-width: 520px; line-height: 1.7;">
  277 DICOM files per study. 13.8 million files per dataset. 69 million syscalls per epoch.
  <strong style="color: #99aabb;">This work proposes a single-file container that eliminates the O(n) I/O tax.</strong>
</p>

<br/>

<p align="center">
  <a href="https://img.shields.io/badge/status-research--prototype-0066CC?style=flat-square&labelColor=0a1628"><img src="https://img.shields.io/badge/status-research--prototype-0066CC?style=flat-square&labelColor=0a1628" alt="Status"></a>
  <a href="https://img.shields.io/badge/license-proprietary-445566?style=flat-square&labelColor=0a1628"><img src="https://img.shields.io/badge/license-proprietary-445566?style=flat-square&labelColor=0a1628" alt="License"></a>
</p>

<br/><br/>

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

## The container

The container is a single file with three regions:

```
┌─────────────────────────────────────────────┐
│ Header                                       │
│ • Magic bytes                                │
│ • Offset table: slice 0..N → byte position   │
│ • Slice metadata: shape, dtype, original size │
├─────────────────────────────────────────────┤
│ Slice data                                    │
│ • Chunk 0: [CRC32] [compressed bytes]        │
│ • Chunk 1: [CRC32] [compressed bytes]        │
│ • ...                                        │
│ • Chunk N: [CRC32] [compressed bytes]        │
└─────────────────────────────────────────────┘
```

Each slice is independently addressable via the offset table. Accessing slice 147 does not require decompressing slices 0–146. There is no full-file extraction step. The offset table is loaded once at open time and kept in memory.

---

## What it eliminates

| Layer | Conventional DICOM | .omnia |
|-------|-------------------|--------|
| **Filesystem** | 13.8M inodes consuming kernel memory | 50K inodes |
| **System calls** | 69M per epoch (open/stat/read/close) | ~50K per epoch |
| **DICOM parsing** | Per-slice header traversal (277× per study) | Once per container |
| **File descriptors** | 3,387 opened and closed per epoch | 15 opened once |
| **Dataset loading** | 127 seconds to enumerate and validate | 0.7 seconds |
| **Storage overhead** | Repeated metadata headers (277×) | Single header |
| **Ingestion atomicity** | 277 writes — crash at 147 leaves corruption | 1 write — atomic |
| **Backup** | 50M files → 3 days | 50K files → 1 hour |

---

## Empirical results

| Metric | Raw DICOM | .omnia |
|--------|-----------|--------|
| Mean epoch time | 40.9 s | 21.9 s |
| GPU utilization | 48% | 93% |
| Storage volume | 1,819 MB | 837 MB |
| Dataset load time | 127.6 s | 0.7 s |
| Cold start (epoch 1) | 215.8 s | 69.3 s |
| System calls per epoch | ~17,000 | ~30 |
| File operations per epoch | 3,387 open+close | 0 |
| Lossless verification | — | 0 errors / 3,387 slices |

*Hardware: NVIDIA RTX A4000 · Model: ResNet‑18 · Data: 3,387 real CT slices (LIDC‑IDRI) · 100 epochs · Batch size: 64*

---

## How the speedup arises

Training pipelines use multi-worker data loaders that prefetch batches in parallel. When DICOM files are involved, worker threads spend most of their time contending for filesystem access. Under high load (multiple workers, pinned memory, prefetch queues), the filesystem becomes the bottleneck — not the GPU, not the model.

The container eliminates filesystem contention entirely. Worker threads issue seeks into already-open file handles rather than competing for new file descriptors. The prefetch queue stays full. The GPU never stalls.

The effect is most pronounced on cold start (3.1× faster) where the filesystem cache is empty and every DICOM access requires physical disk I/O. At steady state with a fully warmed cache on a small dataset (under 2 GB), the advantage narrows to 1.87× — but real-world datasets (10 TB+) never fully warm the cache, so cold-start behavior is the dominant regime.

---

## Comparison with existing formats

| Format | Random access | Lossless | Single file | DICOM-aware | Per-slice CRC | O(1) seek |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Raw DICOM | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| ZIP | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| tar.gz | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Multi-page TIFF | O(n) | ✅ | ✅ | ❌ | ❌ | ❌ |
| NIfTI (.nii) | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| H.265 / AV1 | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **.omnia** | **O(1)** | **✅** | **✅** | **✅** | **✅** | **✅** |

---

## Limitations

- Designed for single-series CT studies. Multi-series, multi-modality studies are not yet supported.
- Not a DICOM transfer syntax — does not replace DICOM communication protocols.
- Requires custom reader — no native PACS or viewer support (SDK available).
- Research prototype. Not clinically validated. Not for diagnostic use.

---

## Related work

Multi-page TIFF stores slices as IFD entries but requires O(n) directory traversal to locate a specific page. ZIP and tar archives require full or sequential decompression. NIfTI was designed for neuroimaging and lacks DICOM metadata preservation. Video codecs (H.265, AV1) achieve high ratios through lossy quantization unsuitable for medical diagnostics. DICOM's encapsulated transfer syntaxes (JPEG 2000, JPEG-LS) operate per-instance and do not address study-level aggregation or filesystem overhead.

To the best of our knowledge, no existing format combines: (1) study-level aggregation, (2) O(1) random access, (3) per-slice integrity verification, and (4) lossless compression in a single container designed for AI training pipelines.

---

## License

Proprietary — All rights reserved.

<br/>

<p align="center">
  <span style="font-size: 11px; color: #445566;">© 2026</span>
</p>
