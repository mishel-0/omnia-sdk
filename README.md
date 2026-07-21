<p align="center">
  <br/><br/>
  <img src="https://capsule-render.vercel.app/api?type=rect&color=0:0a1628,100:0a1628&height=4&width=60&section=header" width="60"/>
  <br/><br/>
  <span style="font-size: 60px; font-weight: 600; color: #0066CC; letter-spacing: -2px;">.omnia</span>
  <br/>
  <span style="font-size: 17px; color: #8899aa; letter-spacing: 3px; font-weight: 300;">A Container Format for Medical Imaging</span>
  <br/><br/>
  <img src="https://capsule-render.vercel.app/api?type=rect&color=0:0a1628,100:0a1628&height=4&width=60&section=header" width="60"/>
  <br/><br/>
</p>

<p align="center" style="font-size: 14px; color: #8899aa; max-width: 560px; line-height: 1.6;">
  .omnia bundles CT studies into single files while preserving lossless quality.
  The project investigates whether consolidating 277 DICOM files per study
  into one container reduces I/O overhead in AI training pipelines.
</p>

<br/>

<p align="center">
  <a href="https://img.shields.io/badge/license-proprietary-445566?style=flat-square&labelColor=0a1628"><img src="https://img.shields.io/badge/license-proprietary-445566?style=flat-square&labelColor=0a1628" alt="License"></a>
</p>

<br/><br/>

---

<br/>

## Abstract

DICOM stores each CT slice as a separate file. A single study routinely produces 277 files, and a training dataset of 50,000 studies represents roughly 13.8 million files. Each training epoch opens, reads, and closes every file — incurring millions of system calls for metadata alone.

This work proposes a container format that collates all slices into a single file with a precomputed offset table, enabling O(1) random access to any slice without per-file overhead.

<br/>

---

<br/>

## Method

| Component | Description |
|-----------|-------------|
| **Container** | Single file per study. Header contains offset table, slice count, and metadata. |
| **Access pattern** | Files opened once, kept open. Slices retrieved by seek + decompress. |
| **Random access** | O(1) — any slice independently addressable via offset table. |
| **Memory footprint** | One file handle per study instead of 277. |
| **Decompression** | Per-slice, on-demand. No full-study extraction required. |

<br/>

---

<br/>

## Benchmark

| Metric | Raw DICOM | .omnia |
|--------|-----------|--------|
| **Mean epoch time** | 40.9 s | 21.9 s |
| **GPU utilization** | 48% | 93% |
| **Storage volume** | 1,819 MB | 837 MB |
| **Dataset load time** | 127.6 s | 0.7 s |
| **Cold start (epoch 1)** | 215.8 s | 69.3 s |
| **Lossless verification** | — | 0 errors / 3,387 slices |
| **System calls per epoch** | ~17,000 | ~30 |
| **File operations** | 3,387 per epoch | 15 per epoch |

<sub>ResNet‑18 · 3,387 real CT slices · NVIDIA RTX A4000 · 100 epochs</sub>

<br/>

---

<br/>

## Motivation

| Problem | Impact |
|---------|--------|
| **File proliferation** | 277 files per study → 13.8M files for 50K studies |
| **I/O overhead** | 69M syscalls per epoch from open/stat/read/close |
| **GPU starvation** | GPU sits idle at 48% utilization waiting on data |
| **Dataset loading** | 127 seconds to enumerate and parse dataset |
| **Storage efficiency** | DICOM headers repeated 277 times per study |
| **Backup complexity** | 50M files take 3+ days to back up |

<br/>

---

<br/>

## Comparison with existing approaches

| Format | Random access | Lossless | Single file | DICOM-aware |
|--------|:---:|:---:|:---:|:---:|
| Raw DICOM | ✅ | ✅ | ❌ | ✅ |
| ZIP / tar.gz | ❌ | ✅ | ✅ | ❌ |
| Multi-page TIFF | O(n) | ✅ | ✅ | ❌ |
| NIfTI (.nii.gz) | ❌ | ✅ | ✅ | ❌ |
| Video (H.265) | ❌ | ❌ | ✅ | ❌ |
| **.omnia** | **O(1)** | **✅** | **✅** | **✅** |

<br/>

---

<br/>

## Limitations

- Currently supports single-series CT studies only
- Container format is not a DICOM transfer syntax — not a drop-in PACS replacement
- Requires dedicated reader — no standard viewer support
- Research prototype — not yet validated in clinical workflows

<br/>

---

<br/>

## Related work

Conventional approaches (multi-page TIFF, NIfTI, ZIP) either lack random access or require full decompression before reading any slice. Video codecs achieve high compression ratios but introduce quantization loss unsuitable for diagnostic imaging. DICOM's own encapsulated transfer syntaxes (JPEG 2000, JPEG-LS) operate at the individual instance level and do not address study-level aggregation.

<br/>

---

<br/>

## License

Proprietary — All rights reserved.

<br/>

---

<br/>

<p align="center">
  <span style="font-size: 12px; color: #445566;">© 2026</span>
</p>

<br/><br/>
