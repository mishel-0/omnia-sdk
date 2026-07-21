<p align="center">
  <br/><br/><br/>
  <span style="font-size: 56px; font-weight: 300; color: #0066CC; letter-spacing: -1px;">.omnia</span>
  <br/><br/>
  <span style="font-size: 16px; color: #778899; letter-spacing: 2px; font-weight: 400;">A Container Format for Medical Imaging</span>
  <br/><br/><br/>
</p>

<p align="center" style="font-size: 14px; color: #8899aa; max-width: 600px;">
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

DICOM stores each CT slice as a separate file. A single study routinely produces 277 files, and a training dataset of 50,000 studies represents roughly 13.8 million files. Each training epoch opens, reads, and closes every file — incurring millions of system calls for metadata alone. This work proposes a container format that collates all slices into a single file with a precomputed offset table, enabling O(1) random access to any slice without per-file overhead.

<br/>

## Method

The container stores compressed slice data sequentially with an index of byte offsets. At training time, files are opened once and held; individual slices are retrieved by seeking to their offset and decompressing. No per-slice file operations or DICOM header parsing occurs after initial loading.

<br/>

## Benchmark

*ResNet‑18 · 3,387 real CT slices · NVIDIA RTX A4000 · 100 epochs*

| Metric | Raw DICOM | .omnia |
|--------|-----------|--------|
| Mean epoch time | 40.9 s | 21.9 s |
| GPU utilization | 48% | 93% |
| Storage volume | 1,819 MB | 837 MB |
| Dataset load time | 127.6 s | 0.7 s |
| Cold start (epoch 1) | 215.8 s | 69.3 s |
| Lossless verification | — | 0 errors / 3,387 slices |

<br/>

## Related work

Conventional approaches (multi-page TIFF, NIfTI, ZIP archives) either lack random access or require full decompression. Video codecs offer high ratios but introduce loss. DICOM's own transfer syntaxes are limited to single-file contexts and do not address study-level aggregation.

<br/>

## Status

Research prototype. Licensed under proprietary terms.

<br/>

---

<br/>

<p align="center">
  <span style="font-size: 12px; color: #556677;">© 2026 — All rights reserved.</span>
</p>

<br/><br/><br/>
