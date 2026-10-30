# Benchmarks per medical image format

One RunPod **A40** (secure cloud, 9 vCPU), ResNet-18 + AMP, shuffled `DataLoader`, 2026-10-30, one run per
number (no repeats). Rate = items per second through a real training loop; "GPU%" = share of the
no-data-loading ceiling. Reproduce: `omnia benchmark formats <files>`.
`.omnia` zstd = level 5, one tile per block (the default). JPEG = quality 90.
Slides are tiles from the second pyramid level (`min_level=1`); radiology is one 2D slice per item.
Samples are public: OpenSlide test data (https://openslide.org/demo/), OpenNeuro ds000114 (CC0),
and the NCI Imaging Data Commons (CC BY 4.0).

## Slides: native reader (openslide) vs `.omnia` — 4 data workers

| Format | Sample | openslide | `.omnia` zstd (lossless) | `.omnia` JPEG q90 | Lossless speed-up |
|---|---|---|---|---|---|
| Aperio SVS (JPEG) | CMU-1 | 436 (18%) | 1,616 (68%) | 2,059 (86%) | **3.7x** |
| Aperio SVS (JPEG 2000) | JP2K-33003-1 | 2,020 (86%) | 2,106 (90%) | 2,275 (97%) | 1.04x |
| Hamamatsu NDPI | CMU-1 | 2,485 (96%) | 2,483 (96%) | 2,264 (88%) | **1.0x (no gain)** |
| Leica SCN | Leica-1 | 822 (35%) | 2,166 (92%) | 2,061 (88%) | **2.6x** |
| Generic tiled TIFF | CMU-1 | 1,535 (65%) | 2,041 (86%) | 2,096 (88%) | 1.3x |
| Philips TIFF | Philips-1 | 481 (75%) | 613 (96%) | see note | 1.3x |
| 3DHISTECH MIRAX | CMU-1 | 545 (24%) | 1,941 (86%) | 1,926 (85%) | **3.6x** |

With **8 workers** most gaps close (openslide reaches 69-98% of the GPU on every slide except MIRAX at 45%), so the
gain is about needing fewer CPU cores, not a guaranteed faster epoch. Tiles read from `.omnia` were byte-identical to
`read_region` output for every slide (`tiles identical to native: True`).

Single-thread random read, items/s: openslide 92-6,553 vs `.omnia` zstd (lossless) 912-29,524 (NDPI is the exception where
openslide is already fast at 6,553).

## Size (second pyramid level and below only, so not comparable to the source file)

| Format | Source | `.omnia` zstd (lossless) | `.omnia` JPEG q90 |
|---|---|---|---|
| SVS JPEG | 178 MB | 60 MB | 12 MB |
| SVS JPEG 2000 | 64 MB | 27 MB | 5 MB |
| NDPI | 198 MB | 519 MB | 433 MB |
| Leica SCN | 292 MB | 143 MB | 45 MB |
| Generic TIFF | 204 MB | 353 MB | 73 MB |
| Philips TIFF | 327 MB | 624 MB | see note |
| MIRAX | 565 MB (zip) | 1,214 MB | 340 MB |

zstd (lossless) `.omnia` is often **larger** than the source for already-compressed slides.

## Radiology

| Format | Sample | Native reader | `.omnia` | Size | Bit-exact |
|---|---|---|---|---|---|
| NIfTI (brain MRI T1w, 256 slices) | OpenNeuro | nibabel slice from `.nii.gz`: 369/s (19%); nibabel with whole volume in RAM: 1,899/s (96%) | **1,951/s (99%)** | 8.7 MB -> 7.7 MB | yes |
| DICOM MR series (192 slices) | IDC | pydicom, one file per slice: 1,997/s (90%) | 1,969/s (89%) | 40 MB -> 11 MB | yes |
| DICOM CT series (114 slices) | IDC | single-thread read 723/s | single-thread read 12,642/s | 60 MB -> 22.5 MB | yes |

- NIfTI: `.omnia` is ~5x faster than reading slices out of the gzip file and equal to decompressing the whole volume
  into RAM first (which costs memory per worker).
- DICOM MR: no training-speed difference at 4 workers (the GPU is the limit); the single-thread read is 32x faster.

## What did not work or is not measured
- **DICOM CT training rate, and DICOM whole-slide:** the first run had too few items per epoch to time a training loop
  (CT) or failed outright (DICOM-WSI, a 5-instance GTEx slide, 8.4 MB output). The benchmark now sizes its batch to the
  data; **re-run needed**. The DICOM-WSI converter completed (0.9 s) but has not been checked for correctness.
- **Philips JPEG row is invalid:** the JPEG converter cropped tiles larger than 256 px to 256x256. Philips' larger tiles
  were cut to a quarter of their pixels, which is why that row looked 3x faster and 20x smaller. Fixed in this
  commit (regression test added); the row must be re-measured.
- **MIRAX "source size"** printed 0.07 MB because only the index file was counted; the real source is the 565 MB zip.
- One GPU, one model, one run per number. The 4-worker vs 8-worker difference between slides shows run-to-run
  noise of about +/-10%; treat gaps smaller than that as ties.
