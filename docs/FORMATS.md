# Benchmarks per medical image format

RunPod **A40** (secure cloud, 9 vCPU), ResNet-18 + AMP, shuffled `DataLoader`, 4 data workers, 2026-10-30.
Rate = items/s through a real training loop; GPU% = share of the no-data-loading ceiling.
Reproduce: `omnia benchmark formats <files> --zstd-levels 1 3 9`.
`.omnia` zstd = level 5, one tile per block (the default) unless a level is named. JPEG = quality 90.
Slides are tiles from the second pyramid level (`min_level=1`); radiology is one 2D slice per item.
Samples are public: OpenSlide test data (https://openslide.org/demo/), OpenNeuro ds000114 (CC0), NCI Imaging Data Commons (CC BY 4.0).

**Two full runs were made and they disagree by up to 2x on some rows** (different physical hosts, same GPU model),
so each speed-up below is a range. Raw numbers: [`benchmarks/format_results_a40.json`](../benchmarks/format_results_a40.json)
(run 2) and the table in this file (run 1).

## Slides: openslide vs `.omnia` zstd (items/s, run 1 / run 2)

| Format | Sample | openslide | `.omnia` zstd | Speed-up |
|---|---|---|---|---|
| Aperio SVS (JPEG) | CMU-1 | 436 / 474 | 1,616 / 2,131 | **3.4-4.5x** |
| Aperio SVS (JPEG 2000) | JP2K-33003-1 | 2,020 / 1,026 | 2,106 / 1,508 | 1.0-1.5x |
| Hamamatsu NDPI | CMU-1 | 2,485 / 3,438 | 2,483 / 3,168 | **none** |
| Leica SCN | Leica-1 | 822 / 798 | 2,166 / 2,283 | **2.6-2.9x** |
| Generic tiled TIFF | CMU-1 | 1,535 / 1,716 | 2,041 / 2,344 | 1.3-1.4x |
| Philips TIFF | Philips-1 | 481 / 470 | 613 / 621 | 1.3x (GPU-limited at ~640/s) |
| 3DHISTECH MIRAX | CMU-1 | 545 / 566 | 1,941 / 2,291 | **3.4-4.0x** |

JPEG q90 `.omnia` was within about 10% of zstd on every slide.
NDPI is the exception: openslide already reads it as fast as the GPU can consume.
Tiles from `.omnia` were identical to `read_region` output on all seven slides.

### Size (second pyramid level and below, so not comparable to the source file)

| Format | Source | zstd L5 | JPEG q90 |
|---|---|---|---|
| SVS JPEG | 178 MB | 60 MB | 12 MB |
| SVS JPEG 2000 | 64 MB | 27 MB | 5 MB |
| NDPI | 198 MB | 519 MB | 282 MB |
| Leica SCN | 292 MB | 143 MB | 45 MB |
| Generic TIFF | 204 MB | 353 MB | 73 MB |
| Philips TIFF | 327 MB | 624 MB | 114 MB |
| MIRAX | 565 MB | 1,214 MB | 340 MB |

Lossless is often **larger** than the source for already-compressed slides.

## Radiology and DICOM (run 2; items/s at 4 workers)

| Format | Sample | Native reader | `.omnia` zstd | Size (source -> L5) | Bit-exact |
|---|---|---|---|---|---|
| NIfTI | brain MRI T1w, 256 slices | nibabel from `.nii.gz`: 1,695 (run 1: 369); whole volume in RAM: 2,158 | 2,096 | 8.7 -> 7.7 MB | yes |
| DICOM CT | NLST, 114 slices | pydicom: 1,049 | 921 | 60 -> 22.5 MB | yes |
| DICOM MR | 192 slices | pydicom: 1,697 | 1,618 | 40 -> 10.9 MB | yes |
| DICOM whole-slide | GTEx, 5 instances, 57 tiles | wsidicom: 1,191 | **564** | 40 -> 8.4 MB | tiles identical |

- Single-thread random read (run 1 and 2 agree): pydicom ~700-970 slices/s vs `.omnia` 12,600-32,500/s (**13-35x**), nibabel gzip slice 15/s vs `.omnia` 19,000-21,700/s.
- **In training, `.omnia` did not beat pydicom** for CT or MR (within noise), and **was about half the speed of wsidicom**
  on the one DICOM whole-slide sample. That sample is tiny (57 tiles, 3 batches), so treat it as "not shown to help".
- The NIfTI gap depends on run: 369/s vs 1,695/s for the gzip slice reader. Against reading the whole volume into RAM it is a tie.

## zstd level (all formats where run, items 4 workers; run 2)

| Sample | L1 size | L3 | L5 | L9 | L1 vs L9 size |
|---|---|---|---|---|---|
| SVS JPEG | 61.6 MB | 61.5 | 60.3 | 59.9 | +2.7% |
| NDPI | 580 | 546 | 519 | 491 | +18% |
| Leica SCN | 152 | 149 | 143 | 143 | +6% |
| Generic TIFF | 379 | 372 | 353 | 349 | +8% |
| NIfTI | 8.61 | 8.24 | 7.70 | 7.64 | +13% |
| DICOM CT | 26.7 | 23.5 | 22.5 | 22.0 | +21% |
| DICOM MR | 11.9 | 11.6 | 10.9 | 10.7 | +11% |
| DICOM whole-slide | 9.2 | 9.2 | 8.4 | 8.2 | +12% |

- **Level 5 is within 1-3% of level 9** on size and costs far less to write (NDPI: 44 s at L5 vs 131 s at L9).
- Level 1 writes fastest but is 3-21% larger. Read speed does not depend on the level: single-thread decode was
  the same within noise (e.g. NDPI 38,800-41,600/s at L1/L3/L9), and the training-rate differences between levels
  are run-to-run noise, not a trend.
- Earlier, on two Aperio slides: 16 tiles per block saved no space and made shuffled reads 6-14x slower
  ([`CODECS.md`](CODECS.md)). Not re-tested per format.

## Fixed along the way
JPEG mode cropped every tile larger than 256 px to 256x256 (found because the first Philips JPEG row looked 20x smaller).
Fixed with a regression test; the Philips JPEG row above (114 MB, full size) is from the corrected build.

## Limits
One GPU model, one model architecture, one run per number, two runs for the rest. Run-to-run differences of up to 2x
on single rows; gaps under about 30% are ties. The radiology and DICOM-WSI samples are small. Eight-worker rates were
measured only in run 1, where most slide gaps closed (openslide reached 69-98% of the GPU except MIRAX at 45%).
