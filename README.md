<h1 align="center">omnia-sdk</h1>

<p align="center"><b>Stop letting slide decoding starve your GPU.</b><br/>
Convert whole-slide images, DICOM and NIfTI into one fast, verifiable <code>.omnia</code> file and train straight from it.</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT"/>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/PyTorch-ready-ee4c2c" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/formats-SVS%20%C2%B7%20MRXS%20%C2%B7%20SCN%20%C2%B7%20NDPI%20%C2%B7%20TIFF%20%C2%B7%20DICOM%20%C2%B7%20NIfTI-6366f1" alt="formats"/>
</p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="benchmarks/charts/hero_speedup_dark.png">
  <img src="benchmarks/charts/hero_speedup_light.png" alt="Training throughput: up to 8.5x faster from .omnia than from the original slide file" width="100%">
</picture>

```bash
pip install "omnia-sdk[all] @ git+https://github.com/mishel-0/omnia-sdk.git"
omnia convert slide.svs        # -> slide.omnia, then train from it
```

## Why

- **Faster training where decoding is the bottleneck.** Measured 1.8x to 8.5x on Aperio SVS, MIRAX and Leica SCN with the same model and GPU ([full results](docs/FORMATS.md)).
- **Exact.** Tiles come back identical to OpenSlide; NIfTI, CT and MR slices bit-for-bit. Every block is CRC-checked.
- **One format for your whole pipeline.** Slides, DICOM series, DICOM whole-slide and NIfTI behind one `omnia convert` and one PyTorch dataset.
- **Private by default.** Converting DICOM keeps only acquisition fields; patient name, ID and dates never enter the file.
- **Measure it on your own data** in a minute: `omnia recommend slide.svs`, or the full run with `omnia benchmark formats`.

## Install

```bash
pip install "omnia-sdk[all] @ git+https://github.com/mishel-0/omnia-sdk.git"   # slides, DICOM, NIfTI
pip install "git+https://github.com/mishel-0/omnia-sdk.git"                    # core only (no format readers)
```

Python 3.11+ (not on PyPI yet). Extras: `[svs]` (OpenSlide slides), `[dicom]`, `[dicom-wsi]`, `[nifti]`.

## Use

```bash
omnia convert slide.svs                 # any supported file or folder -> slide.omnia
omnia convert slide.svs --codec jpeg    # smaller, lossy
omnia verify slide.omnia                # checks every CRC
omnia info slide.omnia
omnia recommend slide.svs               # ~1 minute, no GPU: which codec suits this slide
omnia benchmark formats a.svs scan.nii.gz --zstd-levels 1 9   # compare against the native reader on your machine
```

```python
import omnia_sdk
from torch.utils.data import DataLoader

omnia_sdk.convert("slide.svs")
ds = omnia_sdk.OmniaDataset("slide.omnia")
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)
```

## What it supports

| Input | Reader | Status |
|---|---|---|
| Aperio SVS, Hamamatsu NDPI, 3DHISTECH MIRAX, Leica SCN, generic and Philips TIFF | OpenSlide | tested, tiles identical to `read_region` |
| NIfTI (`.nii`, `.nii.gz`) | nibabel | tested, bit-exact |
| DICOM series (CT, MR, ...) | pydicom | tested, bit-exact; patient identifiers are not copied |
| DICOM whole-slide | wsidicom | works, tiles identical; slower than wsidicom on the one sample tested |

A `.omnia` file is CRC-checked per block, lossless by default (zstd) or lossy (JPEG), with O(1) tile access.
The header allow-list for radiology keeps modality, spacing and rescale values only. Burned-in text in pixels is not detected.

## Where it helps, and where it doesn't

Measured on one NVIDIA A40 (9 vCPU), ResNet-18, 4 data workers, public sample slides, two runs that differed by up to 2x on some rows:

- Faster to train from: Aperio SVS, MIRAX, Leica SCN, NIfTI.
- About the same: Hamamatsu NDPI, DICOM CT and MR.
- Not shown to help: DICOM whole-slide (one 57-tile sample, slower).
- Lossless `.omnia` is often **larger** than an already-compressed slide (NDPI 519 MB vs 198 MB); JPEG mode is 4-12x smaller but lossy,
  and its effect on model accuracy was not measured.
- With 8 workers most slide gaps closed in the earlier run: the win is mostly needing fewer CPU cores.

Reproduce everything with `omnia benchmark formats`. Sample sources and licences are listed in [`benchmarks/results.json`](benchmarks/results.json).

## Development

```bash
pip install -e ".[all,dev]"
pytest
python benchmarks/build_report.py       # regenerate charts and JSON from the recorded runs
```

The earlier README (cuCIM comparison, Colab notebook) is kept in [`docs/LEGACY_README.md`](docs/LEGACY_README.md).

MIT licensed.
