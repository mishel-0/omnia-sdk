"""Convert medical images to .omnia containers.

    omnia_sdk.convert("slide.svs")               # whole-slide image
    omnia_sdk.convert("scan.nii.gz")             # NIfTI volume
    omnia_sdk.convert("study_dir/")              # DICOM series or DICOM whole-slide

Formats
-------
whole-slide   .svs .ndpi .mrxs .scn .vms .vmu .bif .tif .tiff   (via OpenSlide)
dicom-wsi     a folder of VL Whole Slide Microscopy .dcm files   (needs ``omnia-sdk[dicom-wsi]``)
dicom-series  a folder (or single file) of CT/MR/other .dcm      (needs ``omnia-sdk[dicom]``)
nifti         .nii .nii.gz                                       (needs ``omnia-sdk[nifti]``)

A radiology volume is stored as one tile per 2D slice, in the scan's own
dtype (no rescaling, no windowing), so the round trip is bit-exact. The
rescale slope/intercept needed to get Hounsfield units are kept in the
container's ``volume`` metadata.

Privacy: radiology headers are NOT copied. Only an allow-list of acquisition
fields (modality, spacing, slice thickness, rescale, window) is stored; patient
name, ID, birth date, UIDs and dates never enter the container. Pixel data can
still contain burned-in text, which no converter can detect.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .container import OmniaContainer, DEFAULT_BATCH_SIZE, DEFAULT_ZSTD_LEVEL

WSI_SUFFIXES = {".svs", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu", ".bif", ".tif", ".tiff"}
DICOM_WSI_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.77.1.6"

# The only DICOM attributes copied into a container.
DICOM_ALLOW_LIST = (
    "Modality", "PixelSpacing", "SliceThickness", "SpacingBetweenSlices",
    "RescaleSlope", "RescaleIntercept", "WindowCenter", "WindowWidth",
    "BitsStored", "PhotometricInterpretation",
)


class UnsupportedFormat(ValueError):
    """The input is not a medical image format this SDK can convert."""


def _need(module: str, extra: str):
    try:
        return __import__(module)
    except ImportError as e:
        raise ImportError(
            f"Converting this format needs the optional '{module}' package: "
            f"pip install 'omnia-sdk[{extra}]'"
        ) from e


def detect_format(path) -> str:
    """Return 'whole-slide', 'dicom-wsi', 'dicom-series' or 'nifti'."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    name = p.name.lower()
    if name.endswith((".nii", ".nii.gz")):
        return "nifti"
    if p.is_file() and p.suffix.lower() in WSI_SUFFIXES:
        return "whole-slide"
    dicoms = _dicom_files(p)
    if dicoms:
        pydicom = _need("pydicom", "dicom")
        first = pydicom.dcmread(str(dicoms[0]), stop_before_pixels=True)
        if str(getattr(first, "SOPClassUID", "")) == DICOM_WSI_SOP_CLASS:
            return "dicom-wsi"
        return "dicom-series"
    raise UnsupportedFormat(
        f"{p.name}: not a recognised medical image. Supported: whole-slide "
        f"({', '.join(sorted(WSI_SUFFIXES))}), DICOM (folder or .dcm), NIfTI (.nii/.nii.gz)."
    )


def _dicom_files(p: Path) -> list[Path]:
    if p.is_file():
        return [p] if _is_dicom(p) else []
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.is_file() and _is_dicom(f))
    return []


def _is_dicom(f: Path) -> bool:
    if f.suffix.lower() in (".dcm", ".dicom"):
        return True
    try:
        with open(f, "rb") as fh:
            fh.seek(128)
            return fh.read(4) == b"DICM"
    except OSError:
        return False


def _finish(stats: dict, source: Path, fmt: str) -> dict:
    src_bytes = sum(f.stat().st_size for f in source.rglob("*") if f.is_file()) if source.is_dir() else source.stat().st_size
    stats["format"] = fmt
    stats["source"] = source.name
    stats["source_bytes"] = src_bytes
    stats["ratio_vs_source"] = round(src_bytes / stats["compressed_bytes"], 3)
    return stats


# ── NIfTI ────────────────────────────────────────────────────────────────

def nifti_to_omnia(path: Path, out: Path, *, axis: int = 2,
                   batch_size: int = DEFAULT_BATCH_SIZE,
                   zstd_level: int = DEFAULT_ZSTD_LEVEL) -> dict:
    """One tile per 2D slice along ``axis``, in the volume's stored dtype."""
    nib = _need("nibabel", "nifti")
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise UnsupportedFormat(f"{path.name}: expected a 3D volume, got shape {data.shape}")
    if not 0 <= axis < 3:
        raise ValueError("axis must be 0, 1 or 2")
    data = np.ascontiguousarray(np.moveaxis(data, axis, 0))
    zooms = [float(z) for z in img.header.get_zooms()[:3]]
    meta = {
        "source": path.name,
        "volume": {
            "modality": "NIfTI",
            "shape": list(data.shape),
            "axis": axis,
            "spacing": zooms,
            "axcodes": list(nib.aff2axcodes(img.affine)),
            "affine": [[float(v) for v in row] for row in img.affine],
        },
    }
    stats = OmniaContainer.write(out, (data[i] for i in range(data.shape[0])),
                                 metadata=meta, batch_size=batch_size, zstd_level=zstd_level)
    return _finish(stats, path, "nifti")


# ── DICOM series ─────────────────────────────────────────────────────────

def _scalar(v):
    try:
        if hasattr(v, "__len__") and not isinstance(v, str):
            return [float(x) for x in v]
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def dicom_series_to_omnia(path: Path, out: Path, *,
                          batch_size: int = DEFAULT_BATCH_SIZE,
                          zstd_level: int = DEFAULT_ZSTD_LEVEL) -> dict:
    """A folder (or one multi-frame file) of CT/MR/other DICOM as one tile per slice."""
    pydicom = _need("pydicom", "dicom")
    files = _dicom_files(path)
    if not files:
        raise UnsupportedFormat(f"{path}: no DICOM files found")

    slices = []  # (sort_key, array, dataset-without-pixels)
    for f in files:
        ds = pydicom.dcmread(str(f))
        if "PixelData" not in ds:
            continue  # reports, structured data, etc.
        try:
            arr = ds.pixel_array
        except Exception as e:  # compressed transfer syntax without a decoder
            raise UnsupportedFormat(
                f"{f.name}: cannot decode pixel data ({e}). For compressed DICOM "
                f"install 'pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg'."
            ) from e
        pos = getattr(ds, "ImagePositionPatient", None)
        key = (float(pos[2]) if pos is not None and len(pos) == 3 else None,
               int(getattr(ds, "InstanceNumber", 0) or 0))
        frames = arr[None] if arr.ndim == 2 or (arr.ndim == 3 and getattr(ds, "SamplesPerPixel", 1) == 3) else arr
        for j, frame in enumerate(frames):
            slices.append(((key[0] if key[0] is not None else float("inf"), key[1], j), frame, ds))
    if not slices:
        raise UnsupportedFormat(f"{path}: DICOM files contain no pixel data")

    slices.sort(key=lambda s: s[0])
    first_ds = slices[0][2]
    dtypes = {s[1].dtype for s in slices}
    shapes = {s[1].shape for s in slices}
    if len(dtypes) != 1 or len(shapes) != 1:
        raise UnsupportedFormat(
            f"{path}: slices differ in size or type ({sorted(map(str, shapes))}); "
            f"convert each series separately."
        )
    vol = {"Modality": str(getattr(first_ds, "Modality", "")), "n_slices": len(slices)}
    for name in DICOM_ALLOW_LIST:
        if name in first_ds and name not in vol:
            vol[name] = _scalar(first_ds[name].value)
    meta = {"source": path.name, "volume": vol}
    stats = OmniaContainer.write(out, (s[1] for s in slices), metadata=meta,
                                 batch_size=batch_size, zstd_level=zstd_level)
    return _finish(stats, path, "dicom-series")


# ── DICOM whole-slide ────────────────────────────────────────────────────

def dicom_wsi_to_omnia(path: Path, out: Path, *, min_level: int = 0,
                       batch_size: int = DEFAULT_BATCH_SIZE,
                       zstd_level: int = DEFAULT_ZSTD_LEVEL) -> dict:
    """DICOM whole-slide microscopy, tile by tile through wsidicom."""
    wsidicom = _need("wsidicom", "dicom-wsi")
    slide = wsidicom.WsiDicom.open(str(path))
    try:
        levels = list(slide.pyramid)
        if not 0 <= min_level < len(levels):
            raise ValueError(f"min_level must be 0..{len(levels) - 1}")
        base = levels[0].size.width
        levels_meta, plan, idx = [], [], 0
        for lv_no in range(min_level, len(levels)):
            lv = levels[lv_no]
            w, h = lv.size.width, lv.size.height
            tw, th = lv.tile_size.width, lv.tile_size.height
            tx, ty = (w + tw - 1) // tw, (h + th - 1) // th
            levels_meta.append({"w": w, "h": h, "tw": tw, "th": th, "tx": tx, "ty": ty,
                                "tc": tx * ty, "start": idx})
            plan.append((lv_no, lv, tx, ty))
            idx += tx * ty

        def tiles() -> Iterator[np.ndarray]:
            for lv_no, _lv, tx, ty in plan:
                level_id = slide.pyramid[lv_no].level
                for y in range(ty):
                    for x in range(tx):
                        yield np.asarray(slide.read_tile(level_id, (x, y)).convert("RGB"), dtype=np.uint8)

        meta = {"source": path.name, "slide_dims": [base, levels[0].size.height], "levels": levels_meta}
        stats = OmniaContainer.write(out, tiles(), metadata=meta,
                                     batch_size=batch_size, zstd_level=zstd_level)
    finally:
        slide.close()
    return _finish(stats, path, "dicom-wsi")


# ── Dispatcher ───────────────────────────────────────────────────────────

def convert_file(src, out=None, *, fmt: Optional[str] = None, **kwargs) -> dict:
    """Convert one file or study to .omnia and return its stats."""
    src = Path(src)
    fmt = fmt or detect_format(src)
    if out is None:
        out = (src.parent / (src.name.split(".")[0] + ".omnia")) if src.is_file() else src.with_suffix(".omnia")
    out = Path(out)
    if out.is_dir():
        out = out / (src.name.split(".")[0] + ".omnia")
    out.parent.mkdir(parents=True, exist_ok=True)

    if fmt in ("whole-slide", "dicom-wsi"):
        # Measured on an A40 (benchmarks/codec_results_a40.json): 16 tiles per
        # compressed block saved no space on H&E tiles and slowed shuffled reads
        # 6-14x, so slides default to one tile per block.
        kwargs.setdefault("batch_size", 1)
    if fmt == "whole-slide":
        from .svs_to_omnia import svs_to_omnia
        kwargs.setdefault("verbose", False)
        stats = svs_to_omnia(src, out, **kwargs)
        return _finish(stats, src, "whole-slide")
    kwargs.pop("verbose", None)
    if fmt == "nifti":
        return nifti_to_omnia(src, out, **{k: v for k, v in kwargs.items() if k in ("axis", "batch_size", "zstd_level")})
    if fmt == "dicom-series":
        return dicom_series_to_omnia(src, out, **{k: v for k, v in kwargs.items() if k in ("batch_size", "zstd_level")})
    if fmt == "dicom-wsi":
        return dicom_wsi_to_omnia(src, out, **{k: v for k, v in kwargs.items() if k in ("min_level", "batch_size", "zstd_level")})
    raise UnsupportedFormat(f"Unknown format {fmt!r}")
