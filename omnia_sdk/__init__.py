"""
omnia-sdk — PyTorch-native container for whole-slide images.

Medical images (whole-slide, DICOM, NIfTI) in one fast, verifiable container for ML training.
Lossless Zstd container, CRC-verified, zero-copy random access, no openslide
dependency at training time. Ships an auditable benchmark
(`python -m omnia_sdk.benchmark`) that measures both data-only and full-train
speedups live and classifies the machine's regime (data-bound / mixed /
model-bound). See docs/BENCHMARK.md and docs/VERIFICATION.md.
"""

__version__ = "1.1.0"
__author__ = "Mishel Adnan"
__license__ = "MIT"

from .container import OmniaContainer, FormatError, IntegrityError


def __getattr__(name):
    # The dataset classes need PyTorch; converting files does not. Import them
    # on first use so `import omnia_sdk` works on a machine without torch.
    if name in ("OmniaDataset", "ManifestOmniaDataset", "sweep_stale_mmap_dirs"):
        from . import dataset
        return getattr(dataset, name)
    raise AttributeError(f"module 'omnia_sdk' has no attribute {name!r}")


def convert(src, out=None, **kwargs):
    """Convert a medical image to .omnia. One call, sane defaults.

        import omnia_sdk
        omnia_sdk.convert("slide.svs")            # -> slide.omnia
        omnia_sdk.convert("scan.nii.gz")          # NIfTI volume
        omnia_sdk.convert("ct_series/")           # DICOM series or DICOM-WSI folder
        omnia_sdk.convert("slides/", "out/")      # every slide in a directory

    The format is detected automatically (see ``omnia_sdk.converters``).
    For whole-slide images the default is the training-speed profile: lossless
    Zstd, 20x level and below (``min_level=1``); pass ``min_level=0`` for full
    resolution or ``codec="jpeg"`` for a smaller lossy file.

    Returns the stats dict for one input, or a list of them for a directory of slides.
    """
    from pathlib import Path
    from .converters import convert_file, detect_format, WSI_SUFFIXES

    src = Path(src)
    if src.is_dir():
        slides = sorted(p for p in src.iterdir() if p.suffix.lower() in WSI_SUFFIXES)
        if slides:
            dest = Path(out) if out else src
            dest.mkdir(parents=True, exist_ok=True)
            kwargs.setdefault("min_level", 1)
            return [convert_file(p, dest / f"{p.stem}.omnia", fmt="whole-slide", **kwargs) for p in slides]
    fmt = detect_format(src)
    if fmt == "whole-slide":
        kwargs.setdefault("min_level", 1)
    return convert_file(src, out, fmt=fmt, **kwargs)

__all__ = [
    "convert",
    "OmniaContainer",
    "FormatError",
    "IntegrityError",
    "OmniaDataset",
    "ManifestOmniaDataset",
    "sweep_stale_mmap_dirs",
]
