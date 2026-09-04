"""
omnia-sdk — PyTorch-native container for whole-slide images.

Up to 20x faster data feeding than openslide for tile-based WSI training.
Lossless Zstd container, CRC-verified, zero-copy random access, no openslide
dependency at training time. Ships an auditable benchmark
(`python -m omnia_sdk.benchmark`) that measures both data-only and full-train
speedups live and classifies the machine's regime (data-bound / mixed /
model-bound). See docs/BENCHMARK.md and docs/VERIFICATION.md.
"""

__version__ = "1.0.0"
__author__ = "Mishel Adnan"
__license__ = "MIT"

from .container import OmniaContainer, FormatError, IntegrityError
from .dataset import OmniaDataset


def convert(svs, out=None, **kwargs):
    """Convert a whole-slide image to .omnia. One call, sane defaults.

        import omnia_sdk
        omnia_sdk.convert("slide.svs")            # -> slide.omnia
        omnia_sdk.convert("slides/", "out/")      # a whole directory

    Defaults are the training-speed profile: lossless Zstd, 20x level and
    below (`min_level=1`), which is what fits in RAM for preloading. Pass
    `min_level=0` for full resolution, or `codec="jpeg"` for a smaller file.

    Returns the stats dict for a single slide, or a list of them for a
    directory.
    """
    from pathlib import Path
    from .svs_to_omnia import svs_to_omnia

    src = Path(svs)
    kwargs.setdefault("min_level", 1)

    if src.is_dir():
        dest = Path(out) if out else src
        dest.mkdir(parents=True, exist_ok=True)
        slides = sorted(p for p in src.iterdir()
                        if p.suffix.lower() in (".svs", ".tif", ".tiff", ".ndpi"))
        if not slides:
            raise FileNotFoundError(f"No slide files found in {src}")
        return [svs_to_omnia(p, dest / f"{p.stem}.omnia", **kwargs) for p in slides]

    dest = Path(out) if out else src.with_suffix(".omnia")
    if dest.is_dir():
        dest = dest / f"{src.stem}.omnia"
    dest.parent.mkdir(parents=True, exist_ok=True)
    return svs_to_omnia(src, dest, **kwargs)

__all__ = [
    "convert",
    "OmniaContainer",
    "FormatError",
    "IntegrityError",
    "OmniaDataset",
]
