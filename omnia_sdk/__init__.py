"""
omnia-20x — PyTorch-native container for whole-slide images.

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

__all__ = [
    "OmniaContainer",
    "FormatError",
    "IntegrityError",
    "OmniaDataset",
]
